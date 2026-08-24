"""Streamlit UI for the estimator.

Streamlit acts as a pure HTTP client of the FastAPI service. It holds no LLM API
key and never calls a provider directly, the API owns the guardrails, the
caches, the prompt versioning and the provider fallback. Three tabs, three flows:

- **Structured estimate**: a typed ``EstimationRequest`` to
  ``POST /api/v1/estimate``, rendering the validated ``EstimationResult``.
- **Chat**: a free-text transcription streamed from
  ``POST /api/v1/estimate/stream`` and rendered token by token.
- **Project session**: a multi-turn conversation against
  ``POST /api/v1/sessions`` + ``POST /api/v1/sessions/{id}/estimate``, with
  optional PDF/Word attachments. Shows ``project_metadata`` explicitly so the
  separation between conversation history (windowed) and memory (durable) is
  visible, not just a backend implementation detail.

The prompt version is deliberately NOT surfaced: which template the service
runs is a deploy-time decision (``PROMPT_VERSION`` in settings), not something
the person asking for an estimate should have to reason about. It is still
logged and returned in the API response for traceability.
"""

from __future__ import annotations

import json
import time

import httpx
import streamlit as st

from app.config import get_settings
from app.context.examples import ESTIMATION_EXAMPLES, format_examples
from app.schemas.estimation import DetailLevel, OutputFormat, ProjectType
from app.services.llm_service import build_system_prompt

settings = get_settings()
API_BASE = settings.ESTIMATOR_API_BASE_URL.rstrip("/")
ESTIMATE_ENDPOINT = f"{API_BASE}/api/v1/estimate"
STREAM_ENDPOINT = f"{API_BASE}/api/v1/estimate/stream"
SESSIONS_ENDPOINT = f"{API_BASE}/api/v1/sessions"

MIN_DESCRIPTION_LENGTH = 20
MIN_TRANSCRIPTION_LENGTH = 50
MIN_SESSION_MESSAGE_LENGTH = 10

st.set_page_config(page_title="Software Estimator", page_icon="📊")
st.title("Software Estimator")
st.caption(
    "Describe a software project and get a phase-by-phase estimate with costs, "
    "duration and a confidence score."
)


def humanise(value: str) -> str:
    return value.replace("_", " ").capitalize()


def render_estimation_result(result: dict) -> None:
    """Shared rendering for an ``EstimationResult``: used by both the
    structured-estimate form and the project-session tab so a turn's answer
    always shows the same metrics + phases table, not just its summary text."""
    # The service normalises anything it could not size confidently into a
    # summary starting with "Out of scope:", show that as a warning rather
    # than dressing a non-estimate up as a real one.
    if result["summary"].startswith("Out of scope:"):
        st.warning(result["summary"])
        return

    st.success(result["summary"])

    left, middle, right = st.columns(3)
    left.metric("Total duration", f"{result['total_duration_weeks']} weeks")
    middle.metric("Total cost", f"€{result['total_cost_eur']:,}")
    right.metric("Confidence", f"{result['confidence_pct']}%")

    st.subheader("Phases")
    st.table(
        [
            {
                "Phase": phase["name"],
                "Duration (weeks)": phase["duration_weeks"],
                "Cost (EUR)": f"€{phase['cost_eur']:,}",
                "Detail": phase["summary"],
            }
            for phase in result["phases"]
        ]
    )


def describe_rejection(payload: object, status_code: int) -> str:
    """Turn an API error body into a message fit for the end user.

    - Guardrail rejections (400) arrive as ``{"reason": ..., "message": ...}``.
    - FastAPI validation errors (422) arrive as a list of technical dicts.
    - Anything else falls back to the raw text.
    """
    if isinstance(payload, dict) and "message" in payload:
        labels = {
            "prompt_injection": "That looks like an attempt to instruct the AI rather "
            "than describe a project.",
            "pii": "Please remove personal data from the description.",
            "moderation": "That content can't be processed.",
        }
        prefix = labels.get(str(payload.get("reason")), "")
        return f"{prefix} {payload['message']}".strip()
    if isinstance(payload, list):
        parts = []
        for err in payload:
            loc = err.get("loc", [])
            field = humanise(str(loc[-1])) if loc else "Input"
            parts.append(f"{field}: {err.get('msg', 'is invalid')}")
        return "Please check the form, " + "; ".join(parts) + "."
    return f"The service returned an error ({status_code})."


def request_estimation(payload: dict) -> dict:
    """POST to the estimate endpoint, raising RuntimeError with a friendly
    message on any rejection (400 guardrail, 422 validation, 502 upstream)."""
    response = httpx.post(
        ESTIMATE_ENDPOINT, json=payload, timeout=httpx.Timeout(180.0, connect=10.0)
    )
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(describe_rejection(detail, response.status_code))
    return response.json()


def stream_estimation(transcription: str, meta_holder: dict):
    """POST to the SSE endpoint and yield text chunks as they arrive.

    Per the SSE spec a message can carry multiple ``data:`` lines that must be
    joined with a newline, and a blank line terminates the message. The final
    ``meta`` event (cache hit, cost, model) is parsed into ``meta_holder``
    rather than yielded as chat text.
    """
    with httpx.stream(
        "POST",
        STREAM_ENDPOINT,
        json={"transcription": transcription},
        timeout=httpx.Timeout(180.0, connect=10.0),
        headers={"Accept": "text/event-stream"},
    ) as response:
        if response.status_code >= 400:
            # Read the body here, while the stream is still open: once this
            # `with` block exits httpx closes the stream and reading it
            # afterwards raises StreamClosed. Bake the message into a plain
            # exception rather than re-raising raise_for_status()'s.
            response.read()
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise RuntimeError(describe_rejection(detail, response.status_code))

        current_event = "token"
        data_lines: list[str] = []
        for raw_line in response.iter_lines():
            if raw_line == "":
                if data_lines:
                    payload_text = "\n".join(data_lines)
                    data_lines = []
                    if current_event == "token":
                        yield payload_text
                    elif current_event == "meta":
                        meta_holder.update(json.loads(payload_text))
                    elif current_event == "error":
                        yield f"\n\n[error] {payload_text}"
                    elif current_event == "done":
                        return
                current_event = "token"
                continue
            if raw_line.startswith("event:"):
                current_event = raw_line[6:].strip()
            elif raw_line.startswith("data:"):
                data_lines.append(
                    raw_line[6:] if raw_line.startswith("data: ") else raw_line[5:]
                )


def create_session() -> str:
    response = httpx.post(SESSIONS_ENDPOINT, timeout=httpx.Timeout(30.0, connect=10.0))
    response.raise_for_status()
    return response.json()["session_id"]


def request_session_estimation(
    session_id: str,
    transcript: str,
    project_type: ProjectType,
    detail_level: DetailLevel,
    output_format: OutputFormat,
    files: list,
) -> dict:
    """POST one turn to the session endpoint as multipart/form-data. The
    typed selectors are sent every turn for simplicity, but are genuinely
    optional server-side, the session remembers the last value it saw."""
    data = {
        "transcript": transcript,
        "project_type": project_type.value,
        "detail_level": detail_level.value,
        "output_format": output_format.value,
    }
    upload_files = [
        ("attachments", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in files
    ]
    response = httpx.post(
        f"{SESSIONS_ENDPOINT}/{session_id}/estimate",
        data=data,
        files=upload_files,
        timeout=httpx.Timeout(180.0, connect=10.0),
    )
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(describe_rejection(detail, response.status_code))
    return response.json()


form_tab, chat_tab, session_tab = st.tabs(["Structured estimate", "Chat", "Project session"])

with form_tab:
    with st.form("estimation_form"):
        description = st.text_area(
            "Project description",
            height=200,
            placeholder="Describe the project: goals, key features, constraints…",
            help=f"At least {MIN_DESCRIPTION_LENGTH} characters.",
        )
        project_type = st.selectbox(
            "Project type", options=list(ProjectType), format_func=lambda v: humanise(v.value)
        )
        detail_level = st.radio(
            "Detail level",
            options=list(DetailLevel),
            index=1,
            horizontal=True,
            format_func=lambda v: humanise(v.value),
        )
        output_format = st.selectbox(
            "Output format", options=list(OutputFormat), format_func=lambda v: humanise(v.value)
        )
        submitted = st.form_submit_button("Generate estimation", type="primary")

    if submitted:
        if len(description.strip()) < MIN_DESCRIPTION_LENGTH:
            st.error(
                f"The description is too short ({len(description.strip())} characters). "
                f"Please describe the project in at least {MIN_DESCRIPTION_LENGTH} characters."
            )
        else:
            started = time.perf_counter()
            try:
                with st.spinner("Estimating…"):
                    body = request_estimation(
                        {
                            "description": description.strip(),
                            "project_type": project_type.value,
                            "detail_level": detail_level.value,
                            "output_format": output_format.value,
                        }
                    )
            except RuntimeError as exc:
                st.error(str(exc))
            except httpx.HTTPError as exc:
                st.error(f"Could not reach the estimator at `{ESTIMATE_ENDPOINT}`: {exc}")
            else:
                elapsed = round(time.perf_counter() - started, 2)
                render_estimation_result(body["result"])

                st.session_state.last_call = {
                    "elapsed": elapsed,
                    "cached": body.get("cached", False),
                    **body.get("meta", {}),
                }

with chat_tab:
    st.caption(
        "Paste a client-meeting transcription and get a free-text estimation "
        "streamed token by token. Same guardrails and cache as the form."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(
        f"Describe your project (min {MIN_TRANSCRIPTION_LENGTH} characters)…"
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()

            # Fast client-side check: no point round-tripping to the API (and
            # eventually the LLM) for something we already know is too short.
            if len(prompt) < MIN_TRANSCRIPTION_LENGTH:
                answer = (
                    f"Your message is too short ({len(prompt)} characters). Please "
                    f"describe your project in at least {MIN_TRANSCRIPTION_LENGTH} "
                    "characters so there's enough to estimate."
                )
                placeholder.error(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.stop()

            answer = ""
            meta_holder: dict = {}
            started = time.perf_counter()
            try:
                for chunk in stream_estimation(prompt, meta_holder):
                    answer += chunk
                    placeholder.markdown(answer + "▍")
                placeholder.markdown(answer)
            except RuntimeError as exc:
                answer = str(exc)
                placeholder.error(answer)
            except httpx.HTTPError as exc:
                answer = f"Could not reach the estimator at `{STREAM_ENDPOINT}`: {exc}"
                placeholder.error(answer)
            elapsed = round(time.perf_counter() - started, 2)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.last_call = {
            "elapsed": elapsed,
            "cached": meta_holder.get("cache_hit", False),
            "cost_usd": meta_holder.get("cost_usd", 0.0),
            "model": meta_holder.get("model"),
            "provider": meta_holder.get("provider"),
        }
        st.rerun()

with session_tab:
    st.caption(
        "Multi-turn estimation: refine the same project across several messages. "
        "project_metadata (the durable facts, shown below) survives even after an "
        "old turn falls out of the conversation window."
    )

    if "session_id" not in st.session_state:
        try:
            st.session_state.session_id = create_session()
        except httpx.HTTPError as exc:
            st.session_state.session_id = None
            st.error(f"Could not reach the estimator at `{SESSIONS_ENDPOINT}`: {exc}")
    st.session_state.setdefault("session_turns", [])
    st.session_state.setdefault("session_metadata", None)
    st.session_state.setdefault("session_attachments", [])
    st.session_state.setdefault(
        "session_selectors",
        {
            "project_type": ProjectType.WEB_SAAS,
            "detail_level": DetailLevel.MEDIUM,
            "output_format": OutputFormat.PHASES_TABLE,
        },
    )

    session_header_left, session_header_right = st.columns([3, 1])
    with session_header_left:
        if st.session_state.session_id:
            st.caption(f"Session: `{st.session_state.session_id}`")
    with session_header_right:
        if st.button("Nueva conversación"):
            try:
                st.session_state.session_id = create_session()
                st.session_state.session_turns = []
                st.session_state.session_metadata = None
                st.session_state.session_attachments = []
            except httpx.HTTPError as exc:
                st.error(f"Could not reach the estimator at `{SESSIONS_ENDPOINT}`: {exc}")
            st.rerun()

    for turn in st.session_state.session_turns:
        with st.chat_message("user"):
            st.markdown(turn["transcript"])
        with st.chat_message("assistant"):
            render_estimation_result(turn["result"])

    selectors = st.session_state.session_selectors
    with st.form("session_estimation_form", clear_on_submit=True):
        transcript = st.text_area(
            "Message",
            height=150,
            placeholder="Describe the project, or refine what you already discussed…",
            help=f"At least {MIN_SESSION_MESSAGE_LENGTH} characters. The selectors below are "
            "optional, set once, remembered for later turns unless you change them.",
        )
        sel_left, sel_mid, sel_right = st.columns(3)
        with sel_left:
            project_type = st.selectbox(
                "Project type",
                options=list(ProjectType),
                index=list(ProjectType).index(selectors["project_type"]),
                format_func=lambda v: humanise(v.value),
            )
        with sel_mid:
            detail_level = st.selectbox(
                "Detail level",
                options=list(DetailLevel),
                index=list(DetailLevel).index(selectors["detail_level"]),
                format_func=lambda v: humanise(v.value),
            )
        with sel_right:
            output_format = st.selectbox(
                "Output format",
                options=list(OutputFormat),
                index=list(OutputFormat).index(selectors["output_format"]),
                format_func=lambda v: humanise(v.value),
            )
        uploaded_files = st.file_uploader(
            "Attachments (PDF or Word)", type=["pdf", "docx"], accept_multiple_files=True
        )
        session_submitted = st.form_submit_button("Send", type="primary")

    if session_submitted:
        if not st.session_state.session_id:
            st.error("No active session. Reload the page or click 'Nueva conversación'.")
        elif len(transcript.strip()) < MIN_SESSION_MESSAGE_LENGTH:
            st.error(f"Message is too short (min {MIN_SESSION_MESSAGE_LENGTH} characters).")
        else:
            st.session_state.session_selectors = {
                "project_type": project_type,
                "detail_level": detail_level,
                "output_format": output_format,
            }
            started = time.perf_counter()
            try:
                with st.spinner("Estimating…"):
                    body = request_session_estimation(
                        st.session_state.session_id,
                        transcript.strip(),
                        project_type,
                        detail_level,
                        output_format,
                        uploaded_files or [],
                    )
            except RuntimeError as exc:
                st.error(str(exc))
            except httpx.HTTPError as exc:
                st.error(f"Could not reach the estimator at `{SESSIONS_ENDPOINT}`: {exc}")
            else:
                elapsed = round(time.perf_counter() - started, 2)
                st.session_state.session_turns.append(
                    {"transcript": transcript.strip(), "result": body["result"]}
                )
                st.session_state.session_metadata = body["project_metadata"]
                st.session_state.session_attachments = body.get("attachments", [])
                st.session_state.last_call = {
                    "elapsed": elapsed,
                    "cached": body.get("cached", False),
                    **body.get("meta", {}),
                }
                st.rerun()

    if st.session_state.session_attachments:
        st.subheader("Attachments (last turn)")
        st.table(st.session_state.session_attachments)
        for attachment in st.session_state.session_attachments:
            if attachment.get("method") == "too_long":
                st.warning(f"**{attachment['filename']}**: {attachment.get('note')}")

    with st.expander("project_metadata, durable facts, separate from history", expanded=True):
        if st.session_state.session_metadata:
            st.json(st.session_state.session_metadata)
        else:
            st.caption("No facts recorded yet, this is a new session.")

with st.sidebar:
    st.header("Service")
    st.code(ESTIMATE_ENDPOINT, language="text")
    st.code(STREAM_ENDPOINT, language="text")
    st.markdown(f"**Primary model:** `{settings.PRIMARY_MODEL}`")
    st.markdown(f"**Fallback model:** `{settings.FALLBACK_MODEL}`")
    st.markdown(f"**Cache TTL:** `{settings.CACHE_TTL}s`")

    st.header("CAG Context")
    with st.expander("Chat system prompt"):
        st.text_area(
            "Chat system prompt",
            value=build_system_prompt(),
            height=260,
            disabled=True,
            label_visibility="collapsed",
        )
    with st.expander("Injected examples"):
        st.text_area(
            "Injected examples",
            value=format_examples(ESTIMATION_EXAMPLES),
            height=260,
            disabled=True,
            label_visibility="collapsed",
        )

    st.header("Last Call")
    last_call = st.session_state.get("last_call")
    if last_call:
        st.metric("Response time (s)", f"{last_call['elapsed']:.2f}")
        st.metric("Served from cache", "Yes" if last_call.get("cached") else "No")
        # Costs are fractions of a cent : st.metric would round 0.000385 to
        # "$0.00", so format explicitly. A cache hit genuinely costs nothing.
        st.metric("Cost (USD)", f"${last_call.get('cost_usd', 0.0):.6f}")
        if last_call.get("model"):
            st.caption(
                f"Answered by `{last_call['model']}` "
                f"({last_call.get('provider', 'unknown')})"
            )
    else:
        st.caption("No estimations yet.")

# CAG Estimator

![Python](https://img.shields.io/badge/python-%3E%3D3.12-blue)
![FastAPI](https://img.shields.io/badge/framework-FastAPI-009688)
![CAG](https://img.shields.io/badge/architecture-CAG-purple)
![Version](https://img.shields.io/badge/version-0.5.0-lightgrey)
![uv](https://img.shields.io/badge/package%20manager-uv-de5fe9)
![Docker](https://img.shields.io/badge/docker-ready-2496ED)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![License](https://img.shields.io/badge/license-MIT-green)

API for software project estimation using LLMs, based on a CAG (Context-Augmented Generation) architecture: historical estimation examples are injected into the prompt to guide the model toward more accurate and consistent budgets.

A typed request (`description` + project type / detail level / output format) maps to a typed, validated `EstimationResult`, phases, totals and a confidence score, via [Instructor](https://python.useinstructor.com/) + Pydantic. The whole pipeline lives in a single service class, `app/services/estimation.py`:

1. **Input guardrails**, moderation, prompt-injection regex, PII heuristics.
2. **Exact-match cache** lookup (Redis).
3. **Semantic cache** lookup (embedding similarity, Redis Stack + redisvl).
4. **Render** the versioned Jinja2 prompt.
5. **LLM call** via Instructor, re-prompting on validator failures.
6. **Output guardrail**, normalise low-confidence answers into an explicit "Out of scope:" response.
7. **Write both caches**, only after the output passed validation.

Order matters: guardrails run *before* any cache so a malicious or PII description can never be served from cache, and both cache writes happen *after* output validation so a bad estimation is never persisted and replayed for the whole TTL.

Every LLM call goes through a [LiteLLM](https://docs.litellm.ai/)-backed wrapper that adds:
- **Provider fallback**, tries `PRIMARY_MODEL` first (OpenAI by default), retries it, then falls over to `FALLBACK_MODEL` (Anthropic by default) so an expired key, rate limit or outage doesn't take the feature down. Structured calls go through the same Router, so they keep the fallback guarantee.
- **Cost tracking**, every response reports the model, provider, latency and USD cost of that call. A cache hit reports `cost_usd: 0.0`, because nothing was spent.
- **Structured logging**, every phase of a call (guardrails, cache check, prompt render, dispatch, retry, fallback, success/failure) is logged via `structlog`, so a request can be traced end-to-end.

A third pipeline adds multi-turn memory on top of the structured one: `app/sessions/` holds an in-memory `Session` per `session_id`, a sliding-window `ConversationHistory` (last `MAX_TURNS` turns) plus a separate, never-truncated `ProjectMetadata` (project name, team size, technologies, scope, constraints, rejected options). `EstimationService.estimate_in_session` renders a dedicated conversational prompt (`v2`, with a `<project_metadata>` block) and replays the session's history; after the LLM call, it appends the turn and refreshes `ProjectMetadata` through `app/sessions/metadata_extractor.py`, which asks a small dedicated model for only what changed this turn and merges it deterministically with what's already known, see "Updating `project_metadata`" below for why the merge is Python, not the LLM's job. Attachments (PDF/Word) are extracted locally, length-capped, and folded into that turn's prompt.

Streamlit is a thin HTTP client of the API (not a second place that talks to the LLM), with three tabs: a **structured estimate** form that POSTs an `EstimationRequest` and renders the validated result, a **chat** that streams a free-text estimation from the SSE endpoint, and a **project session** tab for the multi-turn flow with file uploads and a live `project_metadata` panel.

## Project Structure
```
cag-estimator/
├── app/
│   ├── main.py             -- FastAPI app entrypoint
│   ├── config.py           -- Settings loaded from .env
│   ├── constants.py        -- Model pricing table
│   ├── dependencies.py     -- Shared singletons (caches, wrapper, service)
│   ├── routers/
│   │   ├── estimations.py      -- POST /api/v1/estimate (HTTP error mapping only)
│   │   ├── estimations_text.py -- Free-text endpoints: /estimate/text, /estimate/stream
│   │   └── sessions.py         -- POST /sessions, POST /sessions/{id}/estimate
│   ├── services/
│   │   ├── estimation.py       -- EstimationService: structured pipeline + session turns
│   │   ├── llm_service.py      -- Free-text prompt building + orchestration
│   │   ├── evaluation.py       -- Static structural scoring of free-text output
│   │   ├── llm_wrapper.py      -- LiteLLM fallback, Instructor, streaming, cost
│   │   └── cache.py            -- Exact-match Redis cache
│   ├── cache/
│   │   └── semantic.py         -- Vector-similarity cache (redisvl + Redis Stack)
│   ├── guardrails/
│   │   ├── input.py            -- Moderation, prompt injection, PII (exception policy)
│   │   └── output.py           -- enforce_scope_response (filter policy)
│   ├── attachments/
│   │   └── extraction.py       -- PDF/Word local text extraction, self-validating, length cap
│   ├── sessions/
│   │   ├── models.py           -- Session, ConversationHistory, ProjectMetadata (+ merge_with), AttachmentReport
│   │   ├── store.py             -- SessionStore (in-memory dict)
│   │   └── metadata_extractor.py -- update_metadata(): LLM delta extraction + deterministic merge
│   ├── prompts/
│   │   ├── loader.py            -- render_estimation_prompt / render_session_prompt / render_metadata_extraction_prompt
│   │   ├── estimation/v1/       -- system.j2, user.j2, metadata_extraction.j2, examples.j2 (single-shot)
│   │   └── estimation/v2/       -- system.j2, user.j2 (conversational: adds <project_metadata>)
│   ├── context/
│   │   └── examples.py         -- CAG reference examples for the free-text flow
│   └── schemas/
│       ├── estimation.py       -- Structured: Request, Draft, Result, Response
│       └── estimations.py      -- Free-text: transcription in, Markdown + evaluation out
├── streamlit_app.py         -- Streamlit UI: structured form + streaming chat + project session
├── tests/
├── .env.example
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## Requirements
- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)
- **Redis Stack**, the semantic cache needs the RediSearch module for vector queries. `docker compose up` starts `redis/redis-stack` automatically. On vanilla `redis:7-alpine` the app still runs, but the semantic cache disables itself at startup (logged as `semantic_cache_disabled`) and only the exact-match layer works.
- At least one of `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`, the app validates this at startup and refuses to boot without one. `OPENAI_API_KEY` is additionally required for the moderation guardrail and the semantic cache's embeddings.

## Setup
```bash
uv sync
cp .env.example .env   # then fill in your API key(s)
```

## Running the API
```bash
uv run uvicorn app.main:app --reload
```
Available at `http://127.0.0.1:8000`, interactive docs at `/docs`. Needs Redis reachable at `REDIS_URL`, start one with `docker run -p 6379:6379 redis/redis-stack:7.4.0-v0` if you're not using `docker compose`.

### Running with Docker
```bash
docker compose up --build
```
Starts the API plus a Redis Stack service (RedisInsight UI on port 8001) and wires them together.

## Running the Streamlit UI
```bash
uv run streamlit run streamlit_app.py
```
It talks to the API over HTTP at `ESTIMATOR_API_BASE_URL` and holds no LLM API key. Three tabs: **Structured estimate** (the form), **Chat** (free-text, streamed), and **Project session** (multi-turn, with attachments and a `project_metadata` panel). The sidebar shows the configured models and cache TTL, the chat's system prompt and injected CAG examples (read-only, for visibility), and for the last call: response time, whether it was served from cache, the USD cost, and which model answered.

## Environment Variables
See [`.env.example`](.env.example) for the full list. Beyond the API keys and models:

| Variable | Default | Purpose |
|----------|---------|---------|
| `PROMPT_VERSION` | `v1` | Which template under `app/prompts/estimation/` serves the single-shot endpoint |
| `CONVERSATIONAL_PROMPT_VERSION` | `v2` | Which template serves the session endpoint (has its own `<project_metadata>` block) |
| `CACHE_TTL` | `86400` | Exact-match cache TTL, in seconds |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embeddings for the semantic cache |
| `SEMANTIC_CACHE_THRESHOLD` | `0.85` | Minimum cosine similarity to serve a semantic hit |
| `SEMANTIC_CACHE_LOG_ONLY` | `false` | Log would-be hits without serving them (threshold calibration) |
| `MAX_TURNS` | `6` | Sliding-window size (in turns) for a session's conversation history |
| `MAX_ATTACHMENT_WORDS` | `8000` | Word-count cap per attachment; over this it's reported as failed, not truncated |
| `METADATA_EXTRACTOR_MODEL` | `gpt-4o-mini` | Primary model for the metadata-extraction side call |
| `METADATA_EXTRACTOR_FALLBACK_MODEL` | `claude-haiku-4-5-20251001` | Its fallback, via the same Router mechanism as the main call |

## Endpoints

| Method | Path                                | Description                                             |
|--------|-------------------------------------|-----------------------------------------------------------|
| GET    | `/health`                           | Health check                                            |
| POST   | `/api/v1/estimate`                  | Structured estimation (typed in, validated schema out)  |
| POST   | `/api/v1/estimate/text`             | Free-text estimation (transcription in, Markdown out)   |
| POST   | `/api/v1/estimate/stream`           | Free-text estimation streamed over Server-Sent Events   |
| POST   | `/api/v1/sessions`                  | Create a multi-turn session, returns `{"session_id": ...}` |
| POST   | `/api/v1/sessions/{id}/estimate`    | One turn of a multi-turn estimation, with optional attachments |

The structured endpoint is the main one: typed request, validated `EstimationResult`, both cache layers. The two free-text endpoints take a raw meeting transcription and return Markdown, they share the same input guardrails, wrapper and exact-match cache, but not the semantic cache (whose bucket is keyed on the typed form options a free-text request doesn't have). `/estimate/text` also runs a static structural evaluation of the generated Markdown (`app/services/evaluation.py`); `/estimate/stream` is what the Streamlit chat consumes.

### `POST /api/v1/estimate`

**Request body:**
```json
{
  "description": "A small B2B SaaS to manage employee equipment loans across teams, with role-based access for HR and IT.",
  "project_type": "web_saas",
  "detail_level": "medium",
  "output_format": "phases_table"
}
```
`project_type`: `mobile_app` | `web_saas` | `internal_tool` | `data_pipeline`. `detail_level`: `summary` | `medium` | `detailed`. `output_format`: `phases_table` | `line_items` | `narrative`.

**Response:**
```json
{
  "result": {
    "summary": "...",
    "confidence_pct": 80,
    "phases": [
      {"name": "Discovery", "duration_weeks": 1, "cost_eur": 2500, "summary": "Workshops and scoping."}
    ],
    "total_duration_weeks": 9,
    "total_cost_eur": 24500
  },
  "prompt_version": "v1",
  "cached": false,
  "meta": {"model": "gpt-4o-mini", "provider": "openai", "cost_usd": 0.000446, "latency_ms": 2607}
}
```

**Errors:**

| Status | When |
|--------|------|
| `400`  | An input guardrail rejected the request. Body is `{"detail": {"reason": "prompt_injection" \| "pii" \| "moderation", "message": "..."}}` |
| `422`  | The request body failed schema validation (missing field, bad enum, description too short) |
| `502`  | The upstream LLM call failed, including Instructor exhausting its retries |

An off-topic or unsizeable description is **not** an error: it returns `200` with a `summary` starting in `"Out of scope:"`, `confidence_pct` below 30, and a single `Not estimated` phase. The UI renders that as a warning rather than a fake estimate.

### `POST /api/v1/estimate/text` and `/api/v1/estimate/stream`

**Request body** (both): `{"transcription": "Client meeting summary describing the project..."}`, minimum 50 characters.

`/estimate/text` responds with the Markdown estimation plus usage, cost, `cache_hit`, and an `evaluation` object scoring the output's structure (does it have the breakdown table, do the declared totals match the summed rows, was it cut off). `/estimate/stream` responds with `text/event-stream`: an `event: token` per chunk, then `event: meta` (cache hit, cost, model) and `event: done`, or `event: error` if the call fails after retries and fallback.

### Totals are computed in Python, not by the LLM

The model fills an `EstimationDraft` (summary, confidence, phases) and the service sums the phases to produce `total_duration_weeks` / `total_cost_eur`. This is a measured decision, not a stylistic one: asking `gpt-4o-mini` for a grand total alongside the phases failed every one of Instructor's 7 attempts in a live run (~24k tokens burned, drifting further each retry) before the request 502'd. Field ordering and an explicit "compute the sum" instruction were not enough. `EstimationResult.phases_sum_matches_total` is kept as a genuine invariant assertion over the computed value.

### Prompt versions

Prompts live under `app/prompts/estimation/<version>/` and are rendered by `app/prompts/loader.py`. There are two independent version settings: `PROMPT_VERSION` (default `v1`) for the single-shot endpoint's `system.j2`/`user.j2`/`examples.j2`, and `CONVERSATIONAL_PROMPT_VERSION` (default `v2`) for the session endpoint's own `system.j2`/`user.j2`, `v2` adds the `<project_metadata>` block and a "this is a multi-turn conversation" framing that `v1` deliberately does not carry, so the single-shot flow's prompt never has to reason about a concept (project memory) that doesn't apply to it. `v2/system.j2` reuses `v1/examples.j2` via `{% include %}` rather than duplicating the few-shot examples. Both versions are deploy-time decisions, not client-supplied parameters, so rolling out a new prompt is a config change and every response echoes back the version that produced it. Each render logs a `prompt_rendered` event with the version and a content hash (not the text, so descriptions stay out of the logs).

### `POST /api/v1/sessions` and `POST /api/v1/sessions/{session_id}/estimate`

Multi-turn estimation: refine the same project across several messages instead of one blocking call.

`POST /api/v1/sessions` takes no body and returns `{"session_id": "<uuid4>"}`. Sessions live in a process-local dict (`app/sessions/store.py`), no database, no Redis. That volatility is accepted deliberately: a session is short-lived working memory for one active conversation, not a system of record, and this phase is about the CAG architecture (separating history from memory), not about surviving a process restart. There's no `GET /sessions/{id}` debug endpoint: with the session in-memory only (no persistence, nothing to inspect after a restart) and `project_metadata` already returned inline on every `POST .../estimate` response, a separate read endpoint wouldn't expose anything the client can't already see.

`POST /api/v1/sessions/{session_id}/estimate` is `multipart/form-data`:

| Field | Required | Notes |
|-------|----------|-------|
| `transcript` | yes | The turn's message, 10–80,000 chars |
| `project_type`, `detail_level`, `output_format` | no | Same enums as `/estimate`. Optional per turn but **remembered**: set once (or defaulted on session creation to `web_saas` / `medium` / `phases_table`) and only overwritten when a turn explicitly supplies a new value, the multi-turn flow keeps the "typed request, not free chat" contract instead of dropping it |
| `attachments` | no | Zero or more PDF/Word files |

**Response**, `SessionEstimateResponse`: the same `result` / `prompt_version` / `cached` (always `false`) / `meta` shape as `/estimate`, plus:
```json
{
  "session_id": "...",
  "project_metadata": {
    "project_name": "Aurora",
    "assumed_team_size": 3,
    "mentioned_technologies": ["React", "PostgreSQL"],
    "agreed_scope": null,
    "explicit_constraints": [],
    "rejected_options": []
  },
  "attachments": [
    {"filename": "spec.pdf", "method": "pypdf", "ok": true, "note": null}
  ]
}
```

**Why no caching here.** Both cache layers are skipped for this endpoint. The exact-match cache is keyed on the description alone, but the same transcript text means something different depending on the session's history and `project_metadata`, serving a cached answer would silently ignore the conversation. Making it cache-safe would mean folding the entire history + metadata into the key, which defeats the point of reusing anything.

**History vs. memory.** `ConversationHistory` (`app/sessions/models.py`, a Pydantic model) keeps a sliding window of the last `MAX_TURNS` `(user, assistant)` pairs, appended via `.append(user=..., assistant=...)` and trimmed automatically. The system prompt is rebuilt fresh each turn from the current `ProjectMetadata` and prepended by the caller, never stored in the window, so "always preserve the system prompt" is automatic rather than a truncation special case. The assistant side of each turn stores the estimation's **full JSON** (`EstimationResult.model_dump_json()`), not a hand-written summary, so nothing about a past turn's answer is lossy-compressed before it even leaves the window.

`ProjectMetadata` is a separate, never-truncated Pydantic model (project name, team size, technologies, agreed scope, constraints, rejected options) injected into the `v2` system prompt's `<project_metadata>` block on every turn. A fact from turn 1 still informs turn 20 even after turn 1 has scrolled out of the window, because it lives on the session, not inside the window, this is what makes a plain sliding window a reasonable default instead of needing summarization from day one.

**Updating `project_metadata`: an LLM delta + a deterministic Python merge.** After each turn, `app/sessions/metadata_extractor.py`'s `update_metadata()` asks a small, dedicated model (`METADATA_EXTRACTOR_MODEL`, with its own fallback via the Router, see below) for **only what's new or changed this turn**, not the full object. `ProjectMetadata.merge_with()` then combines that with what's already known: scalar fields are overwritten only when the update provides a non-null value, and list fields (`mentioned_technologies`, `explicit_constraints`, `rejected_options`) are unioned rather than replaced. This split matters because an LLM is non-deterministic: if the extractor call forgot to restate `project_name` on some turn, asking it for "the full updated object" would silently erase it. Asking for a delta and merging in Python means a forgotten field just falls back to the previous value, it can't regress. Both the extractor prompt and this merge were chosen and validated against exactly that failure mode; a regex heuristic was the cheaper alternative but doesn't hold up against free-form, bilingual (ES/EN) input as well as an LLM delta does. The extractor call fails open: if it errors after retries, `update_metadata` logs it and returns the previous metadata unchanged, so a flaky side call never turns a successful estimation into a 502.

**A dedicated, cheaper model for extraction, with fallback of its own.** The metadata call runs against its own Router group (`"metadata_extractor"` in `LLMWrapper`), configured via `METADATA_EXTRACTOR_MODEL` / `METADATA_EXTRACTOR_FALLBACK_MODEL`, independent of `PRIMARY_MODEL` / `FALLBACK_MODEL`. This is a deliberate choice not to reuse the (potentially larger/pricier) main model for a small side call, but unlike routing that call around the Router entirely (which would drop the fallback guarantee, the same trade-off flagged for the main call in `llm_wrapper.py`), it still goes through `LLMWrapper.complete_structured_with_messages(..., model_name="metadata_extractor")`, so a flaky provider on the cheap model still fails over instead of just failing.

**Attachments: local extraction, self-validating, with an LLM fallback and a length cap.** PDFs and Word docs are parsed locally (`app/attachments/extraction.py`) rather than uploaded to a provider's Files API, to keep the estimator independent of whichever model the Router happens to be using for a given call. The pipeline for each file:

1. Try `pypdf` (PDF) or `python-docx` (Word).
2. Validate the result against hard rules: non-empty, a minimum characters-per-page floor (catches a scanned/image-only PDF with no text layer), a minimum printable-character ratio (catches a garbled/mis-decoded extraction).
3. If a PDF fails validation, retry with `PyMuPDF`, which occasionally recovers text `pypdf` misses.
4. If it still fails, fall back to the LLM's own multimodal document support for that one file, the raw bytes go in as an inline `file` content block alongside the turn's text. If that call itself fails (unsupported model, provider quirk), the request degrades to a text-only retry noting the file couldn't be processed, rather than 502ing an otherwise-answerable turn.
5. If extraction *succeeds* but the result is longer than `MAX_ATTACHMENT_WORDS`, it's rejected anyway, not truncated. A silently truncated attachment produces a plausible-looking but partial answer with no indication anything was cut; reporting it as failed (`method: "too_long"`) tells the user outright. The message is deliberately non-technical, word counts and a multiple-of-the-limit ("~3.2x over the limit"), never "tokens", since that's a number a non-technical user can actually reason about. An oversized file is never routed to step 4's multimodal fallback either: the problem there is cost/context budget, and re-sending the same content as a raw file would make that worse, not better.

This keeps the common case (born-digital PDFs, Word docs) fully local while still reaching an answer for a scanned document, without ever letting attachment handling take down the whole request. Each attachment's outcome (`pypdf` / `pymupdf` / `docx` / `llm_fallback` / `too_long` / `failed`) is reported back in the response so the client can show what happened; Streamlit's Project session tab surfaces a `too_long` one as an explicit warning.

## Testing
```bash
uv run pytest
```
The suite is offline: template rendering, schema validators, guardrail regexes, and the endpoints with either the service or just the LLM call faked out. No API keys or Redis needed.

The session tests are split by concern, mirroring the four things Session 05 asks to verify independently:

- `tests/test_sessions_models.py`, pure unit tests, no FastAPI, no fakes: `ConversationHistory` window trimming and role alternation, `ProjectMetadata.is_empty()`, and `ProjectMetadata.merge_with()`'s scalar-overwrite/list-union semantics (the exact mechanism that keeps a fact from regressing if the extractor forgets to restate it).
- `tests/test_sessions_metadata.py`, integration: metadata accumulates correctly across two turns of the same session.
- `tests/test_sessions_attachments.py`, integration: a PDF's content reaches the LLM and changes the estimate; an oversized attachment is reported (`method: "too_long"`) without blocking the request.
- `tests/test_sessions_window.py`, integration: session creation, 404 on an unknown session id, and the sliding window never exceeding `MAX_TURNS` in what's actually sent to the LLM.

The three integration files share `tests/_session_test_helpers.py` (`FakeLLMWrapper` + a real-PDF builder via PyMuPDF), only `LLMWrapper` is faked, not the whole service, so the session/history/metadata/attachment logic itself runs for real, including parsing an actual generated PDF back with pypdf to check that attachment content really reaches the prompt.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "description": "A small B2B SaaS to manage employee equipment loans across teams.",
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "phases_table"
  }'
```

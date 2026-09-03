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

A third pipeline adds multi-turn memory on top of the structured one: `app/sessions/` holds an in-memory `Session` per `session_id`, a sliding-window `ConversationHistory` (last `MAX_TURNS` turns) plus a separate, never-truncated `ProjectMetadata` (project name, team size, technologies, scope, constraints, rejected options). `EstimationService.estimate_in_session` renders a dedicated conversational prompt (`v3`, with a `<project_metadata>` block and an `<audience>` block) and replays the session's history; after the LLM call, it appends the turn, runs `app/sessions/compression` (Session 5: rescue durable-commitment turns as anchors, fold the rest into a running summary, then trim the window), and refreshes `ProjectMetadata` through `app/sessions/metadata_extractor.py`, which asks a small dedicated model for only what changed this turn and merges it deterministically with what's already known, see "Updating `project_metadata`" below for why the merge is Python, not the LLM's job. A rule chain in `app/sessions/tier_resolver.py` also resolves an audience tier (executive / pm / developer / default) from the transcript + metadata each turn, echoed back on the response. Attachments (PDF/Word) are extracted locally, length-capped, and folded into that turn's prompt.

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
│   │   ├── sessions.py         -- POST /sessions, GET /sessions/{id}, POST /sessions/{id}/estimate
│   │   └── indexing.py         -- POST /index/run, GET /index/runs/{id}, GET /index/audit, POST /query
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
│   │   ├── models.py           -- Session, ConversationHistory (+ anchors, summary), ProjectMetadata (+ merge_with), AttachmentReport
│   │   ├── store.py             -- SessionStore (in-memory dict)
│   │   ├── metadata_extractor.py -- update_metadata(): LLM delta extraction + deterministic merge
│   │   ├── tier_resolver.py     -- resolve_tier(): rule chain deriving the audience tier at runtime
│   │   └── compression/
│   │       ├── anchors.py       -- AnchorDetector: heuristic (regex, EN+ES) or LLM-based
│   │       ├── summarizer.py    -- CumulativeSummarizer: folds evicted turns into a running summary
│   │       └── policy.py        -- CompressionPolicy: trims the window, promotes anchors, calls the summarizer
│   ├── prompts/
│   │   ├── loader.py            -- render_estimation_prompt / render_session_prompt / render_metadata_extraction_prompt / render_conversation_summary_prompt
│   │   ├── estimation/v1/       -- system.j2, user.j2, metadata_extraction.j2, examples.j2 (single-shot)
│   │   ├── estimation/v2/       -- system.j2, user.j2 (conversational: adds <project_metadata>)
│   │   ├── estimation/v3/       -- system.j2, user.j2 (conversational: adds <audience>, the resolved tier)
│   │   └── conversation_summary/v1/ -- system.j2, user.j2 (the cumulative-summary pass)
│   ├── ingest/                  -- Offline ingestion subsystem (Session 6)
│   │   ├── catalog.py          -- Typed data_catalog.yaml: what the pipeline is allowed to process
│   │   ├── census.py           -- Facts-only inspection CLI + catalog-vs-disk drift check
│   │   ├── architecture.py     -- CAG vs RAG decision, with this project's measured numbers
│   │   ├── models.py           -- ParsedUnit (parser output) and Document (canonical contract)
│   │   ├── loaders/            -- How to reach the bytes (filesystem; Drive/S3 would slot in here)
│   │   ├── parsers/            -- One per format: json, txt, docx, pdf, xlsx (+ shared extraction)
│   │   ├── cleaning/           -- Normalisation (pandas) + validation (Pandera) + failure routing
│   │   ├── anonymization/      -- Presidio + Faker pseudonymisation, HMAC-keyed mapping store
│   │   ├── normalizers/        -- ParsedUnit -> Document, deterministic ids
│   │   ├── orchestrator.py     -- Runs the whole offline pipeline per catalog source
│   │   └── audit.py            -- Renders the catalog as a human-readable audit report
│   ├── context/
│   │   └── examples.py         -- CAG reference examples for the free-text flow
│   └── schemas/
│       ├── estimation.py       -- Structured: Request, Draft, Result, Response
│       └── estimations.py      -- Free-text: transcription in, Markdown + evaluation out
├── streamlit_app.py         -- Streamlit UI: structured form + streaming chat + project session
├── data/
│   ├── build_corpus.py      -- Deterministic generator for the synthetic corpus
│   ├── data_catalog.yaml    -- The audited source catalog (the pipeline obeys it)
│   ├── AUDIT_REPORT.md      -- Generated from the catalog; not edited by hand
│   └── corpus/              -- The corpus itself: budgets, transcripts, proposals, contracts, rate card
├── evals/
│   ├── metrics.py           -- MetricResult, Metric protocol, run_all_metrics
│   └── stress/              -- Session 6 pre-work: load scenarios, budget metrics, runner, REPORT.md
├── CAG_LIMITS.md            -- Where CAG stops being viable here, with measured numbers
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
| `CONVERSATIONAL_PROMPT_VERSION` | `v3` | Which template serves the session endpoint (`v3` has `<project_metadata>` and `<audience>`) |
| `CACHE_TTL` | `86400` | Exact-match cache TTL, in seconds |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embeddings for the semantic cache |
| `SEMANTIC_CACHE_THRESHOLD` | `0.85` | Minimum cosine similarity to serve a semantic hit |
| `SEMANTIC_CACHE_LOG_ONLY` | `false` | Log would-be hits without serving them (threshold calibration) |
| `MAX_TURNS` | `6` | Sliding-window size (in turns) for a session's conversation history |
| `MAX_ATTACHMENT_WORDS` | `8000` | Word-count cap per attachment; over this it's reported as failed, not truncated |
| `METADATA_EXTRACTOR_MODEL` | `gpt-4o-mini` | Primary model for the metadata-extraction side call |
| `METADATA_EXTRACTOR_FALLBACK_MODEL` | `claude-haiku-4-5-20251001` | Its fallback, via the same Router mechanism as the main call |
| `COMPRESSION_MODEL` | `gpt-5-nano` | Primary model for the compression side calls (summary + optional LLM anchor classifier) |
| `COMPRESSION_FALLBACK_MODEL` | `claude-haiku-4-5-20251001` | Its fallback, via the same Router mechanism as the main call |
| `ANCHOR_DETECTION_MODE` | `heuristic` | `heuristic` (regex, no LLM call) or `llm` (Instructor classifier per evicted turn) |
| `DATA_CATALOG_PATH` | `data/data_catalog.yaml` | The catalog the ingestion pipeline obeys |
| `CORPUS_ROOT` | `data/corpus` | Where the catalog's `file://` locations resolve to |
| `PSEUDONYM_MAPPING_PATH` | `data/.pseudonyms.json` | Pseudonym mapping table. Git-ignored: it is regulated data |
| `PSEUDONYM_HASH_SALT` | `change-me-in-prod` | Keys the HMAC the mapping table stores. **Must be overridden in any real deployment** |
| `PSEUDONYM_LOCALE` | `es_ES` | Faker locale for generated pseudonyms |
| `PII_SCORE_THRESHOLD` | `0.7` | Minimum Presidio confidence. Raised from the 0.5 default: the Spanish NER model tags common nouns as PERSON too often at 0.5 |

## Endpoints

| Method | Path                                | Description                                             |
|--------|-------------------------------------|-----------------------------------------------------------|
| GET    | `/health`                           | Health check                                            |
| POST   | `/api/v1/estimate`                  | Structured estimation (typed in, validated schema out)  |
| POST   | `/api/v1/estimate/text`             | Free-text estimation (transcription in, Markdown out)   |
| POST   | `/api/v1/estimate/stream`           | Free-text estimation streamed over Server-Sent Events   |
| POST   | `/api/v1/sessions`                  | Create a multi-turn session, returns `{"session_id": ...}` |
| GET    | `/api/v1/sessions/{id}`             | Read-only snapshot of a session's memory (summary, anchors, metadata, last turn) |
| POST   | `/api/v1/sessions/{id}/estimate`    | One turn of a multi-turn estimation, with optional attachments |
| POST   | `/api/v1/index/run`                 | Schedule an offline indexing run (202, work happens in the background) |
| GET    | `/api/v1/index/runs/{run_id}`       | Status and per-source report of an indexing run |
| GET    | `/api/v1/index/audit`               | The data audit report as Markdown, generated from the catalog |
| POST   | `/api/v1/query`                     | Online retrieval pipeline. **501** until there is a vector index |

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

Prompts live under `app/prompts/estimation/<version>/` and are rendered by `app/prompts/loader.py`. There are two independent version settings: `PROMPT_VERSION` (default `v1`) for the single-shot endpoint's `system.j2`/`user.j2`/`examples.j2`, and `CONVERSATIONAL_PROMPT_VERSION` (default `v3`) for the session endpoint's own `system.j2`/`user.j2`. `v2` added the `<project_metadata>` block and a "this is a multi-turn conversation" framing that `v1` deliberately does not carry, so the single-shot flow's prompt never has to reason about a concept (project memory) that doesn't apply to it. `v3` (Session 5) adds on top of `v2` an `<audience>` block driven by the tier `EstimationService` resolves for the turn (see "Dynamic audience tier" below): it doesn't add a new field to the schema, it changes how the model frames `summary`/`phases` for that turn's reader. Every version reuses `v1/examples.j2` via `{% include %}` rather than duplicating the few-shot examples. All versions are deploy-time decisions, not client-supplied parameters, so rolling out a new prompt is a config change and every response echoes back the version that produced it. Each render logs a `prompt_rendered` event with the version and a content hash (not the text, so descriptions stay out of the logs).

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
  ],
  "resolved_tier": "developer",
  "tier_rule": "technical_audience"
}
```

**Why no caching here.** Both cache layers are skipped for this endpoint. The exact-match cache is keyed on the description alone, but the same transcript text means something different depending on the session's history and `project_metadata`, serving a cached answer would silently ignore the conversation. Making it cache-safe would mean folding the entire history + metadata into the key, which defeats the point of reusing anything.

**History vs. memory.** `ConversationHistory` (`app/sessions/models.py`, a Pydantic model) keeps a sliding window of the last `MAX_TURNS` `(user, assistant)` pairs, appended via `.append(user=..., assistant=...)`. `append` itself no longer trims (Session 5): trimming is `CompressionPolicy`'s job, since deciding whether an overflowing turn is disposable or must be rescued verbatim requires inspecting it first, see "Hybrid compression" below. The system prompt is rebuilt fresh each turn from the current `ProjectMetadata` and prepended by the caller, never stored in the window, so "always preserve the system prompt" is automatic rather than a truncation special case. The assistant side of each turn stores the estimation's **full JSON** (`EstimationResult.model_dump_json()`), not a hand-written summary, so nothing about a past turn's answer is lossy-compressed before it even leaves the window.

`ProjectMetadata` is a separate, never-truncated Pydantic model (project name, team size, technologies, agreed scope, constraints, rejected options) injected into the system prompt's `<project_metadata>` block on every turn. A fact from turn 1 still informs turn 20 even after turn 1 has scrolled out of the window, because it lives on the session, not inside the window.

**Updating `project_metadata`: an LLM delta + a deterministic Python merge.** After each turn, `app/sessions/metadata_extractor.py`'s `update_metadata()` asks a small, dedicated model (`METADATA_EXTRACTOR_MODEL`, with its own fallback via the Router, see below) for **only what's new or changed this turn**, not the full object. `ProjectMetadata.merge_with()` then combines that with what's already known: scalar fields are overwritten only when the update provides a non-null value, and list fields (`mentioned_technologies`, `explicit_constraints`, `rejected_options`) are unioned rather than replaced. This split matters because an LLM is non-deterministic: if the extractor call forgot to restate `project_name` on some turn, asking it for "the full updated object" would silently erase it. Asking for a delta and merging in Python means a forgotten field just falls back to the previous value, it can't regress. Both the extractor prompt and this merge were chosen and validated against exactly that failure mode; a regex heuristic was the cheaper alternative but doesn't hold up against free-form, bilingual (ES/EN) input as well as an LLM delta does. The extractor call fails open: if it errors after retries, `update_metadata` logs it and returns the previous metadata unchanged, so a flaky side call never turns a successful estimation into a 502.

**A dedicated, cheaper model for extraction, with fallback of its own.** The metadata call runs against its own Router group (`"metadata_extractor"` in `LLMWrapper`), configured via `METADATA_EXTRACTOR_MODEL` / `METADATA_EXTRACTOR_FALLBACK_MODEL`, independent of `PRIMARY_MODEL` / `FALLBACK_MODEL`. This is a deliberate choice not to reuse the (potentially larger/pricier) main model for a small side call, but unlike routing that call around the Router entirely (which would drop the fallback guarantee, the same trade-off flagged for the main call in `llm_wrapper.py`), it still goes through `LLMWrapper.complete_structured_with_messages(..., model_name="metadata_extractor")`, so a flaky provider on the cheap model still fails over instead of just failing.

**Attachments: local extraction, self-validating, with an LLM fallback and a length cap.** PDFs and Word docs are parsed locally (`app/attachments/extraction.py`) rather than uploaded to a provider's Files API, to keep the estimator independent of whichever model the Router happens to be using for a given call. The pipeline for each file:

1. Try `pypdf` (PDF) or `python-docx` (Word).
2. Validate the result against hard rules: non-empty, a minimum characters-per-page floor (catches a scanned/image-only PDF with no text layer), a minimum printable-character ratio (catches a garbled/mis-decoded extraction).
3. If a PDF fails validation, retry with `PyMuPDF`, which occasionally recovers text `pypdf` misses.
4. If it still fails, fall back to the LLM's own multimodal document support for that one file, the raw bytes go in as an inline `file` content block alongside the turn's text. If that call itself fails (unsupported model, provider quirk), the request degrades to a text-only retry noting the file couldn't be processed, rather than 502ing an otherwise-answerable turn.
5. If extraction *succeeds* but the result is longer than `MAX_ATTACHMENT_WORDS`, it's rejected anyway, not truncated. A silently truncated attachment produces a plausible-looking but partial answer with no indication anything was cut; reporting it as failed (`method: "too_long"`) tells the user outright. The message is deliberately non-technical, word counts and a multiple-of-the-limit ("~3.2x over the limit"), never "tokens", since that's a number a non-technical user can actually reason about. An oversized file is never routed to step 4's multimodal fallback either: the problem there is cost/context budget, and re-sending the same content as a raw file would make that worse, not better.

This keeps the common case (born-digital PDFs, Word docs) fully local while still reaching an answer for a scanned document, without ever letting attachment handling take down the whole request. Each attachment's outcome (`pypdf` / `pymupdf` / `docx` / `llm_fallback` / `too_long` / `failed`) is reported back in the response so the client can show what happened; Streamlit's Project session tab surfaces a `too_long` one as an explicit warning.

### Hybrid compression: anchors + cumulative summary (Session 5)

A plain sliding window eventually forgets things a long negotiation cannot afford to lose, a signed NDA mentioned on turn 2 is gone from the model's context by turn 10. `app/sessions/compression/` (`CompressionPolicy.apply`, called right after `history.append` in `EstimationService.estimate_in_session`) replaces the silent drop with two escape hatches, one per pair evicted from the window:

1. **`AnchorDetector`** inspects the evicted user turn. If it carries a durable commitment, signed contract, frozen scope, locked budget, legal/compliance mention (NDA, GDPR/RGPD, HIPAA, …), both messages of the pair move to `history.anchors` instead of being dropped: anchors are never evicted again, and `to_messages()` always splices them back in verbatim, after the summary and before the recent window. Two strategies, set via `ANCHOR_DETECTION_MODE`: `heuristic` (default) is a curated regex phrase list, duplicated for Spanish for the same reason `app/guardrails/input.py`'s prompt-injection patterns are, this service accepts descriptions in any language, so an English-only pattern list would be a real gap, not just incompleteness; `llm` classifies the turn through the same `LLMWrapper.complete_structured_with_messages` primitive every other structured call uses (routed to the `"compression"` Router group), more robust to paraphrase, at the cost of one extra call per evicted turn, and it fails open onto the heuristic if that call errors.
2. **`CumulativeSummarizer`** takes whatever wasn't promoted to an anchor and folds it into `history.summary`, a single rolling free-text summary (not a chain of them), via its own Instructor call. On failure it keeps the previous summary intact, losing one compaction pass is fine, wiping state the conversation already committed to is not.

`ConversationHistory.to_messages()` composes the result as `[summary?] + anchors_in_order + recent_sliding_window`, the summary is wrapped as a synthetic, visually-marked user message so the model treats it as prior context rather than an instruction from the current turn. Anchors have no count cap in this phase: with `MAX_TURNS` small and anchors meant to be rare, unbounded accumulation isn't expected to be a real problem before it would be worth the added complexity of capping them.

Both the summarizer and the LLM anchor classifier run against a dedicated, cheaper Router group (`"compression"`, `COMPRESSION_MODEL` / `COMPRESSION_FALLBACK_MODEL`, default `gpt-5-nano` / `claude-haiku-4-5-20251001`), same reasoning as the metadata extractor's own group: a small, high-volume side call doesn't need the main model's budget, but it still needs the fallback guarantee, so it goes through the Router rather than bypassing it.

### Dynamic audience tier (Session 5)

`app/sessions/tier_resolver.py`'s `resolve_tier()` derives an audience tier, `executive` / `pm` / `developer` / `default`, from the current transcript plus the session's accumulated `ProjectMetadata`, and `v3`'s `<audience>` block reframes the same structured output for that reader (executive: risk-first, 3-4 plain-English phases; pm: milestone-oriented, capped phases, conservative confidence; developer: technical detail, up to 6-8 phases). It's a precedence-ordered chain of pure-function rules, evaluated in order, first match wins: `nda_detected` and `regulatory_context` (HIPAA/GDPR/RGPD/…) resolve to `executive`, `technical_audience` (two or more distinct technical keywords, one mention alone is too noisy to promote) resolves to `developer`, `low_budget_pm` (`assumed_team_size <= 2`) resolves to `pm`, anything else is `default`. A rule whose predicate raises is logged and skipped rather than aborting resolution for the remaining rules. Both `resolved_tier` and the `rule` name that fired are echoed back on `SessionEstimateResponse` (`resolved_tier`, `tier_rule`) so a client can show *why* a given framing was chosen, not just which one. Resolution is fully automatic in this phase, there is no per-call override parameter; the API surface can grow one when a real caller needs to force a tier.

### Offline ingestion and the data catalog (Session 6)

The service now has two pipelines with nothing in common, and keeping them apart is the architectural decision, not a stylistic one.

**Offline** (`POST /api/v1/index/run`) runs ingest → parse → clean → validate → anonymise. It is triggered by ingestion events, never by a user question, and its budget is minutes. **Online** (`POST /api/v1/query`) is retrieve → augment → generate, with a user waiting and a budget under three seconds. It returns `501` today: retrieval needs a vector index, and the contract is written down now so the shape of the API is fixed before anything depends on it. A service that indexes two hundred PDFs inside the request path that answers questions is a service that stops answering questions.

`data/data_catalog.yaml` is the control surface, not documentation. `app/ingest/orchestrator.py` iterates over the sources it marks `include` and processes nothing else, so a source marked `exclude` is not indexed however present its files are. Three decisions are valid (`include`, `review`, `exclude`), and anything that is not `include` **must** carry a `decision_reason` — an exclusion with no recorded justification is indistinguishable from an oversight six months later. The catalog rejects unknown keys, duplicate source names and unsupported formats at load time rather than halfway through a run.

Quality is scored on four dimensions (completeness, consistency, actuality, reliability) which **compose rather than average**: a source at 5 on completeness and 1 on reliability is not a 3, it is a source whose data is complete and possibly false, which is the worst thing to put in a retrieval index. `app/ingest/census.py` is the factual half — it measures what is on disk and scores nothing, and `--against` re-measures every declared source so the catalog can be checked instead of trusted.

Parsing is one layer per kind of problem: **loaders** know how to reach bytes, **parsers** know what is inside them, **normalizers** produce the canonical `Document`. Each format gets the right tool rather than one universal library: JSON budgets render to structured markdown (a `json.dumps` blob mixes technical keys with semantic values and makes every record's vector look alike), transcripts split into turns carrying speaker and timestamp, DOCX splits by heading, PDF by page, XLSX to a markdown table. `unstructured` with `hi_res` is deliberately absent: the corpus is five predictable formats and born-digital PDFs, so it would cost an order of magnitude in latency and compute to recover tables and OCR that are not needed.

Cleaning sits between parser and normalizer so there is exactly one place where invariants are enforced, and it is split in two halves. Normalisation (`cleaning/budgets.py`, `cleaning/text.py`) transforms what can be transformed and decides nothing. Validation (`cleaning/schemas.py` with Pandera, `cleaning/policy.py`) states the contract and routes each failure: **repair** (already attempted upstream), **quarantine** (recoverable, kept for a human) or **discard** (contamination). `lazy=True` is what makes the routing possible at all — without it Pandera stops at the first error and the policy would be blind to every other one.

Every document carries a deterministic id (`source:document:unit`) that is stable across re-ingestions, so the downstream index replaces a document instead of accumulating a second copy of the corpus on every run, plus the catalog version that produced it.

### PII, pseudonymisation and GDPR (Session 6)

Anonymisation runs **before** indexing, not as a filter on responses. Once a value is in the vector space it is reachable by any query that comes semantically close, and there is no permission check in front of that.

Detection is Presidio wired to Spanish (`es_core_news_md`; the English default silently returns zero PERSON entities on Hispanic names, which is worse than a false positive because it goes unnoticed), extended with the identifiers this domain uses: `BUDGET_ID`, `CLIENT_CODE` and local phone numbers. The confidence threshold is raised to 0.7 and a short blacklist handles the words the Spanish model is confidently wrong about (`Mar`, `Sol`, `Cruz`).

Replacement is **reversible pseudonymisation**, not `<PERSON>` tokens: each entity becomes a stable fake value of the same kind, so the corpus keeps its shape instead of collapsing every person onto one token. Consistency is per original value, so the same person becomes the same pseudonym in all four hundred chunks that mention them — without that, the privacy layer would reintroduce exactly the vector-space fragmentation the cleaning layer exists to remove.

The mapping store is keyed on a **salted HMAC-SHA256 of the original value; the plaintext never reaches it**. Consistency still works (hash the value, find the row) and so does erasure (hash the value, delete the row), but the table cannot be enumerated: a leaked mapping file is a list of opaque digests and fake names, not a directory of every real person in the corpus. An HMAC rather than a bare digest because a plain SHA-256 of a personal name falls to a dictionary attack in seconds. `PSEUDONYM_HASH_SALT` must be overridden in any real deployment.

Known limits, stated rather than hidden: the JSON backing is not encrypted at rest and rewrites the whole file per save, and GDPR Article 17 is only partly answerable — the store covers locating the pseudonyms, knowing which sources mention a person and deleting the mapping, but removing the affected chunks from the index and writing the audit-log entry need an index that does not exist yet.

### Where CAG stops being viable (Session 6)

[`CAG_LIMITS.md`](CAG_LIMITS.md) holds the decision, with numbers measured from this repository's own stress run rather than assumed. The short version: the corpus **fits** (15.6% of the window) and is **cheap** ($0.0030 per query), and CAG is still the wrong architecture here. Two of the four ceilings fail — latency (9.5 s projected against a 3 s conversational budget) and quality under load (100% of superseded facts survive) — and the decision tree does not even reach them: traceability and per-user access control are structural requirements that no amount of context-window headroom can satisfy. It is the difference between "it does not fit" and "it does not serve".

## Testing
```bash
uv run pytest
```
The suite is offline: template rendering, schema validators, guardrail regexes, and the endpoints with either the service or just the LLM call faked out. No API keys or Redis needed.

The session tests are split by concern:

- `tests/test_sessions_models.py`, pure unit tests, no FastAPI, no fakes: `ConversationHistory.append`/`to_messages` (including the summary/anchors composition order), `ProjectMetadata.is_empty()`, and `ProjectMetadata.merge_with()`'s scalar-overwrite/list-union semantics (the exact mechanism that keeps a fact from regressing if the extractor forgets to restate it).
- `tests/test_sessions_metadata.py`, integration: metadata accumulates correctly across two turns of the same session.
- `tests/test_sessions_attachments.py`, integration: a PDF's content reaches the LLM and changes the estimate; an oversized attachment is reported (`method: "too_long"`) without blocking the request.
- `tests/test_sessions_window.py`, integration: session creation, 404 on an unknown session id, and the sliding window never exceeding `MAX_TURNS` raw turns once compression has kicked in.
- `tests/test_compression_anchors.py`, pure unit tests: the heuristic regex library (English + Spanish) and the LLM mode's dispatch + fail-open-onto-heuristic behaviour.
- `tests/test_compression_policy.py`, pure unit tests: the window trim itself, anchor promotion vs. summarization per evicted pair, and the summary accumulating correctly across repeated compression passes.
- `tests/test_tier_resolver.py`, pure unit tests: each rule in the precedence chain, EN/ES pattern coverage, and a broken predicate being skipped in favour of the next rule.

The integration files share `tests/_session_test_helpers.py` (`FakeLLMWrapper` + a real-PDF builder via PyMuPDF), only `LLMWrapper` is faked, not the whole service, so the session/history/metadata/attachment/compression logic itself runs for real, including parsing an actual generated PDF back with pypdf to check that attachment content really reaches the prompt.

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

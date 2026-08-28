"""Pipeline orchestrator. Glue between guardrails, caches, prompt rendering and
the LLM wrapper. The router holds none of this logic, its only job is to
translate HTTP errors.

Pipeline:

    1. Input guardrails (moderation + prompt injection + PII heuristics)
    2. Exact-match cache lookup  -> return cached=True on hit
    3. Semantic cache lookup     -> return cached=True on hit
    4. Render the versioned prompt
    5. LLM call via Instructor with response_model=EstimationResult
    6. Output guardrail (enforce_scope_response, filter policy)
    7. Write to BOTH caches (exact + semantic)
    8. Return EstimationResponse with cached=False

Order rationale: guardrails go before any cache because a malicious or PII
description should never be served from cache : skipping them to save 50 ms
would let an attacker replay a cached answer without passing moderation. The
exact-match cache goes before the semantic cache because it's the cheapest (no
embedding call). Both cache writes happen AFTER output validation so a failed
or hallucinated estimation is never persisted and replayed for the whole TTL.
"""

from __future__ import annotations

import base64
import hashlib
import json

import structlog

from app.attachments import enforce_length_limit, extract_attachment, format_attachments_block
from app.cache.semantic import EstimationSemanticCache
from app.guardrails.input import check_input
from app.guardrails.output import enforce_scope_response
from app.prompts import render_estimation_prompt, render_session_prompt
from app.schemas.estimation import (
    CallMeta,
    DetailLevel,
    EstimationDraft,
    EstimationRequest,
    EstimationResponse,
    EstimationResult,
    OutputFormat,
    ProjectType,
)
from app.services.cache import EstimationCache
from app.services.llm_wrapper import LLMWrapper
from app.sessions import (
    AttachmentReport,
    Session,
    SessionEstimateResponse,
    apply_compression,
    resolve_tier,
    update_metadata,
)

log = structlog.get_logger()


def _exact_cache_key(request: EstimationRequest, prompt_version: str, model: str) -> str:
    """Deterministic SHA-256 key over the typed request + prompt_version + model."""
    payload = json.dumps(
        {
            "description": request.description,
            "project_type": request.project_type.value,
            "detail_level": request.detail_level.value,
            "output_format": request.output_format.value,
            "prompt_version": prompt_version,
            "model": model,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"estimation:{digest}"


class EstimationService:
    """Single entry point for the structured estimation pipeline."""

    def __init__(
        self,
        *,
        llm_wrapper: LLMWrapper,
        exact_cache: EstimationCache,
        semantic_cache: EstimationSemanticCache | None = None,
        openai_client=None,
        prompt_version: str = "v1",
        conversational_prompt_version: str = "v3",
        max_attachment_words: int = 8000,
        anchor_detection_mode: str = "heuristic",
    ) -> None:
        self.llm_wrapper = llm_wrapper
        self.exact_cache = exact_cache
        self.semantic_cache = semantic_cache
        self.openai_client = openai_client
        self.prompt_version = prompt_version
        self.conversational_prompt_version = conversational_prompt_version
        self.max_attachment_words = max_attachment_words
        self.anchor_detection_mode = anchor_detection_mode

    def estimate(self, request: EstimationRequest) -> EstimationResponse:
        # 1. Input guardrails, raises InputGuardrailViolation on rejection.
        check_input(request.description, openai_client=self.openai_client)

        # 2. Exact-match cache lookup.
        cache_key = _exact_cache_key(request, self.prompt_version, self.llm_wrapper.primary_model)
        cached = self.exact_cache.get(cache_key)
        if cached:
            log.info("estimation_cache_hit", kind="exact", key_prefix=cache_key[:24])
            return EstimationResponse(
                result=EstimationResult.model_validate(cached["result"]),
                prompt_version=self.prompt_version,
                cached=True,
                # Nothing was spent on this call; model/provider describe
                # whichever deployment originally produced the cached answer.
                meta=CallMeta(**{**cached.get("meta", {}), "cost_usd": 0.0, "latency_ms": 0}),
            )

        # 3. Semantic cache lookup.
        if self.semantic_cache is not None:
            semantic_hit = self.semantic_cache.lookup(request, self.prompt_version)
            if semantic_hit is not None:
                log.info("estimation_cache_hit", kind="semantic")
                return EstimationResponse(
                    result=semantic_hit,
                    prompt_version=self.prompt_version,
                    cached=True,
                    meta=CallMeta(),
                )

        # 4. Render the versioned prompt.
        system_prompt, user_message = render_estimation_prompt(
            request, version=self.prompt_version
        )

        # 5. LLM call with Instructor + Pydantic validators (re-prompts on failure).
        #    The model fills an EstimationDraft (no totals); the totals are
        #    summed here in Python : see EstimationDraft's docstring for the
        #    measured reason why asking the LLM for them does not work.
        draft, meta = self.llm_wrapper.complete_structured(
            system_prompt=system_prompt,
            user_message=user_message,
            response_model=EstimationDraft,
        )
        result = EstimationResult.from_draft(draft)
        log.info(
            "estimation_generated",
            prompt_version=self.prompt_version,
            confidence_pct=result.confidence_pct,
            total_cost_eur=result.total_cost_eur,
            phases=len(result.phases),
            **meta,
        )

        # 6. Output guardrail (filter): normalises low-confidence answers.
        result = enforce_scope_response(result)

        # 7. Cache the validated payload only (never persist failed validations).
        self.exact_cache.set(
            cache_key,
            {"result": result.model_dump(mode="json"), "meta": meta},
        )
        if self.semantic_cache is not None:
            self.semantic_cache.store(request, result, self.prompt_version)

        # 8. Return.
        return EstimationResponse(
            result=result,
            prompt_version=self.prompt_version,
            cached=False,
            meta=CallMeta(**meta),
        )

    def estimate_in_session(
        self,
        session: Session,
        *,
        transcript: str,
        attachment_files: list[tuple[str, str | None, bytes]],
        project_type: ProjectType | None = None,
        detail_level: DetailLevel | None = None,
        output_format: OutputFormat | None = None,
    ) -> SessionEstimateResponse:
        """One turn of the multi-turn, session-scoped estimation flow.

        Deliberately skips both caches, unlike the single-shot pipeline. The
        exact-match cache is keyed on the description alone, but here the same
        transcript text means something different depending on this session's
        history and project_metadata; serving a cached answer would silently
        ignore the conversation. The semantic cache has the same problem one
        level up (its bucket has no notion of "this session"). Memoizing a
        stateful conversation would need the cache key to fold in the entire
        history + metadata, which defeats the point of reusing it.

        Pipeline:
            1. Input guardrails on the raw transcript.
            2. Remember any typed selectors this turn supplied.
            3. Resolve the audience tier from the transcript + accumulated
               metadata (see ``app.sessions.tier_resolver``).
            4. Extract attachments locally (enforcing a length cap), falling
               back to the LLM's native file support for anything local
               extraction couldn't validate.
            5. Render the system prompt (with project_metadata + tier) + user
               turn, using the dedicated conversational prompt version.
            6. LLM call over session.history + this turn's messages, degrading
               to a text-only retry if the multimodal fallback itself fails.
            7. Output guardrail.
            8. Append the turn to history (the full result JSON, not a
               summary) and run the compression policy: trim the sliding
               window, rescuing anchors and folding the rest into the running
               summary (see ``app.sessions.compression``).
            9. Refresh project_metadata via the extractor + deterministic
               merge (see ``app.sessions.update_metadata``).
        """
        check_input(transcript, openai_client=self.openai_client)
        self._apply_session_selectors(session, project_type, detail_level, output_format)

        resolved_tier, tier_rule = resolve_tier(
            transcript=transcript, metadata=session.project_metadata
        )
        session.last_resolved_tier = resolved_tier.value
        session.last_tier_rule = tier_rule

        attachments_block, multimodal_blocks, attachment_reports = self._process_attachments(
            attachment_files, max_words=self.max_attachment_words
        )

        system_prompt, user_message = render_session_prompt(
            transcript=transcript,
            project_type=session.project_type,
            detail_level=session.detail_level,
            output_format=session.output_format,
            project_metadata=session.project_metadata.model_dump(),
            metadata_is_empty=session.project_metadata.is_empty(),
            tier=resolved_tier.value,
            attachments_block=attachments_block,
            version=self.conversational_prompt_version,
        )
        messages = [{"role": "system", "content": system_prompt}, *session.history.to_messages()]
        user_content = (
            [{"type": "text", "text": user_message}, *multimodal_blocks]
            if multimodal_blocks
            else user_message
        )
        messages.append({"role": "user", "content": user_content})

        try:
            draft, meta = self.llm_wrapper.complete_structured_with_messages(
                messages=messages, response_model=EstimationDraft
            )
        except Exception:
            if not multimodal_blocks:
                raise
            draft, meta, user_message = self._retry_without_multimodal(
                session=session,
                transcript=transcript,
                attachments_block=attachments_block,
                multimodal_blocks=multimodal_blocks,
                attachment_reports=attachment_reports,
                system_prompt=system_prompt,
            )

        result = enforce_scope_response(EstimationResult.from_draft(draft))
        # The full result JSON, not a hand-rolled summary: it's what the
        # metadata extractor reads too, and re-deriving a compact string here
        # would just be a lossier copy of data we already have.
        assistant_content = result.model_dump_json()
        session.history.append(user=user_message, assistant=assistant_content)
        apply_compression(
            session.history,
            llm_wrapper=self.llm_wrapper,
            anchor_detection_mode=self.anchor_detection_mode,
        )
        session.project_metadata = update_metadata(
            previous=session.project_metadata,
            transcript=transcript,
            result=result,
            llm_wrapper=self.llm_wrapper,
        )

        log.info(
            "session_estimation_generated",
            session_id=session.session_id,
            turns=len(session.history),
            confidence_pct=result.confidence_pct,
            attachments=len(attachment_files),
            resolved_tier=resolved_tier.value,
            tier_rule=tier_rule,
            **meta,
        )

        return SessionEstimateResponse(
            result=result,
            prompt_version=self.conversational_prompt_version,
            cached=False,
            meta=CallMeta(**meta),
            session_id=session.session_id,
            project_metadata=session.project_metadata,
            attachments=attachment_reports,
            resolved_tier=resolved_tier.value,
            tier_rule=tier_rule,
        )

    @staticmethod
    def _apply_session_selectors(
        session: Session,
        project_type: ProjectType | None,
        detail_level: DetailLevel | None,
        output_format: OutputFormat | None,
    ) -> None:
        """Selectors are optional per turn but remembered on the session,
        set once (or defaulted on creation), only overwritten when a turn
        explicitly supplies a new value."""
        if project_type is not None:
            session.project_type = project_type
        if detail_level is not None:
            session.detail_level = detail_level
        if output_format is not None:
            session.output_format = output_format

    @staticmethod
    def _process_attachments(
        attachment_files: list[tuple[str, str | None, bytes]],
        *,
        max_words: int,
    ) -> tuple[str, list[dict], list[AttachmentReport]]:
        """Extract text locally, enforce the length cap, and for files that
        fail validation (not size, see below) build an LLM multimodal
        fallback block (PDFs only, that's what providers' inline file input
        actually targets) or, for anything else, a plain textual failure
        note. Returns ``(attachments_block, multimodal_blocks, reports)``.

        An oversized attachment is deliberately NOT routed to the multimodal
        fallback: the problem there is cost/context budget, not readability,
        so sending the same content as a raw file would make it worse, not
        better. It's reported back as a failed attachment instead, so the
        estimate still proceeds without it.
        """
        raw_by_filename = {filename: raw for filename, _ct, raw in attachment_files}
        content_type_by_filename = {filename: ct for filename, ct, _raw in attachment_files}
        extracted = [
            enforce_length_limit(extract_attachment(filename, raw), max_words)
            for filename, _content_type, raw in attachment_files
        ]
        attachments_block = format_attachments_block(extracted)

        multimodal_blocks: list[dict] = []
        unrecoverable_notes: list[str] = []
        reports: list[AttachmentReport] = []

        for r in extracted:
            if r.ok:
                reports.append(AttachmentReport(filename=r.filename, method=r.method, ok=True))
                continue
            if r.method == "too_long":
                unrecoverable_notes.append(
                    f"<attachment filename='{r.filename}' status='failed'>"
                    f"{r.note} This attachment is skipped for this estimate.</attachment>"
                )
                reports.append(AttachmentReport(filename=r.filename, method="too_long", ok=False, note=r.note))
            elif r.filename.lower().endswith(".pdf"):
                content_type = content_type_by_filename.get(r.filename) or "application/pdf"
                b64 = base64.b64encode(raw_by_filename[r.filename]).decode("ascii")
                multimodal_blocks.append(
                    {
                        "type": "file",
                        "file": {
                            "filename": r.filename,
                            "file_data": f"data:{content_type};base64,{b64}",
                        },
                    }
                )
                reports.append(
                    AttachmentReport(filename=r.filename, method="llm_fallback", ok=True, note=r.note)
                )
            else:
                unrecoverable_notes.append(
                    f"<attachment filename='{r.filename}' status='failed'>"
                    f"Local extraction failed ({r.note}) and no multimodal fallback applies to "
                    f"this file type. Ignore this file for the estimate.</attachment>"
                )
                reports.append(AttachmentReport(filename=r.filename, method="failed", ok=False, note=r.note))

        if unrecoverable_notes:
            attachments_block = f"{attachments_block}\n\n{chr(10).join(unrecoverable_notes)}".strip()

        return attachments_block, multimodal_blocks, reports

    def _retry_without_multimodal(
        self,
        *,
        session: Session,
        transcript: str,
        attachments_block: str,
        multimodal_blocks: list[dict],
        attachment_reports: list[AttachmentReport],
        system_prompt: str,
    ) -> tuple[EstimationDraft, dict, str]:
        """The multimodal fallback call itself failed (unsupported model,
        provider quirk). Degrade to a text-only retry rather than 502ing a
        request that would otherwise succeed on the other attachments/turns."""
        failed_names = {block["file"]["filename"] for block in multimodal_blocks}
        log.warning(
            "session_multimodal_fallback_failed", session_id=session.session_id, files=sorted(failed_names)
        )
        for report in attachment_reports:
            if report.filename in failed_names:
                report.method = "failed"
                report.ok = False

        degraded_notes = "\n".join(
            f"<attachment filename='{name}' status='failed'>"
            f"Neither local extraction nor the model's native file support could process "
            f"this file. Ignore it for the estimate.</attachment>"
            for name in sorted(failed_names)
        )
        degraded_block = f"{attachments_block}\n\n{degraded_notes}".strip()
        _, user_message = render_session_prompt(
            transcript=transcript,
            project_type=session.project_type,
            detail_level=session.detail_level,
            output_format=session.output_format,
            project_metadata=session.project_metadata.model_dump(),
            metadata_is_empty=session.project_metadata.is_empty(),
            attachments_block=degraded_block,
            version=self.conversational_prompt_version,
        )
        messages = [{"role": "system", "content": system_prompt}, *session.history.to_messages()]
        messages.append({"role": "user", "content": user_message})
        draft, meta = self.llm_wrapper.complete_structured_with_messages(
            messages=messages, response_model=EstimationDraft
        )
        return draft, meta, user_message

"""
api/routes/query.py — POST /query

Pipeline: job_id -> ExtractionResult (document_store) -> RAG prompt
(prompt_builder) -> LLM completion (llm.client) -> secret scrub +
hallucination score (output_guard) -> QueryResponse.

AGENT-LOOP SEAM — read before wiring this into agents/*:
  This route currently calls LLMClient.complete() directly. Per the open
  question on whether /query should go through agents/planner.py +
  executor.py + critic.py instead of hitting the LLM client straight, that
  decision was deferred. The call is isolated in _run_completion() below
  specifically so that swapping it for an agent-loop call later is a
  one-function change, not a rewrite of this whole route. Everything above
  and below _run_completion() (lookup, prompt building, guarding) stays
  the same either way.

Async note: unlike api/routes/ingest.py, this route does NOT need
run_in_executor. pdfplumber/Tesseract in ingest.py are blocking C-calls
with no async API, which is why that route offloads them to a thread.
LLMClient uses AsyncOpenAI, which is natively async — awaiting it directly
is correct and does not block the event loop.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from api.document_store import document_store
from api.schemas import GuardInfo, QueryRequest, QueryResponse, TrustInfo
from llm.client import LLMClient, LLMClientError
from llm.output_guard import scrub_secrets, score_hallucination
from llm.prompt_builder import build_rag_prompt

logger = logging.getLogger(__name__)

router = APIRouter()

# Single shared client for the process — same "construct once, reuse"
# pattern as DocExtractor in api/routes/ingest.py. Retries/timeouts are
# configured once via LLMConfig, not per-request.
_llm_client = LLMClient()


@router.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest) -> QueryResponse:
    result = document_store.get(request.job_id)
    if result is None:
        # Distinguish "never existed" from "expired/restarted" is not
        # possible with the current in-memory store (see document_store.py
        # gap note) — both look identical from here, so we report the
        # honest, narrower claim.
        raise HTTPException(
            status_code=404,
            detail=f"No document found for job_id '{request.job_id}'. "
                   f"It may not exist, or the server may have restarted "
                   f"since it was ingested.",
        )

    bundle = build_rag_prompt(
        result,
        request.question,
        client_token_counter=lambda text: _llm_client.count_tokens(text),
    )

    try:
        completion = await _run_completion(bundle, request)
    except LLMClientError as exc:
        logger.error("Query failed for job_id=%s: %s", request.job_id, exc)
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc

    scrub_result = scrub_secrets(completion.text)
    hallucination = score_hallucination(
        answer=completion.text,
        context_text=bundle.user,
        trust=bundle.trust,
        pages_included=bundle.pages_included,
    )

    cited_pages = _extract_cited_pages(completion.text)

    return QueryResponse(
        answer=scrub_result.scrubbed_text,
        pages_cited=cited_pages,
        pages_used=bundle.pages_included,
        pages_truncated=bundle.pages_truncated,
        trust=TrustInfo(
            overall_method=bundle.trust.overall_method,
            pages_total=bundle.trust.pages_total,
            ocr_pages=bundle.trust.ocr_pages,
            low_confidence_pages=bundle.trust.low_confidence_pages,
            mean_ocr_confidence=bundle.trust.mean_ocr_confidence,
            has_extraction_errors=bundle.trust.has_extraction_errors,
        ),
        guard=GuardInfo(
            secrets_found=scrub_result.secrets_found,
            hallucination_risk_score=hallucination.risk_score,
            hallucination_risk_level=hallucination.risk_level.value,
            hallucination_signals=hallucination.signals,
        ),
        model=completion.model,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        total_tokens=completion.total_tokens,
        elapsed_seconds=round(completion.latency_seconds, 3),
        retries_used=completion.retries_used,
    )


async def _run_completion(bundle, request: QueryRequest):
    """Isolated on purpose — see module docstring's AGENT-LOOP SEAM note.
    Today: a direct LLMClient.complete() call.
    Later (if the agent-loop route is chosen): replace this body with a
    call into agents/executor.py, passing bundle as the tool/context input,
    while the rest of query_document() (guarding, response shaping) is
    unaffected."""
    return await _llm_client.complete(
        [
            {"role": "system", "content": bundle.system},
            {"role": "user", "content": bundle.user},
        ],
        model=request.model,
        temperature=request.temperature,
        max_output_tokens=request.max_output_tokens,
    )


def _extract_cited_pages(answer_text: str) -> list[int]:
    """Pull "(p. N)"-style citations out of the answer for QueryResponse.
    Reuses the same pattern output_guard.py checks against — kept here
    as a tiny local helper rather than importing output_guard's private
    regex, since this is a presentation concern (what to show the caller)
    not a risk-scoring concern."""
    import re
    return sorted({int(m) for m in re.findall(r"\(p\.?\s*(\d+)\)", answer_text, re.IGNORECASE)})
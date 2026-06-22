"""
llm/prompt_builder.py — RAG-style prompt construction from ExtractionResult.

Two gaps this module exists specifically to close (see tools/doc_extractor.py):

  GAP 1 — ExtractionResult.full_text joins page.text only. Tables never
  appear in it. In ops documents (invoices, contracts, reports) the answer
  is very often IN a table, not the surrounding prose. If we naively did
  `prompt = result.full_text`, every table silently disappears from the
  model's context with no error, no warning — just a wrong answer.
  Fix: walk pages explicitly, render each page's tables via TableData
  .to_markdown() right after that page's text, so tables stay anchored
  next to the prose that introduces them.

  GAP 2 — ocr_confidence lives per-page with no aggregation. Without one,
  output_guard.py would have no way to say "this answer rests on a page
  that was OCR'd at 41% confidence" vs "100% native text". Fix: this
  module computes that aggregation once (since it already walks every
  page to build context) and returns it as TrustMetadata, which both
  output_guard.py and QueryResponse can read instead of re-deriving it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tools.doc_extractor import ExtractionMethod, ExtractionResult, PageContent

# Below this OCR confidence, a page is flagged as low-trust in the prompt
# itself (so the model is told to be cautious) and in TrustMetadata (so
# output_guard.py can weight its hallucination score accordingly).
LOW_CONFIDENCE_THRESHOLD = 60.0

# Conservative headroom subtracted from the model's context budget.
# Token counting can be approximate (see llm.client._ApproxEncoding —
# the offline fallback), and we always need room left over for the
# system prompt, the user's question, and the model's own output tokens.
CONTEXT_SAFETY_MARGIN_TOKENS = 500

_SYSTEM_PROMPT = """You are a document analysis assistant for an operations \
platform. Answer the user's question using ONLY the document context below.

Rules:
- If the answer is not present in the context, say so explicitly. Do not guess.
- When you state a fact, cite the page number it came from, like: (p. 3).
- Tables are provided in Markdown. Read them carefully — figures, dates, and \
totals are frequently the answer.
- Some pages were extracted via OCR and are marked [LOW CONFIDENCE OCR]. \
Treat numbers and exact text on those pages with extra caution and say so \
if your answer depends on one."""


@dataclass
class TrustMetadata:
    """Aggregated, document-level trust signal — the thing GAP 2 was
    missing. Computed once here, consumed by output_guard.py and surfaced
    to the caller in QueryResponse so it's never a silent unknown."""

    overall_method: str
    pages_total: int
    ocr_pages: list[int] = field(default_factory=list)
    low_confidence_pages: list[int] = field(default_factory=list)
    mean_ocr_confidence: float | None = None
    has_extraction_errors: bool = False


@dataclass
class PromptBundle:
    """Everything llm/client.py and output_guard.py need, bundled together
    so query.py doesn't have to re-derive any of it."""

    system: str
    user: str
    trust: TrustMetadata
    pages_included: list[int]
    pages_truncated: list[int]
    context_token_estimate: int


def build_rag_prompt(
    result: ExtractionResult,
    question: str,
    *,
    client_token_counter=None,
    max_context_tokens: int = 6000,
) -> PromptBundle:
    """
    Build a RAG prompt from an ExtractionResult + a user question.

    Args:
        result: the document's ExtractionResult (from DocExtractor).
        question: the user's natural-language question.
        client_token_counter: optional callable(str) -> int, normally
            LLMClient.count_tokens. Passed in rather than imported, so this
            module stays testable without an OpenAI client/key — tests can
            pass a trivial `len(s.split())` stub instead.
        max_context_tokens: budget for the document context portion only
            (system prompt + question are accounted separately, not
            included in this number).

    Returns:
        PromptBundle ready to hand to LLMClient.complete() as:
            [{"role": "system", "content": bundle.system},
             {"role": "user", "content": bundle.user}]
    """
    counter = client_token_counter or (lambda text: max(1, len(text) // 4))

    trust = _compute_trust_metadata(result)

    budget = max_context_tokens - CONTEXT_SAFETY_MARGIN_TOKENS
    if budget <= 0:
        budget = max_context_tokens  # caller passed an unreasonably small budget; don't go negative

    context_parts: list[str] = []
    pages_included: list[int] = []
    pages_truncated: list[int] = []
    running_tokens = 0

    for page in result.pages:
        block = _render_page_block(page, low_confidence_threshold=LOW_CONFIDENCE_THRESHOLD)
        if not block.strip():
            continue  # blank/empty page — nothing to anchor a citation to

        block_tokens = counter(block)

        if running_tokens + block_tokens > budget and pages_included:
            # Keep whatever we already fit; stop adding more pages rather
            # than silently dropping from the middle of an already-included
            # page. Every page included or excluded is tracked explicitly —
            # no silent loss the way full_text alone would cause.
            pages_truncated.append(page.page_number)
            continue

        context_parts.append(block)
        pages_included.append(page.page_number)
        running_tokens += block_tokens

    if pages_truncated:
        context_parts.append(
            f"\n[NOTE: {len(pages_truncated)} additional page(s) "
            f"({', '.join(str(p) for p in pages_truncated)}) were omitted "
            f"to fit the context window. Say so if the answer might be there.]"
        )

    context_text = "\n\n".join(context_parts) if context_parts else "[No extractable content in this document.]"

    user_message = (
        f"=== DOCUMENT CONTEXT ({result.source_filename}) ===\n"
        f"{context_text}\n"
        f"=== END DOCUMENT CONTEXT ===\n\n"
        f"Question: {question}"
    )

    return PromptBundle(
        system=_SYSTEM_PROMPT,
        user=user_message,
        trust=trust,
        pages_included=pages_included,
        pages_truncated=pages_truncated,
        context_token_estimate=running_tokens,
    )


def _render_page_block(page: PageContent, *, low_confidence_threshold: float) -> str:
    """One page's contribution to the context: header + text + its tables,
    in that order, so a table always sits next to the prose that introduced
    it rather than being hoisted to a separate "all tables" section where
    the model loses the connection to its source page."""
    is_low_confidence = (
        page.method == ExtractionMethod.OCR
        and page.ocr_confidence is not None
        and page.ocr_confidence < low_confidence_threshold
    )

    header = f"--- Page {page.page_number} ({page.method.value}"
    if is_low_confidence:
        header += ", LOW CONFIDENCE OCR"
    header += ") ---"

    lines = [header]
    if page.text.strip():
        lines.append(page.text.strip())

    for table in page.tables:
        md = table.to_markdown()
        if md:
            lines.append(f"\n[Table {table.table_index} on page {page.page_number}]\n{md}")

    return "\n".join(lines)


def _compute_trust_metadata(result: ExtractionResult) -> TrustMetadata:
    """Single aggregation pass over pages — the fix for GAP 2.
    output_guard.py reads this instead of re-walking result.pages itself,
    so there is exactly one place that defines what 'low confidence' means."""
    ocr_pages: list[int] = []
    low_confidence_pages: list[int] = []
    confidences: list[float] = []

    for page in result.pages:
        if page.method == ExtractionMethod.OCR:
            ocr_pages.append(page.page_number)
            if page.ocr_confidence is not None:
                confidences.append(page.ocr_confidence)
                if page.ocr_confidence < LOW_CONFIDENCE_THRESHOLD:
                    low_confidence_pages.append(page.page_number)

    mean_confidence = sum(confidences) / len(confidences) if confidences else None

    return TrustMetadata(
        overall_method=result.method.value,
        pages_total=result.total_pages,
        ocr_pages=ocr_pages,
        low_confidence_pages=low_confidence_pages,
        mean_ocr_confidence=round(mean_confidence, 1) if mean_confidence is not None else None,
        has_extraction_errors=len(result.errors) > 0,
    )
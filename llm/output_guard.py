"""
llm/output_guard.py — last line of defense before a model's answer reaches
the caller.

Two independent jobs, kept separate on purpose:
  1. scrub_secrets()  — pattern-match and redact anything that looks like a
     credential, before it ever leaves the process. This runs regardless of
     hallucination risk; a correct answer that happens to echo back an API
     key from the source document is just as dangerous as a wrong one.
  2. score_hallucination() — a heuristic risk score for whether the model's
     answer is actually grounded in the provided context, using the
     TrustMetadata that prompt_builder.py already computed (GAP 2's fix) so
     we are not re-deriving OCR confidence from scratch here.

This is deliberately NOT a second LLM call ("LLM judging LLM"). That adds
latency, cost, and a second hallucination surface on top of the first. These
are cheap, deterministic, explainable signals — good enough to flag review
candidates, not meant to be a courtroom-grade fact-checker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from llm.prompt_builder import TrustMetadata

# =============================================================================
# Secret scrubbing
# =============================================================================

# Each pattern is (label, compiled regex). Label is shown in flagged output
# without showing the matched text itself — see scrub_secrets() docstring
# for why we never log/return the raw matched value.
_SECRET_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("openai_api_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("anthropic_api_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("aws_access_key_id", re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("generic_bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9._-]{20,}\b", re.IGNORECASE)),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    ("private_key_block", re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    # 13–19 digits covers the major card-number lengths; this is a coarse
    # net (will catch some non-card numbers too) by design — false positives
    # here just mean an over-eager redaction, which is the safe failure mode.
    ("possible_credit_card", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("generic_api_key_assignment", re.compile(
        r"\b(api[_-]?key|secret|password|token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-/+]{12,}['\"]?",
        re.IGNORECASE,
    )),
]


@dataclass
class ScrubResult:
    scrubbed_text: str
    secrets_found: list[str] = field(default_factory=list)  # labels only, never matched values

    @property
    def has_secrets(self) -> bool:
        return len(self.secrets_found) > 0


def scrub_secrets(text: str) -> ScrubResult:
    """
    Redact anything matching a known credential pattern.

    Why secrets_found stores LABELS, not the matched substrings:
        Returning the matched text defeats the purpose — anything holding
        the ScrubResult (logs, mlops/cost_logger.py, an error message shown
        to a different user) would now itself contain the secret we just
        redacted from the response. The label ("aws_access_key_id") tells
        an operator what kind of thing was caught without reproducing it.

    Args:
        text: raw LLM output (or, defensively, anything else worth checking
              before it leaves the process boundary).

    Returns:
        ScrubResult with the cleaned text and which pattern labels fired.
    """
    found: list[str] = []
    scrubbed = text

    for label, pattern in _SECRET_PATTERNS:
        if pattern.search(scrubbed):
            found.append(label)
            scrubbed = pattern.sub(f"[REDACTED:{label.upper()}]", scrubbed)

    return ScrubResult(scrubbed_text=scrubbed, secrets_found=found)


# =============================================================================
# Hallucination scoring
# =============================================================================

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class HallucinationScore:
    risk_score: float  # 0.0 (fully grounded) – 1.0 (likely fabricated)
    risk_level: RiskLevel
    signals: list[str] = field(default_factory=list)  # human-readable reasons, for the caller/operator


# Tunable weights — kept as module constants (not buried magic numbers
# inside the function) so they can be tuned without re-reading the logic,
# same "every tunable in one place" principle as ExtractionConfig.
_WEIGHT_LOW_OVERLAP = 0.45
_WEIGHT_NO_CITATION = 0.20
_WEIGHT_INVALID_CITATION = 0.25
_WEIGHT_LOW_OCR_TRUST = 0.20
_WEIGHT_EXTRACTION_ERRORS = 0.10

_OVERLAP_FLOOR = 0.15  # below this lexical overlap with context, treat as "no real overlap"
_CITATION_PATTERN = re.compile(r"\(p\.?\s*(\d+)\)", re.IGNORECASE)

_LEVEL_THRESHOLDS = (  # (max_score_for_level, level) — checked in order
    (0.3, RiskLevel.LOW),
    (0.6, RiskLevel.MEDIUM),
)

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on", "for",
    "and", "or", "it", "this", "that", "as", "be", "by", "with", "at", "from",
})


def score_hallucination(
    answer: str,
    context_text: str,
    trust: TrustMetadata,
    pages_included: list[int],
) -> HallucinationScore:
    """
    Heuristic risk score for whether `answer` is grounded in `context_text`.

    Three independent signals, combined:
      1. Lexical overlap — does the answer reuse vocabulary that actually
         appears in the context, or does it look unmoored from the source?
         (Crude proxy for grounding — not semantic similarity, just shared
         non-stopword tokens. Cheap, explainable, no extra model call.)
      2. Citation behaviour — the system prompt asks the model to cite
         page numbers like "(p. 3)". An answer with zero citations, or one
         citing a page that was never actually in context, is a concrete
         red flag rather than a vibe.
      3. Source trust — an answer resting on pages prompt_builder.py already
         flagged as low-confidence OCR (or a document with extraction
         errors) is inherently riskier even if signals 1 and 2 look fine —
         the SOURCE itself was untrustworthy, not just the model's use of it.

    Args:
        answer: the model's raw response text (scrub_secrets() first if this
                will be shown to a user — scoring and scrubbing are separate
                steps, see module docstring).
        context_text: the document context that was actually sent to the
                model (PromptBundle.user, or just the context portion).
        trust: TrustMetadata from prompt_builder.build_rag_prompt — reused,
               not recomputed, so there's one definition of "low confidence".
        pages_included: PromptBundle.pages_included — used to validate that
               cited page numbers were actually in context.

    Returns:
        HallucinationScore with a 0–1 risk score, a banded level, and the
        specific signals that contributed (for logging/operator review).
    """
    signals: list[str] = []
    score = 0.0

    # ── Signal 1: lexical overlap ───────────────────────────────────────
    overlap = _lexical_overlap(answer, context_text)
    if overlap < _OVERLAP_FLOOR:
        score += _WEIGHT_LOW_OVERLAP
        signals.append(f"low lexical overlap with document context ({overlap:.0%})")

    # ── Signal 2: citation behaviour ────────────────────────────────────
    cited_pages = [int(m) for m in _CITATION_PATTERN.findall(answer)]
    if not cited_pages and "not found" not in answer.lower() and "no information" not in answer.lower():
        score += _WEIGHT_NO_CITATION
        signals.append("answer makes claims with no page citation")
    else:
        invalid = [p for p in cited_pages if p not in pages_included]
        if invalid:
            score += _WEIGHT_INVALID_CITATION
            signals.append(f"answer cites page(s) not present in context: {invalid}")

    # ── Signal 3: source trust ───────────────────────────────────────────
    if trust.mean_ocr_confidence is not None and trust.mean_ocr_confidence < 60.0:
        score += _WEIGHT_LOW_OCR_TRUST
        signals.append(f"source document mean OCR confidence is low ({trust.mean_ocr_confidence})")
    if trust.has_extraction_errors:
        score += _WEIGHT_EXTRACTION_ERRORS
        signals.append("source document had non-fatal extraction errors")

    score = min(score, 1.0)
    level = RiskLevel.HIGH
    for threshold, candidate_level in _LEVEL_THRESHOLDS:
        if score <= threshold:
            level = candidate_level
            break

    return HallucinationScore(risk_score=round(score, 2), risk_level=level, signals=signals)


def _lexical_overlap(answer: str, context_text: str) -> float:
    """Fraction of the answer's non-stopword vocabulary that also appears
    in the context. Deliberately crude (no embeddings, no extra model
    call) — a fast, explainable first-pass signal, not a semantic judge."""
    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return 1.0  # empty answer has nothing to be hallucinated; not this function's concern
    context_tokens = _tokenize(context_text)
    if not context_tokens:
        return 0.0
    overlap_count = sum(1 for t in answer_tokens if t in context_tokens)
    return overlap_count / len(answer_tokens)


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}
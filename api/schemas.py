#shapes the data flowing inside and outside the API
#Stands as a bridge between input and the internal workflow logic.

#Important for validation of input data, data hiding to expel exposures and secrets, API Contracts where FastAPI reads
#your schemas and auto-generates OpenAI docs


from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str

class IngestRequest(BaseModel):
    filename: str
    content_type: str #pdf, docs, images or text

class IngestResponse(BaseModel):
    job_id: str
    status: str
    message: str

    # ── Extraction summary ────────────────────────────────────────────────────
    extraction_method: str | None = None
    total_pages: int | None = None
    total_words: int | None = None
    total_tables: int | None = None
    elapsed_seconds: float | None = None
    has_errors: bool = False
    errors: list[str] = []


# =============================================================================
# Day 3 — Query layer
# =============================================================================

class QueryRequest(BaseModel):
    """Input contract for POST /query.

    job_id ties this request back to a document previously processed by
    POST /ingest. NOTE: as of Day 3, nothing in the ingest path persists
    ExtractionResult keyed by job_id anywhere durable — see the seam
    comment in api/routes/query.py. This schema is written against the
    intended contract (lookup by job_id), not against a stand-in like
    "paste the full document text again", because the latter would bake
    a workaround into the API surface that's painful to remove later.
    """

    job_id: str
    question: str

    # Per-call overrides — optional, fall back to LLMConfig defaults.
    model: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None


class TrustInfo(BaseModel):
    """Mirrors llm.prompt_builder.TrustMetadata — kept as a separate
    Pydantic model (not the dataclass directly) so the API's response
    shape doesn't silently change if the internal dataclass evolves."""

    overall_method: str
    pages_total: int
    ocr_pages: list[int] = []
    low_confidence_pages: list[int] = []
    mean_ocr_confidence: float | None = None
    has_extraction_errors: bool = False


class GuardInfo(BaseModel):
    """Mirrors llm.output_guard's ScrubResult + HallucinationScore,
    flattened into one block for the response."""

    secrets_found: list[str] = []
    hallucination_risk_score: float
    hallucination_risk_level: str
    hallucination_signals: list[str] = []


class QueryResponse(BaseModel):
    answer: str
    pages_cited: list[int] = []
    pages_used: list[int] = []
    pages_truncated: list[int] = []

    trust: TrustInfo
    guard: GuardInfo

    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    elapsed_seconds: float
    retries_used: int = 0
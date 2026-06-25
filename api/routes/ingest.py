"""
api/routes/ingest.py
====================
POST /ingest — main data entry point of the pipeline.

Receives a raw file upload, runs DocExtractor, and returns a structured
IngestResponse that downstream agents (planner, executor, critic) can act on.

Design decisions
----------------
UploadFile + Form (not JSON body):
    Binary files must travel as multipart/form-data.
    JSON body with base64-encoded bytes would work but adds ~33% size
    overhead and forces clients to encode before sending. UploadFile is
    the correct FastAPI pattern for file ingestion.

run_in_executor (thread pool):
    pdfplumber and pytesseract are blocking (synchronous) C-extension calls.
    Running them directly inside an async route would block the entire event
    loop, making the server unresponsive to other requests during extraction.
    asyncio.get_event_loop().run_in_executor() offloads the work to a thread
    pool so the event loop stays free.

File size guard (50 MB):
    Reject oversized files before reading bytes — avoids loading a 2 GB PDF
    into memory only to fail later. Raises HTTP 413 immediately.

Error mapping:
    ValueError  (unsupported extension) → HTTP 422 Unprocessable Entity
    Generic Exception                   → HTTP 500 Internal Server Error
    Both carry a structured detail message for API clients.

Logging:
    extraction summary is logged at INFO after every successful run —
    safe for log streams because summary() never includes document text.

Day 3 addition
--------------
document_store.put(job_id, result) — see api/document_store.py.
POST /query needs to look up the ExtractionResult for a given job_id.
Without this line, every /query call 404s no matter what job_id is sent,
because nothing ever persisted the result anywhere. This is a stopgap
(in-memory, single-process — see document_store.py's own docstring for
the real gap), not the final storage layer.
"""

from __future__ import annotations

import uuid
import asyncio
import logging
from functools import partial

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from api.document_store import document_store
from api.schemas import IngestResponse
from tools.doc_extractor import DocExtractor, ExtractionConfig

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Module-level singleton ────────────────────────────────────────────────────
# DocExtractor is instantiated once at module load.
# _check_tesseract() runs here — server startup reveals missing binaries
# immediately rather than on the first upload request.
_extractor = DocExtractor(ExtractionConfig())

# ── Constants ─────────────────────────────────────────────────────────────────
_MAX_FILE_BYTES = 50 * 1024 * 1024   # 50 MB hard ceiling


# =============================================================================
# Route
# =============================================================================

@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest a document",
    description=(
        "Upload a PDF or image file for extraction. "
        "Returns structured text, table counts, and extraction metadata. "
        "Supports: .pdf, .png, .jpg, .jpeg, .tiff, .tif, .bmp, .webp"
    ),
)
async def ingest_document(
    file: UploadFile = File(..., description="PDF or image file to extract"),
) -> IngestResponse:
    """
    Extract text, tables, and metadata from an uploaded document.

    The file is read into memory as bytes, then passed to DocExtractor which
    auto-detects the extraction strategy (native text vs OCR) per page.
    Heavy extraction work runs in a thread pool to keep the event loop free.

    Returns IngestResponse with extraction summary fields populated.
    """
    job_id   = str(uuid.uuid4())
    filename = file.filename or "unknown"

    logger.info("job=%s  file=%s  content_type=%s", job_id, filename, file.content_type)

    # ── 1. Read bytes ─────────────────────────────────────────────────────────
    raw_bytes = await file.read()

    # ── 2. File size guard ────────────────────────────────────────────────────
    if len(raw_bytes) > _MAX_FILE_BYTES:
        size_mb = len(raw_bytes) / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=(
                f"File '{filename}' is {size_mb:.1f} MB. "
                f"Maximum allowed size is {_MAX_FILE_BYTES // (1024*1024)} MB."
            ),
        )

    if not raw_bytes:
        raise HTTPException(
            status_code=422,
            detail=f"File '{filename}' is empty.",
        )

    # ── 3. Extract — offload blocking work to thread pool ────────────────────
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,                                           # default ThreadPoolExecutor
            partial(_extractor.extract_bytes, raw_bytes, filename),
        )

    except ValueError as exc:
        # Unsupported file extension — client error
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as exc:
        # Unexpected extraction failure — server error
        logger.error(
            "job=%s  Extraction failed for '%s': %s",
            job_id, filename, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Extraction failed for '{filename}': {exc}",
        ) from exc

    # ── 4. Persist for downstream lookup (Day 3: POST /query needs this) ─────
    # Stored even on a "partial" result (some pages errored) — a document
    # with 18 good pages and 2 OCR failures is still queryable; query.py's
    # prompt_builder already accounts for per-page errors via TrustMetadata.
    document_store.put(job_id, result)

    # ── 5. Build response ─────────────────────────────────────────────────────
    summary = result.summary()

    status  = "partial" if result.errors else "completed"
    message = (
        f"Extracted {summary['pages']} page(s) from '{filename}' "
        f"using {summary['method']} in {summary['elapsed_seconds']}s."
    )

    logger.info("job=%s  %s", job_id, summary)

    return IngestResponse(
        job_id=job_id,
        status=status,
        message=message,
        extraction_method=summary["method"],
        total_pages=summary["pages"],
        total_words=summary["total_words"],
        total_tables=summary["total_tables"],
        elapsed_seconds=summary["elapsed_seconds"],
        has_errors=summary["has_errors"],
        errors=summary["errors"],
    )
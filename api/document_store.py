"""
api/document_store.py — TEMPORARY bridge between POST /ingest and POST /query.

THIS IS A KNOWN GAP, NOT A DESIGN DECISION.

POST /ingest returns a job_id (api/schemas.py:IngestResponse) but nothing
in the Day 2 pipeline persists the ExtractionResult anywhere keyed by that
job_id. POST /query (Day 3) needs to look up "what did we extract for this
job_id" to build a prompt. Without this module, query.py would have no way
to get a document's context at all.

This in-memory dict is a stand-in, not a solution:
  - Lost on every process restart.
  - Not shared across multiple uvicorn/gunicorn workers — a query hitting
    a different worker than the one that ran ingest will 404.
  - Unbounded growth — nothing ever evicts an entry.

It exists so Day 3 is testable end-to-end today. Replacing it later means:
  1. Swap InMemoryDocumentStore's body for a real backend (the repo already
     has mlops/tracking.py and tools/memory_tool.py — one of those may
     already be the intended home for this; both are unread as of Day 3,
     worth checking before building a third storage mechanism).
  2. Nothing in query.py changes — it only calls .get(job_id) / .put(...).
     The interface is the contract; only this file's internals should
     need to change.
"""

from __future__ import annotations

import threading

from tools.doc_extractor import ExtractionResult


class InMemoryDocumentStore:
    """NOT production storage. See module docstring."""

    def __init__(self) -> None:
        self._data: dict[str, ExtractionResult] = {}
        self._lock = threading.Lock()

    def put(self, job_id: str, result: ExtractionResult) -> None:
        with self._lock:
            self._data[job_id] = result

    def get(self, job_id: str) -> ExtractionResult | None:
        with self._lock:
            return self._data.get(job_id)

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._data.pop(job_id, None)


# Module-level singleton — same lifecycle as the FastAPI process.
# api/routes/ingest.py needs ONE addition to wire this up:
#     from api.document_store import document_store
#     ...
#     document_store.put(job_id, extraction_result)
# right after extraction succeeds and before building IngestResponse.
document_store = InMemoryDocumentStore()
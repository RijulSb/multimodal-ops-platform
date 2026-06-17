#Main data entry-point of the pipeline. Receives raw data and transforms them into chunks for embedding and storage.
#Ensures that the noises are removed and usable, actionable and retrievable context in maintained
#across the pipeline.

from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from api.schemas import IngestRequest, IngestResponse
import uuid

router = APIRouter()

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(body: IngestRequest):
    job_id = str(uuid.uuid4())
    # Day 2: real OCR + extraction goes here
    return IngestResponse(
        job_id=job_id,
        status="queued",
        message=f"Document '{body.filename}' received. Pipeline not yet wired."
    )


#notes: - OCR(Optical Character Recognition) is used to convert documents into images
#so that the pytesseract wrapper wraps the image and extracts 'text' to be used as strings
#The combination is helpful for character recognition and prevent ingestion pipeline crashes.

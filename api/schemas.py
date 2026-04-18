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



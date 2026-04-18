from fastapi import FastAPI
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from api.routes import health, ingest

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 MultiModal Ops Platform starting up...")
    yield
    print("🛑 Shutting down...")

app = FastAPI(
    title="Multimodal AI/MLOps Platform",
    description="Document Intelligence Platform with Agentic AI Pipeline",
    version="0.1.0",
    lifespan=lifespan
)

app.include_router(health.router, tags=['System'])
app.include_router(ingest.router, tags=['Ingestion'])


@app.get('/')
async def root():
    return {
        "message": "Multimodal Ops Platform",
        "docs": "/docs",
        "version": "0.1.0"
    }
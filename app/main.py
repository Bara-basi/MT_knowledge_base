from fastapi import FastAPI

from app.api.v1.router import router as api_v1_router
from app.services.retrieval import get_retrieval_service


app = FastAPI(
    title="MTSCO Knowledge Base API",
    version="0.1.0",
    description="Internal knowledge base backend exposed for n8n workflows.",
)


@app.on_event("startup")
def warmup_retrieval_models() -> None:
    get_retrieval_service().warmup_models()


@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "service": "mtsco-knowledge-base-api",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(api_v1_router, prefix="/api/v1")

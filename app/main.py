from fastapi import FastAPI

from app.api.v1.router import router as api_v1_router


app = FastAPI(
    title="MTSCO Knowledge Base API",
    version="0.1.0",
    description="Internal knowledge base backend exposed for n8n workflows.",
)


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

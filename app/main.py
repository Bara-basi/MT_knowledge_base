import logging
import time

from fastapi import FastAPI, Request

from app.api.v1.feishu import router as feishu_router
from app.api.v1.router import router as api_v1_router
from app.core.config import settings
from app.services.retrieval import get_retrieval_service

logger = logging.getLogger(__name__)


app = FastAPI(
    title="MTSCO Knowledge Base API",
    version="0.1.0",
    description="Internal knowledge base backend exposed for n8n workflows.",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %s %.1fms",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.on_event("startup")
def warmup_retrieval_models() -> None:
    if settings.skip_retrieval_warmup:
        logger.warning("Skipping retrieval warmup because SKIP_RETRIEVAL_WARMUP=true")
        return
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
app.include_router(feishu_router)

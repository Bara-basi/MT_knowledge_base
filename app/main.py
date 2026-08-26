import asyncio
import logging
import time

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse

from app.api.v1.feishu import router as feishu_router
from app.api.v1.router import router as api_v1_router
from app.core.config import settings
from app.db.postgres import (
    answer_job_metrics,
    check_postgres_health,
    harness_session_metrics,
)
from app.services.metrics import observe_http, render_prometheus
from app.services.harness import close_local_harnesses
from app.services.retrieval import get_retrieval_service

logger = logging.getLogger(__name__)


app = FastAPI(
    title="MTSCO Knowledge Base API",
    version="0.1.0",
    description="Internal knowledge base backend powered by DeepSeek Harness.",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    observe_http(
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_seconds=duration_ms / 1000,
    )
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


@app.on_event("shutdown")
def shutdown_harness_runtimes() -> None:
    closed = close_local_harnesses()
    if closed:
        logger.info("Closed %s local Harness runtime(s)", closed)


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


@app.get("/health/ready")
async def readiness_check() -> dict[str, object]:
    try:
        postgres = await asyncio.to_thread(check_postgres_health)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="postgres is unavailable") from exc
    return {
        "status": "ready",
        "postgres": {
            "database": postgres.get("database"),
            "schema": postgres.get("schema"),
        },
        "harness_enabled": settings.harness_enabled,
        "durable_queue_enabled": settings.feishu_durable_queue_enabled,
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics() -> str:
    jobs, sessions = await asyncio.gather(
        asyncio.to_thread(answer_job_metrics),
        asyncio.to_thread(harness_session_metrics),
    )
    return render_prometheus(answer_jobs=jobs, harness_sessions=sessions)


async def proxy_dev_request(path: str, request: Request) -> Response:
    if not settings.dev_proxy_target:
        raise HTTPException(status_code=404, detail="dev proxy is not configured")

    target_url = f"{settings.dev_proxy_target}/{path}".rstrip("/")
    if request.url.query:
        target_url = f"{target_url}?{request.url.query}"

    excluded_headers = {
        "connection",
        "content-length",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in excluded_headers
    }
    body = await request.body()
    timeout = httpx.Timeout(settings.dev_proxy_timeout)

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        proxied = await client.request(
            request.method,
            target_url,
            content=body,
            headers=headers,
        )

    response_headers = {
        key: value
        for key, value in proxied.headers.items()
        if key.lower() not in excluded_headers
    }
    return Response(
        content=proxied.content,
        status_code=proxied.status_code,
        headers=response_headers,
        media_type=proxied.headers.get("content-type"),
    )


@app.api_route(
    "/dev",
    methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
async def proxy_dev_root(request: Request) -> Response:
    return await proxy_dev_request("", request)


@app.api_route(
    "/dev/{path:path}",
    methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
async def proxy_dev_path(path: str, request: Request) -> Response:
    return await proxy_dev_request(path, request)


app.include_router(api_v1_router, prefix=f"{settings.api_route_prefix}/api/v1")
app.include_router(feishu_router, prefix=settings.feishu_route_prefix)

from typing import Any

from fastapi import APIRouter, Request

from app.api.v1.feishu import router as feishu_router
from app.api.v1.query import router as query_router
from app.api.v1.retrieval import router as retrieval_router

router = APIRouter()
router.include_router(feishu_router)
router.include_router(query_router)
router.include_router(retrieval_router)


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.api_route("/test", methods=["GET", "POST"])
async def test_n8n_call(request: Request) -> dict[str, Any]:
    body: Any = None
    if request.method == "POST":
        try:
            body = await request.json()
        except ValueError:
            body = await request.body()
            body = body.decode("utf-8", errors="replace") if body else None

    message = "收到"
    print(f"n8n test接口{message}: method={request.method}, body={body}", flush=True)

    return {
        "message": message,
        "received": True,
        "method": request.method,
        "body": body,
    }

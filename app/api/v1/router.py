from fastapi import APIRouter

from app.api.v1.documents import router as documents_router
from app.api.v1.external import router as external_router
from app.api.v1.feishu import router as feishu_router
from app.api.v1.query import router as query_router
from app.api.v1.retrieval import router as retrieval_router

router = APIRouter()
router.include_router(documents_router)
router.include_router(external_router)
router.include_router(feishu_router)
router.include_router(query_router)
router.include_router(retrieval_router)


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

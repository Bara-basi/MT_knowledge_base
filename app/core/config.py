from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    milvus_uri: str = os.getenv("MILVUS_URI", "http://localhost:19530")
    milvus_user: str = os.getenv("MILVUS_USER", "")
    milvus_password: str = os.getenv("MILVUS_PASSWORD", "")
    milvus_db_name: str = os.getenv("MILVUS_DB_NAME", "")
    milvus_collection_name: str = os.getenv(
        "MILVUS_COLLECTION_NAME",
        "mtsco_knowledge_chunks",
    )
    milvus_vector_dim: int = int(os.getenv("MILVUS_VECTOR_DIM", "1024"))
    milvus_metric_type: str = os.getenv("MILVUS_METRIC_TYPE", "COSINE")
    milvus_index_type: str = os.getenv("MILVUS_INDEX_TYPE", "HNSW")
    milvus_index_m: int = int(os.getenv("MILVUS_INDEX_M", "16"))
    milvus_index_ef_construction: int = int(
        os.getenv("MILVUS_INDEX_EF_CONSTRUCTION", "200")
    )

    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "BAAI/bge-large-zh-v1.5",
    )
    use_local_embedding_model: bool = _env_bool("USE_LOCAL_EMBEDDING_MODEL", True)
    embedding_cache_dir: str = os.getenv("EMBEDDING_CACHE_DIR", r"E:\models")
    embedding_device: str | None = os.getenv("EMBEDDING_DEVICE") or None
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
    embedding_normalize: bool = (
        os.getenv("EMBEDDING_NORMALIZE", "true").lower() == "true"
    )

    reranker_model_name: str = os.getenv(
        "RERANKER_MODEL_NAME",
        "BAAI/bge-reranker-v2-m3",
    )
    use_local_rerank_model: bool = _env_bool("USE_LOCAL_RERANK_MODEL", True)
    reranker_cache_dir: str = os.getenv("RERANKER_CACHE_DIR", embedding_cache_dir)
    reranker_device: str | None = os.getenv("RERANKER_DEVICE") or embedding_device
    reranker_batch_size: int = int(os.getenv("RERANKER_BATCH_SIZE", "8"))
    reranker_max_length: int = int(os.getenv("RERANKER_MAX_LENGTH", "512"))
    retrieval_recall_multiplier: int = int(
        os.getenv("RETRIEVAL_RECALL_MULTIPLIER", "5")
    )

    siliconflow_api_key: str = _get_first_env(
        "SILICONFLOW_API_KEY",
        "LLM_API_KEY",
        "OPENAI_API_KEY",
    )
    siliconflow_base_url: str = _get_first_env(
        "SILICONFLOW_BASE_URL",
        "LLM_BASE_URL",
        "OPENAI_BASE_URL",
        default="https://api.siliconflow.cn/v1",
    ).rstrip("/")
    siliconflow_timeout: float = float(os.getenv("SILICONFLOW_TIMEOUT", "180"))
    siliconflow_connect_timeout: float = float(os.getenv("SILICONFLOW_CONNECT_TIMEOUT", "10"))
    siliconflow_read_timeout: float = float(
        os.getenv("SILICONFLOW_READ_TIMEOUT", str(siliconflow_timeout))
    )
    siliconflow_write_timeout: float = float(os.getenv("SILICONFLOW_WRITE_TIMEOUT", "60"))
    siliconflow_pool_timeout: float = float(os.getenv("SILICONFLOW_POOL_TIMEOUT", "10"))

    n8n_query_webhook_url: str = os.getenv(
        "N8N_QUERY_WEBHOOK_URL",
        "http://n8n:5678/webhook/fastapi-test",
    )
    n8n_query_timeout: float = float(os.getenv("N8N_QUERY_TIMEOUT", "180"))
    n8n_query_connect_timeout: float = float(os.getenv("N8N_QUERY_CONNECT_TIMEOUT", "10"))


settings = Settings()

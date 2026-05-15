from __future__ import annotations

import os
from dataclasses import dataclass


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
    embedding_cache_dir: str = os.getenv("EMBEDDING_CACHE_DIR", r"E:\models")
    embedding_device: str | None = os.getenv("EMBEDDING_DEVICE") or None
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
    embedding_normalize: bool = (
        os.getenv("EMBEDDING_NORMALIZE", "true").lower() == "true"
    )


settings = Settings()

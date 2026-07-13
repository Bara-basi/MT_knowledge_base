from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


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


def _env_optional_float(name: str, default: float | None) -> float | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    if not value:
        return None
    return float(value)


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
    milvus_search_ef: int = int(os.getenv("MILVUS_SEARCH_EF", "64"))

    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "mtsco_knowledge_base")
    postgres_user: str = os.getenv("POSTGRES_USER", "mtsco")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "")
    database_url: str = os.getenv("DATABASE_URL", "")
    postgres_chat_table: str = os.getenv("POSTGRES_CHAT_TABLE", "chat_messages")
    postgres_conversation_topics_table: str = os.getenv(
        "POSTGRES_CONVERSATION_TOPICS_TABLE",
        "conversation_topics",
    )
    postgres_timezone: str = os.getenv("POSTGRES_TIMEZONE", os.getenv("PGTZ", "")).strip()
    chat_message_encryption_key: str = os.getenv(
        "CHAT_MESSAGE_ENCRYPTION_KEY",
        "change-me-before-production-chat-message-encryption-key",
    )
    conversation_topic_recent_limit: int = int(os.getenv("CONVERSATION_TOPIC_RECENT_LIMIT", "5"))
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")
    neo4j_graph_name: str = os.getenv("NEO4J_GRAPH_NAME", "MTSCO知识图谱")

    minio_endpoint: str = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
    minio_public_endpoint: str = os.getenv("MINIO_PUBLIC_ENDPOINT", minio_endpoint)
    minio_access_key_id: str = _get_first_env(
        "MINIO_ACCESS_KEY_ID",
        "MINIO_ROOT_USER",
        default="minioadmin",
    )
    minio_secret_access_key: str = _get_first_env(
        "MINIO_SECRET_ACCESS_KEY",
        "MINIO_ROOT_PASSWORD",
        default="minioadmin",
    )
    minio_bucket: str = _get_first_env(
        "MINIO_BUCKET",
        "APP_MINIO_BUCKET",
        default="knowledge-raw-docs",
    )
    minio_secure: bool = _env_bool("MINIO_SECURE", False)

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
    jieba_expanded_vocab_file: str = os.getenv(
        "JIEBA_EXPANDED_VOCAB_FILE",
        os.getenv(
            "JIEBA_EXPAND_VOCAB_FILE",
            str(Path("data") / "vocab" / "expanded_vocab.csv"),
        ),
    )
    jieba_expanded_vocab_freq: int = int(
        os.getenv(
            "JIEBA_EXPANDED_VOCAB_FREQ",
            os.getenv("JIEBA_EXPAND_VOCAB_FREQ", "100000"),
        )
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
    rerank_score_threshold: float | None = _env_optional_float(
        "RERANK_SCORE_THRESHOLD",
        0.0004,
    )
    rerank_score_cliff_delta: float = float(
        os.getenv("RERANK_SCORE_CLIFF_DELTA", "1")
    )
    retrieval_recall_multiplier: int = int(
        os.getenv("RETRIEVAL_RECALL_MULTIPLIER", "5")
    )
    skip_retrieval_warmup: bool = _env_bool("SKIP_RETRIEVAL_WARMUP", False)

    siliconflow_api_key: str = _get_first_env(
        "SILICONFLOW_API_KEY",
    )
    siliconflow_base_url: str = _get_first_env(
        "SILICONFLOW_BASE_URL",
        "SILICONFLOW_API_URL",
        default="https://api.siliconflow.cn/v1",
    ).rstrip("/")
    siliconflow_timeout: float = float(os.getenv("SILICONFLOW_TIMEOUT", "180"))
    siliconflow_connect_timeout: float = float(os.getenv("SILICONFLOW_CONNECT_TIMEOUT", "10"))
    siliconflow_read_timeout: float = float(
        os.getenv("SILICONFLOW_READ_TIMEOUT", str(siliconflow_timeout))
    )
    siliconflow_write_timeout: float = float(os.getenv("SILICONFLOW_WRITE_TIMEOUT", "60"))
    siliconflow_pool_timeout: float = float(os.getenv("SILICONFLOW_POOL_TIMEOUT", "10"))
    llm_api_key: str = _get_first_env(
        "KIMI_API_KEY",
        "LLM_API_KEY",
        "OPENAI_API_KEY",
        "SILICONFLOW_API_KEY",
    )
    llm_base_url: str = _get_first_env(
        "KIMI_BASE_URL",
        "KIMI_API_URL",
        "LLM_BASE_URL",
        "OPENAI_BASE_URL",
        "SILICONFLOW_BASE_URL",
        "SILICONFLOW_API_URL",
        default="https://api.moonshot.cn/v1",
    ).rstrip("/")
    llm_model: str = _get_first_env(
        "KIMI_MODEL",
        "LLM_MODEL",
        "OPENAI_MODEL",
        "SILICONFLOW_MODEL",
        default="kimi-k2.6",
    )
    immediate_feedback_model: str = os.getenv(
        "IMMEDIATE_FEEDBACK_MODEL",
        "deepseek-ai/DeepSeek-V4-Flash",
    )
    immediate_feedback_timeout: float = float(os.getenv("IMMEDIATE_FEEDBACK_TIMEOUT", "3"))
    immediate_feedback_connect_timeout: float = float(
        os.getenv("IMMEDIATE_FEEDBACK_CONNECT_TIMEOUT", "1")
    )
    immediate_feedback_max_tokens: int = int(os.getenv("IMMEDIATE_FEEDBACK_MAX_TOKENS", "80"))
    immediate_feedback_enable_thinking: str = os.getenv(
        "IMMEDIATE_FEEDBACK_ENABLE_THINKING",
        "off",
    )
    immediate_feedback_retry_without_thinking_options: bool = _env_bool(
        "IMMEDIATE_FEEDBACK_RETRY_WITHOUT_THINKING_OPTIONS",
        True,
    )

    n8n_query_webhook_url: str = os.getenv(
        "N8N_QUERY_WEBHOOK_URL",
        "http://n8n:5678/webhook/fastapi-test",
    )
    n8n_query_timeout: float = float(os.getenv("N8N_QUERY_TIMEOUT", "600"))
    n8n_query_connect_timeout: float = float(os.getenv("N8N_QUERY_CONNECT_TIMEOUT", "10"))
    n8n_api_base_url: str = os.getenv("N8N_API_BASE_URL", "").rstrip("/")
    n8n_api_key: str = os.getenv("N8N_API_KEY", "")
    n8n_progress_enabled: bool = _env_bool("N8N_PROGRESS_ENABLED", True)
    n8n_progress_poll_interval: float = float(os.getenv("N8N_PROGRESS_POLL_INTERVAL", "0.5"))
    n8n_progress_lookback_seconds: float = float(
        os.getenv("N8N_PROGRESS_LOOKBACK_SECONDS", "5")
    )
    n8n_query_workflow_id: str = os.getenv("N8N_QUERY_WORKFLOW_ID", "KZKRj0Y1QW2xTS0J")
    n8n_retrieval_workflow_id: str = os.getenv("N8N_RETRIEVAL_WORKFLOW_ID", "0N11uTxPrWDr7G9O")

    feishu_app_id: str = _get_first_env("FEISHU_APP_ID", "LARK_APP_ID")
    feishu_app_secret: str = _get_first_env("FEISHU_APP_SECRET", "LARK_APP_SECRET")
    feishu_verification_token: str = _get_first_env(
        "FEISHU_VERIFICATION_TOKEN",
        "LARK_VERIFICATION_TOKEN",
    )
    feishu_base_url: str = os.getenv("FEISHU_BASE_URL", "https://open.feishu.cn").rstrip("/")
    feishu_timeout: float = float(os.getenv("FEISHU_TIMEOUT", "30"))
    feishu_feedback_form_url: str = os.getenv(
        "FEISHU_FEEDBACK_FORM_URL",
        "https://tmqhw1h9zt.feishu.cn/wiki/LbjCwUPA6iUbF5k2SFbcowT8nne",
    ).strip()
    feishu_feedback_window_seconds: float = float(
        os.getenv("FEISHU_FEEDBACK_WINDOW_SECONDS", "1800")
    )
    daily_report_enabled: bool = _env_bool("DAILY_REPORT_ENABLED", True)
    daily_report_target_union_id: str = _get_first_env(
        "DAILY_REPORT_TARGET_UNION_ID",
        "DAILY_REPORT_TARGET_OPEN_ID",
        default="on_ebc25d5669cabb3440819db2cfaa5c7c",
    ).strip()
    daily_report_target_session_id: str = os.getenv(
        "DAILY_REPORT_TARGET_SESSION_ID",
        "oc_b4325718ab22291bc7625ebd63d6f915",
    ).strip()
    daily_report_timezone: str = os.getenv("DAILY_REPORT_TIMEZONE", "Asia/Shanghai").strip()
    public_base_url: str = os.getenv(
        "PUBLIC_BASE_URL",
        "https://shopper-washable-crock.ngrok-free.dev",
    ).rstrip("/")
    api_route_prefix: str = os.getenv("API_ROUTE_PREFIX", "").rstrip("/")
    feishu_route_prefix: str = os.getenv("FEISHU_ROUTE_PREFIX", "").rstrip("/")
    dev_proxy_target: str = os.getenv("DEV_PROXY_TARGET", "").rstrip("/")
    dev_proxy_timeout: float = float(os.getenv("DEV_PROXY_TIMEOUT", "600"))


settings = Settings()

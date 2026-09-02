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
    postgres_external_chat_table: str = os.getenv(
        "POSTGRES_EXTERNAL_CHAT_TABLE",
        "chat_messages_external",
    )
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
    neo4j_query_timeout: float = float(os.getenv("NEO4J_QUERY_TIMEOUT", "10"))
    neo4j_graph_name: str = os.getenv("NEO4J_GRAPH_NAME", "MTSCO知识图谱")

    minio_endpoint: str = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
    minio_public_endpoint: str = os.getenv("MINIO_PUBLIC_ENDPOINT", minio_endpoint)
    minio_access_key_id: str = _get_first_env(
        "MINIO_ACCESS_KEY_ID",
        "MINIO_ROOT_USER",
        default="",
    )
    minio_secret_access_key: str = _get_first_env(
        "MINIO_SECRET_ACCESS_KEY",
        "MINIO_ROOT_PASSWORD",
        default="",
    )
    minio_bucket: str = _get_first_env(
        "MINIO_BUCKET",
        "APP_MINIO_BUCKET",
        default="knowledge-raw-docs",
    )
    minio_standard_asset_bucket: str = os.getenv(
        "MINIO_STANDARD_ASSET_BUCKET",
        "knowledge-standard-assets",
    )
    minio_processed_document_bucket: str = os.getenv(
        "MINIO_PROCESSED_DOCUMENT_BUCKET",
        "knowledge-processed-docs",
    )
    minio_secure: bool = _env_bool("MINIO_SECURE", False)

    # Canonical raw knowledge documents are stored in Alibaba Cloud OSS.
    aliyun_oss_endpoint: str = os.getenv("ALIYUN_OSS_ENDPOINT", "").strip()
    aliyun_access_key_id: str = os.getenv("ALIYUN_ACCESS_KEY_ID", "").strip()
    aliyun_access_key_secret: str = os.getenv("ALIYUN_ACCESS_KEY_SECRET", "").strip()
    aliyun_raw_data_bucket: str = os.getenv("ALIYUN_RAW_DATA_BUCKET", "").strip()

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
        # Scores from bge-reranker-v2-m3 are not calibrated probabilities, but
        # a 0.7 gap is a strong relevance boundary in the configured score
        # scale.  Keep the runtime default aligned with the documented
        # deployment configuration; 1.0 effectively disables tail trimming
        # for most requests and lets low-relevance chunks reach the QA agent.
        os.getenv("RERANK_SCORE_CLIFF_DELTA", "0.7")
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

    harness_enabled: bool = _env_bool("HARNESS_ENABLED", False)
    harness_gateway_url: str = os.getenv("HARNESS_GATEWAY_URL", "").rstrip("/")
    harness_model: str = os.getenv("HARNESS_MODEL", "deepseek-v4-flash")
    harness_provider: str = os.getenv("HARNESS_PROVIDER", "deepseek")
    harness_timeout: float = float(os.getenv("HARNESS_TIMEOUT", "600"))
    harness_idle_seconds: int = int(os.getenv("HARNESS_IDLE_SECONDS", "25200"))
    harness_session_root: str = os.getenv("HARNESS_SESSION_ROOT", "data/harness_sessions")
    # The bundled Harness installation patch resumes JSONL sessions natively.
    # Set false only as an emergency rollback to inject the application
    # transcript into a create-only third-party runtime.
    harness_native_jsonl_resume: bool = _env_bool("HARNESS_NATIVE_JSONL_RESUME", True)
    harness_workdir: str = os.getenv("HARNESS_WORKDIR", "data/processing")
    harness_memory_bucket: str = os.getenv("HARNESS_MEMORY_BUCKET", "knowledge-chat-memory")
    harness_memory_summary_model: str = os.getenv(
        "HARNESS_MEMORY_SUMMARY_MODEL", os.getenv("HARNESS_MODEL", "deepseek-v4-flash")
    )
    harness_memory_summary_max_tokens: int = int(
        os.getenv("HARNESS_MEMORY_SUMMARY_MAX_TOKENS", "1200")
    )
    harness_scheduler_interval_seconds: int = int(os.getenv("HARNESS_SCHEDULER_INTERVAL_SECONDS", "300"))
    harness_context_archive_tokens: int = int(
        os.getenv("HARNESS_CONTEXT_ARCHIVE_TOKENS", "90000")
    )
    # Zero keeps the developer process unconstrained. Production pins this to
    # two so every API/worker process shares the same PostgreSQL-backed budget.
    harness_global_concurrency: int = int(os.getenv("HARNESS_GLOBAL_CONCURRENCY", "0"))
    harness_attachment_root: str = os.getenv(
        "HARNESS_ATTACHMENT_ROOT", "data/harness_attachments"
    )
    harness_attachment_max_bytes: int = int(
        os.getenv("HARNESS_ATTACHMENT_MAX_BYTES", str(50 * 1024 * 1024))
    )
    harness_attachment_ttl_seconds: int = int(
        os.getenv("HARNESS_ATTACHMENT_TTL_SECONDS", "86400")
    )
    harness_attachment_api_token: str = os.getenv(
        "HARNESS_ATTACHMENT_API_TOKEN", ""
    ).strip()

    feishu_durable_queue_enabled: bool = _env_bool(
        "FEISHU_DURABLE_QUEUE_ENABLED", False
    )
    answer_worker_concurrency: int = int(os.getenv("ANSWER_WORKER_CONCURRENCY", "1"))
    answer_worker_poll_seconds: float = float(os.getenv("ANSWER_WORKER_POLL_SECONDS", "1"))
    answer_job_lease_seconds: int = int(os.getenv("ANSWER_JOB_LEASE_SECONDS", "900"))
    answer_job_max_attempts: int = int(os.getenv("ANSWER_JOB_MAX_ATTEMPTS", "3"))

    feishu_rate_limit_per_minute: int = int(
        os.getenv("FEISHU_RATE_LIMIT_PER_MINUTE", "10")
    )
    feishu_rate_limit_burst: int = int(os.getenv("FEISHU_RATE_LIMIT_BURST", "3"))
    external_rate_limit_per_minute: int = int(
        os.getenv("EXTERNAL_RATE_LIMIT_PER_MINUTE", "60")
    )
    shared_rate_limit_enabled: bool = _env_bool("SHARED_RATE_LIMIT_ENABLED", False)

    feishu_app_id: str = _get_first_env("FEISHU_APP_ID", "LARK_APP_ID")
    feishu_app_secret: str = _get_first_env("FEISHU_APP_SECRET", "LARK_APP_SECRET")
    feishu_verification_token: str = _get_first_env(
        "FEISHU_VERIFICATION_TOKEN",
        "LARK_VERIFICATION_TOKEN",
    )
    feishu_base_url: str = os.getenv("FEISHU_BASE_URL", "https://open.feishu.cn").rstrip("/")
    feishu_timeout: float = float(os.getenv("FEISHU_TIMEOUT", "30"))
    feishu_trust_env: bool = _env_bool("FEISHU_TRUST_ENV", False)
    feishu_connect_retries: int = max(
        0,
        int(os.getenv("FEISHU_CONNECT_RETRIES", "2")),
    )
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
    daily_report_departments: str = os.getenv("DAILY_REPORT_DEPARTMENTS", "").strip()
    weekly_report_enabled: bool = _env_bool("WEEKLY_REPORT_ENABLED", True)
    weekly_report_target_union_id: str = _get_first_env(
        "WEEKLY_REPORT_TARGET_UNION_ID",
        "WEEKLY_REPORT_TARGET_OPEN_ID",
        "DAILY_REPORT_TARGET_UNION_ID",
        "DAILY_REPORT_TARGET_OPEN_ID",
        default="on_ebc25d5669cabb3440819db2cfaa5c7c",
    ).strip()
    weekly_report_target_session_id: str = _get_first_env(
        "WEEKLY_REPORT_TARGET_SESSION_ID",
        "DAILY_REPORT_TARGET_SESSION_ID",
        default="oc_b4325718ab22291bc7625ebd63d6f915",
    ).strip()
    weekly_report_timezone: str = _get_first_env(
        "WEEKLY_REPORT_TIMEZONE",
        "DAILY_REPORT_TIMEZONE",
        default="Asia/Shanghai",
    ).strip()
    weekly_report_departments: str = _get_first_env(
        "WEEKLY_REPORT_DEPARTMENTS",
        "DAILY_REPORT_DEPARTMENTS",
    ).strip()
    public_base_url: str = os.getenv(
        "PUBLIC_BASE_URL",
        "https://shopper-washable-crock.ngrok-free.dev",
    ).rstrip("/")
    api_route_prefix: str = os.getenv("API_ROUTE_PREFIX", "").rstrip("/")
    feishu_route_prefix: str = os.getenv("FEISHU_ROUTE_PREFIX", "").rstrip("/")
    dev_proxy_target: str = os.getenv("DEV_PROXY_TARGET", "").rstrip("/")
    dev_proxy_timeout: float = float(os.getenv("DEV_PROXY_TIMEOUT", "600"))


settings = Settings()

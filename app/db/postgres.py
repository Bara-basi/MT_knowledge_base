from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import time
from typing import Any, Iterator
from uuid import UUID, uuid4

import psycopg
from psycopg import Connection, sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.core.config import settings


CHAT_MESSAGES_TABLE_COMMENT = "MTSCO knowledge base encrypted chat messages"
EXTERNAL_CHAT_MESSAGES_TABLE_COMMENT = (
    "MTSCO knowledge base encrypted external API chat messages"
)
EXTERNAL_CHAT_ID_PREFIX = "external:v1:"
CONVERSATION_TOPICS_TABLE = "conversation_topics"
CONVERSATION_TOPICS_TABLE_COMMENT = "Virtual conversation topics for Feishu chat sessions"
CHAT_TEXT_ENCRYPTION_PREFIX = "fernet:v1:"
FEISHU_ANSWER_FEEDBACK_TABLE = "feishu_answer_feedback_states"
FEISHU_ANSWER_JOBS_TABLE = "feishu_answer_jobs"
API_RATE_LIMITS_TABLE = "api_rate_limits"
HARNESS_SESSIONS_TABLE = "harness_sessions"
HARNESS_MEMORIES_TABLE = "harness_memories"


def get_postgres_connection() -> Connection[Any]:
    """Create a PostgreSQL connection from environment-backed settings."""

    options: dict[str, Any] = {}
    if settings.postgres_timezone:
        options["options"] = f"-c timezone={settings.postgres_timezone}"

    if settings.database_url:
        return psycopg.connect(settings.database_url, row_factory=dict_row, **options)

    return psycopg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
        row_factory=dict_row,
        **options,
    )


@contextmanager
def postgres_connection() -> Iterator[Connection[Any]]:
    conn = get_postgres_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def postgres_advisory_lock(lock_key: int) -> Iterator[None]:
    """Hold one cross-process PostgreSQL advisory lock for a long operation."""

    conn = get_postgres_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_lock(%s)", (lock_key,))
        yield
    finally:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))
        finally:
            conn.close()


@contextmanager
def postgres_capacity_slot(
    *, namespace: int, slots: int, timeout_seconds: float
) -> Iterator[int | None]:
    """Lease one of ``slots`` shared advisory locks, or wait until timeout."""

    if slots <= 0:
        yield None
        return
    deadline = time.monotonic() + max(0.1, timeout_seconds)
    conn = get_postgres_connection()
    acquired: int | None = None
    try:
        while acquired is None:
            with conn.cursor() as cur:
                for slot in range(slots):
                    lock_key = namespace + slot
                    cur.execute("SELECT pg_try_advisory_lock(%s) AS acquired", (lock_key,))
                    row = cur.fetchone() or {}
                    if row.get("acquired"):
                        acquired = lock_key
                        break
            if acquired is not None:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for a shared Harness capacity slot")
            time.sleep(0.25)
        yield acquired - namespace
    finally:
        if acquired is not None:
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT pg_advisory_unlock(%s)", (acquired,))
            finally:
                conn.close()
        else:
            conn.close()


def ensure_chat_messages_table(table_name: str | None = None) -> dict[str, Any]:
    """Create the chat message table if it does not already exist."""

    table_name = table_name or settings.postgres_chat_table
    table = sql.Identifier(table_name)

    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {table} (
                        message_id UUID NOT NULL DEFAULT gen_random_uuid(),
                        source_message_id TEXT,
                        user_id VARCHAR(128) NOT NULL,
                        user_name TEXT,
                        feishu_user_id TEXT,
                        feishu_open_id TEXT,
                        department_ids TEXT[],
                        department_names TEXT[],
                        job_title TEXT,
                        employee_type TEXT,
                        user_profile_updated_at TIMESTAMPTZ,
                        session_id VARCHAR(128) NOT NULL,
                        conversation_id VARCHAR(128) NOT NULL,
                        topic_id UUID,
                        create_time TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL DEFAULT ''
                    )
                    """
                ).format(table=table)
            )
            cur.execute(
                sql.SQL(
                    "ALTER TABLE {table} ADD COLUMN IF NOT EXISTS message_id "
                    "UUID DEFAULT gen_random_uuid()"
                ).format(table=table)
            )
            cur.execute(
                sql.SQL("UPDATE {table} SET message_id = gen_random_uuid() WHERE message_id IS NULL").format(
                    table=table
                )
            )
            cur.execute(
                sql.SQL("ALTER TABLE {table} ALTER COLUMN message_id SET NOT NULL").format(
                    table=table
                )
            )
            cur.execute(
                sql.SQL("ALTER TABLE {table} ALTER COLUMN message_id SET DEFAULT gen_random_uuid()").format(
                    table=table
                )
            )
            cur.execute(
                sql.SQL("CREATE UNIQUE INDEX IF NOT EXISTS {index} ON {table} (message_id)").format(
                    index=sql.Identifier(f"{table_name}_message_id_uidx"),
                    table=table,
                )
            )
            cur.execute(
                sql.SQL("ALTER TABLE {table} ADD COLUMN IF NOT EXISTS source_message_id TEXT").format(
                    table=table
                )
            )
            cur.execute(
                sql.SQL(
                    "CREATE UNIQUE INDEX IF NOT EXISTS {index} ON {table} (source_message_id) "
                    "WHERE source_message_id IS NOT NULL"
                ).format(
                    index=sql.Identifier(f"{table_name}_source_message_id_uidx"),
                    table=table,
                )
            )
            cur.execute(
                sql.SQL("ALTER TABLE {table} ADD COLUMN IF NOT EXISTS user_name TEXT").format(
                    table=table
                )
            )
            profile_columns = (
                ("feishu_user_id", "TEXT"),
                ("feishu_open_id", "TEXT"),
                ("department_ids", "TEXT[]"),
                ("department_names", "TEXT[]"),
                ("job_title", "TEXT"),
                ("employee_type", "TEXT"),
                ("user_profile_updated_at", "TIMESTAMPTZ"),
            )
            for column_name, column_type in profile_columns:
                cur.execute(
                    sql.SQL("ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {type}").format(
                        table=table,
                        column=sql.Identifier(column_name),
                        type=sql.SQL(column_type),
                    )
                )
            cur.execute(
                sql.SQL("ALTER TABLE {table} ADD COLUMN IF NOT EXISTS topic_id UUID").format(
                    table=table
                )
            )
            cur.execute(
                sql.SQL(
                    "ALTER TABLE {table} ADD COLUMN IF NOT EXISTS create_time "
                    "TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP"
                ).format(table=table)
            )
            cur.execute(
                sql.SQL(
                    "UPDATE {table} SET create_time = CURRENT_TIMESTAMP "
                    "WHERE create_time IS NULL"
                ).format(table=table)
            )
            cur.execute(
                sql.SQL("ALTER TABLE {table} ALTER COLUMN create_time SET NOT NULL").format(
                    table=table
                )
            )
            cur.execute(
                sql.SQL("ALTER TABLE {table} ALTER COLUMN answer SET DEFAULT ''").format(
                    table=table
                )
            )
            cur.execute(
                sql.SQL("ALTER TABLE {table} DROP COLUMN IF EXISTS fallback").format(
                    table=table
                )
            )
            cur.execute(
                sql.SQL("ALTER TABLE {table} DROP COLUMN IF EXISTS reason").format(
                    table=table
                )
            )
            cur.execute(
                sql.SQL("COMMENT ON TABLE {table} IS {comment}").format(
                    table=table,
                    comment=sql.Literal(CHAT_MESSAGES_TABLE_COMMENT),
                )
            )
            cur.execute(
                sql.SQL(
                    """
                    CREATE INDEX IF NOT EXISTS {index}
                    ON {table} (topic_id, create_time DESC)
                    WHERE topic_id IS NOT NULL
                    """
                ).format(
                    index=sql.Identifier(f"{table_name}_topic_recent_idx"),
                    table=table,
                )
            )
            cur.execute(
                """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = %s
                ORDER BY ordinal_position
                """,
                (table_name,),
            )
            columns = cur.fetchall()

    return {
        "table_name": table_name,
        "columns": columns,
    }


def ensure_external_chat_messages_table(
    table_name: str | None = None,
) -> dict[str, Any]:
    """Create the privacy-minimized message table used only by external APIs."""

    table_name = table_name or settings.postgres_external_chat_table
    if table_name == settings.postgres_chat_table:
        raise ValueError("external chat table must differ from the Feishu chat table")
    table = sql.Identifier(table_name)

    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {table} (
                        message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        service_id VARCHAR(128) NOT NULL,
                        user_id VARCHAR(128) NOT NULL,
                        session_id VARCHAR(128) NOT NULL,
                        conversation_id VARCHAR(128) NOT NULL,
                        topic_id UUID,
                        create_time TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL DEFAULT ''
                    )
                    """
                ).format(table=table)
            )
            required_columns = (
                ("message_id", "UUID DEFAULT gen_random_uuid()"),
                ("service_id", "VARCHAR(128)"),
                ("user_id", "VARCHAR(128)"),
                ("session_id", "VARCHAR(128)"),
                ("conversation_id", "VARCHAR(128)"),
                ("topic_id", "UUID"),
                ("create_time", "TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP"),
                ("question", "TEXT"),
                ("answer", "TEXT DEFAULT ''"),
            )
            for column_name, column_type in required_columns:
                cur.execute(
                    sql.SQL(
                        "ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {type}"
                    ).format(
                        table=table,
                        column=sql.Identifier(column_name),
                        type=sql.SQL(column_type),
                    )
                )

            # The external table intentionally must not accumulate Feishu profile data.
            for column_name in (
                "user_name",
                "feishu_user_id",
                "feishu_open_id",
                "department_ids",
                "department_names",
                "job_title",
                "employee_type",
                "user_profile_updated_at",
            ):
                cur.execute(
                    sql.SQL("ALTER TABLE {table} DROP COLUMN IF EXISTS {column}").format(
                        table=table,
                        column=sql.Identifier(column_name),
                    )
                )

            cur.execute(
                sql.SQL("UPDATE {table} SET message_id = gen_random_uuid() WHERE message_id IS NULL").format(
                    table=table
                )
            )
            cur.execute(
                sql.SQL("UPDATE {table} SET answer = '' WHERE answer IS NULL").format(table=table)
            )
            cur.execute(
                sql.SQL("ALTER TABLE {table} ALTER COLUMN message_id SET NOT NULL").format(
                    table=table
                )
            )
            cur.execute(
                sql.SQL("ALTER TABLE {table} ALTER COLUMN answer SET DEFAULT ''").format(table=table)
            )
            cur.execute(
                sql.SQL("ALTER TABLE {table} ALTER COLUMN answer SET NOT NULL").format(table=table)
            )
            for column_name in (
                "service_id",
                "user_id",
                "session_id",
                "conversation_id",
                "question",
            ):
                cur.execute(
                    sql.SQL("ALTER TABLE {table} ALTER COLUMN {column} SET NOT NULL").format(
                        table=table,
                        column=sql.Identifier(column_name),
                    )
                )
            cur.execute(
                sql.SQL("COMMENT ON TABLE {table} IS {comment}").format(
                    table=table,
                    comment=sql.Literal(EXTERNAL_CHAT_MESSAGES_TABLE_COMMENT),
                )
            )
            cur.execute(
                sql.SQL(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS {index}
                    ON {table} (message_id)
                    """
                ).format(
                    index=sql.Identifier(f"{table_name}_message_id_uidx"),
                    table=table,
                )
            )
            cur.execute(
                sql.SQL(
                    """
                    CREATE INDEX IF NOT EXISTS {index}
                    ON {table} (service_id, user_id, session_id, create_time DESC)
                    """
                ).format(
                    index=sql.Identifier(f"{table_name}_service_session_recent_idx"),
                    table=table,
                )
            )
            cur.execute(
                sql.SQL(
                    """
                    CREATE INDEX IF NOT EXISTS {index}
                    ON {table} (topic_id, create_time DESC)
                    WHERE topic_id IS NOT NULL
                    """
                ).format(
                    index=sql.Identifier(f"{table_name}_topic_recent_idx"),
                    table=table,
                )
            )
            cur.execute(
                """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = %s
                ORDER BY ordinal_position
                """,
                (table_name,),
            )
            columns = cur.fetchall()

    return {"table_name": table_name, "columns": columns}


def is_external_chat_identity(user_id: str | None) -> bool:
    return str(user_id or "").startswith(EXTERNAL_CHAT_ID_PREFIX)


def chat_message_table_for_identity(user_id: str | None) -> str:
    if is_external_chat_identity(user_id):
        return settings.postgres_external_chat_table
    return settings.postgres_chat_table


def ensure_chat_user_profile_columns(table_name: str | None = None) -> dict[str, Any]:
    """Add only the Feishu profile columns needed by the backfill job."""

    table_name = table_name or settings.postgres_chat_table
    table = sql.Identifier(table_name)
    profile_columns = (
        ("feishu_user_id", "TEXT"),
        ("feishu_open_id", "TEXT"),
        ("department_ids", "TEXT[]"),
        ("department_names", "TEXT[]"),
        ("job_title", "TEXT"),
        ("employee_type", "TEXT"),
        ("user_profile_updated_at", "TIMESTAMPTZ"),
    )
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            for column_name, column_type in profile_columns:
                cur.execute(
                    sql.SQL("ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {type}").format(
                        table=table,
                        column=sql.Identifier(column_name),
                        type=sql.SQL(column_type),
                    )
                )
    return {"table_name": table_name, "columns": [name for name, _ in profile_columns]}


def list_chat_user_profile_backfill_candidates(
    *,
    table_name: str | None = None,
    union_ids: list[str] | tuple[str, ...] | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    """List distinct Feishu union IDs whose historical rows need enrichment."""

    table_name = table_name or settings.postgres_chat_table
    conditions = [sql.SQL("BTRIM(user_id) <> ''"), sql.SQL("user_id <> 'unknown-user'")]
    parameters: list[Any] = []

    normalized_ids = tuple(dict.fromkeys(str(item).strip() for item in union_ids or () if str(item).strip()))
    if normalized_ids:
        conditions.append(sql.SQL("user_id = ANY(%s)"))
        parameters.append(list(normalized_ids))
    if not force:
        conditions.append(sql.SQL("user_profile_updated_at IS NULL"))

    query = sql.SQL(
        """
        SELECT user_id AS union_id, COUNT(*)::INTEGER AS row_count
        FROM {table}
        WHERE {conditions}
        GROUP BY user_id
        ORDER BY user_id
        """
    ).format(
        table=sql.Identifier(table_name),
        conditions=sql.SQL(" AND ").join(conditions),
    )
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, parameters)
            return list(cur.fetchall())


def update_chat_user_profile(
    *,
    union_id: str,
    feishu_user_id: str | None,
    feishu_open_id: str | None,
    user_name: str | None,
    department_ids: list[str] | tuple[str, ...],
    department_names: list[str] | tuple[str, ...],
    job_title: str | None,
    employee_type: Any,
    table_name: str | None = None,
) -> int:
    """Apply one Feishu user profile to every historical message for that user."""

    table_name = table_name or settings.postgres_chat_table
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    UPDATE {table}
                    SET user_name = COALESCE(%s, user_name),
                        feishu_user_id = %s,
                        feishu_open_id = %s,
                        department_ids = %s,
                        department_names = %s,
                        job_title = %s,
                        employee_type = %s,
                        user_profile_updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = %s
                    """
                ).format(table=sql.Identifier(table_name)),
                (
                    user_name,
                    feishu_user_id,
                    feishu_open_id,
                    list(department_ids),
                    list(department_names),
                    job_title,
                    None if employee_type is None else str(employee_type),
                    union_id,
                ),
            )
            return cur.rowcount


def ensure_conversation_topics_table(
    table_name: str | None = None,
) -> dict[str, Any]:
    """Create the virtual conversation topic table if needed."""

    table_name = table_name or settings.postgres_conversation_topics_table
    table = sql.Identifier(table_name)

    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {table} (
                        topic_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        user_id VARCHAR(128) NOT NULL,
                        session_id VARCHAR(128) NOT NULL,
                        topic TEXT NOT NULL,
                        summary TEXT NOT NULL DEFAULT '',
                        started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        last_message_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        message_count INTEGER NOT NULL DEFAULT 0,
                        status VARCHAR(16) NOT NULL DEFAULT 'active',
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        CONSTRAINT conversation_topics_message_count_nonnegative
                            CHECK (message_count >= 0),
                        CONSTRAINT conversation_topics_status_check
                            CHECK (status IN ('active', 'archived'))
                    )
                    """
                ).format(table=table)
            )
            cur.execute(
                sql.SQL("ALTER TABLE {table} ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ").format(
                    table=table
                )
            )
            cur.execute(
                sql.SQL(
                    "UPDATE {table} SET updated_at = COALESCE(updated_at, started_at, CURRENT_TIMESTAMP) "
                    "WHERE updated_at IS NULL"
                ).format(table=table)
            )
            cur.execute(
                sql.SQL("ALTER TABLE {table} ALTER COLUMN updated_at SET DEFAULT CURRENT_TIMESTAMP").format(
                    table=table
                )
            )
            cur.execute(
                sql.SQL("ALTER TABLE {table} ALTER COLUMN updated_at SET NOT NULL").format(
                    table=table
                )
            )
            cur.execute(
                sql.SQL(
                    "ALTER TABLE {table} ADD COLUMN IF NOT EXISTS last_message_at TIMESTAMPTZ"
                ).format(table=table)
            )
            cur.execute(
                sql.SQL(
                    "UPDATE {table} SET last_message_at = COALESCE(last_message_at, updated_at, started_at, CURRENT_TIMESTAMP) "
                    "WHERE last_message_at IS NULL"
                ).format(table=table)
            )
            cur.execute(
                sql.SQL(
                    "ALTER TABLE {table} ALTER COLUMN last_message_at SET DEFAULT CURRENT_TIMESTAMP"
                ).format(table=table)
            )
            cur.execute(
                sql.SQL("ALTER TABLE {table} ALTER COLUMN last_message_at SET NOT NULL").format(
                    table=table
                )
            )
            cur.execute(
                sql.SQL("ALTER TABLE {table} ADD COLUMN IF NOT EXISTS message_count INTEGER").format(
                    table=table
                )
            )
            cur.execute(
                sql.SQL(
                    "UPDATE {table} SET message_count = COALESCE(message_count, 0) "
                    "WHERE message_count IS NULL"
                ).format(table=table)
            )
            cur.execute(
                sql.SQL("ALTER TABLE {table} ALTER COLUMN message_count SET DEFAULT 0").format(
                    table=table
                )
            )
            cur.execute(
                sql.SQL("ALTER TABLE {table} ALTER COLUMN message_count SET NOT NULL").format(
                    table=table
                )
            )
            cur.execute(
                sql.SQL("ALTER TABLE {table} ADD COLUMN IF NOT EXISTS status VARCHAR(16)").format(
                    table=table
                )
            )
            cur.execute(
                sql.SQL(
                    "UPDATE {table} SET status = COALESCE(NULLIF(status, ''), 'active') "
                    "WHERE status IS NULL OR status = ''"
                ).format(table=table)
            )
            cur.execute(
                sql.SQL("ALTER TABLE {table} ALTER COLUMN status SET DEFAULT 'active'").format(
                    table=table
                )
            )
            cur.execute(
                sql.SQL("ALTER TABLE {table} ALTER COLUMN status SET NOT NULL").format(
                    table=table
                )
            )
            cur.execute(
                sql.SQL("ALTER TABLE {table} ADD COLUMN IF NOT EXISTS metadata JSONB").format(
                    table=table
                )
            )
            cur.execute(
                sql.SQL(
                    "UPDATE {table} SET metadata = COALESCE(metadata, '{{}}'::jsonb) "
                    "WHERE metadata IS NULL"
                ).format(table=table)
            )
            cur.execute(
                sql.SQL("ALTER TABLE {table} ALTER COLUMN metadata SET DEFAULT '{{}}'::jsonb").format(
                    table=table
                )
            )
            cur.execute(
                sql.SQL("ALTER TABLE {table} ALTER COLUMN metadata SET NOT NULL").format(
                    table=table
                )
            )
            cur.execute(
                sql.SQL("COMMENT ON TABLE {table} IS {comment}").format(
                    table=table,
                    comment=sql.Literal(CONVERSATION_TOPICS_TABLE_COMMENT),
                )
            )
            cur.execute(
                sql.SQL(
                    """
                    CREATE INDEX IF NOT EXISTS {index}
                    ON {table} (user_id, session_id, last_message_at DESC)
                    WHERE status = 'active'
                    """
                ).format(
                    index=sql.Identifier(f"{table_name}_active_recent_idx"),
                    table=table,
                )
            )
            cur.execute(
                sql.SQL(
                    """
                    CREATE INDEX IF NOT EXISTS {index}
                    ON {table} (user_id, session_id, started_at DESC)
                    """
                ).format(
                    index=sql.Identifier(f"{table_name}_started_at_idx"),
                    table=table,
                )
            )
            cur.execute(
                """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = %s
                ORDER BY ordinal_position
                """,
                (table_name,),
            )
            columns = cur.fetchall()

    return {
        "table_name": table_name,
        "columns": columns,
    }


def create_conversation_topic(
    *,
    user_id: str,
    session_id: str,
    topic: str,
    summary: str = "",
    topic_id: UUID | str | None = None,
    started_at: datetime | None = None,
    metadata: dict[str, Any] | None = None,
    table_name: str | None = None,
) -> dict[str, Any]:
    """Insert a virtual topic row and return the persisted record."""

    table_name = table_name or settings.postgres_conversation_topics_table
    table = sql.Identifier(table_name)
    topic_id = topic_id or uuid4()
    metadata = metadata or {}
    _require_encrypted_topic_text("topic", topic)
    _require_encrypted_topic_text("summary", summary)

    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {table} (
                        topic_id,
                        user_id,
                        session_id,
                        topic,
                        summary,
                        started_at,
                        updated_at,
                        last_message_at,
                        message_count,
                        status,
                        metadata
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        COALESCE(%s, CURRENT_TIMESTAMP),
                        CURRENT_TIMESTAMP,
                        CURRENT_TIMESTAMP,
                        0,
                        'active',
                        %s
                    )
                    RETURNING
                        topic_id,
                        user_id,
                        session_id,
                        topic,
                        summary,
                        started_at,
                        updated_at,
                        last_message_at,
                        message_count,
                        status,
                        metadata
                    """
                ).format(table=table),
                (
                    topic_id,
                    user_id,
                    session_id,
                    topic,
                    summary,
                    started_at,
                    Jsonb(metadata),
                ),
            )
            row = cur.fetchone()

    return dict(row or {})


def update_conversation_topic(
    *,
    topic_id: UUID | str,
    topic: str | None = None,
    summary: str | None = None,
    message_increment: int = 1,
    metadata: dict[str, Any] | None = None,
    table_name: str | None = None,
) -> dict[str, Any]:
    """Refresh a topic after a new turn has been assigned to it."""

    table_name = table_name or settings.postgres_conversation_topics_table
    table = sql.Identifier(table_name)
    if topic is not None:
        _require_encrypted_topic_text("topic", topic)
    if summary is not None:
        _require_encrypted_topic_text("summary", summary)

    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    UPDATE {table}
                    SET topic = COALESCE(%s, topic),
                        summary = COALESCE(%s, summary),
                        updated_at = CURRENT_TIMESTAMP,
                        last_message_at = CURRENT_TIMESTAMP,
                        message_count = message_count + %s,
                        metadata = CASE
                            WHEN %s::jsonb IS NULL THEN metadata
                            ELSE metadata || %s::jsonb
                        END
                    WHERE topic_id = %s
                    RETURNING
                        topic_id,
                        user_id,
                        session_id,
                        topic,
                        summary,
                        started_at,
                        updated_at,
                        last_message_at,
                        message_count,
                        status,
                        metadata
                    """
                ).format(table=table),
                (
                    topic,
                    summary,
                    message_increment,
                    Jsonb(metadata) if metadata is not None else None,
                    Jsonb(metadata) if metadata is not None else None,
                    topic_id,
                ),
            )
            row = cur.fetchone()

    return dict(row or {})


def list_recent_conversation_topics(
    *,
    user_id: str,
    session_id: str,
    limit: int = 5,
    table_name: str | None = None,
) -> list[dict[str, Any]]:
    """Return the active recent topics that can still be considered for context stitching."""

    table_name = table_name or settings.postgres_conversation_topics_table
    table = sql.Identifier(table_name)
    limit = max(1, min(int(limit), 50))

    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT
                        topic_id,
                        user_id,
                        session_id,
                        topic,
                        summary,
                        started_at,
                        updated_at,
                        last_message_at,
                        message_count,
                        status,
                        metadata
                    FROM {table}
                    WHERE user_id = %s
                      AND session_id = %s
                      AND status = 'active'
                    ORDER BY last_message_at DESC, started_at DESC
                    LIMIT %s
                    """
                ).format(table=table),
                (user_id, session_id, limit),
            )
            rows = cur.fetchall()

    return [dict(row) for row in rows]


def get_conversation_topic(
    *,
    topic_id: UUID | str,
    user_id: str,
    session_id: str,
    table_name: str | None = None,
) -> dict[str, Any]:
    """Return one active topic scoped to the Feishu user/session."""

    table_name = table_name or settings.postgres_conversation_topics_table
    table = sql.Identifier(table_name)

    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT
                        topic_id,
                        user_id,
                        session_id,
                        topic,
                        summary,
                        started_at,
                        updated_at,
                        last_message_at,
                        message_count,
                        status,
                        metadata
                    FROM {table}
                    WHERE topic_id = %s
                      AND user_id = %s
                      AND session_id = %s
                      AND status = 'active'
                    """
                ).format(table=table),
                (topic_id, user_id, session_id),
            )
            row = cur.fetchone()

    return dict(row or {})


def touch_conversation_topic_activity(
    *,
    topic_id: UUID | str,
    user_id: str,
    session_id: str,
    table_name: str | None = None,
) -> dict[str, Any]:
    """Mark a topic as active after a completed turn is assigned to it."""

    table_name = table_name or settings.postgres_conversation_topics_table
    table = sql.Identifier(table_name)

    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    UPDATE {table}
                    SET updated_at = CURRENT_TIMESTAMP,
                        last_message_at = CURRENT_TIMESTAMP,
                        message_count = message_count + 1
                    WHERE topic_id = %s
                      AND user_id = %s
                      AND session_id = %s
                      AND status = 'active'
                    RETURNING
                        topic_id,
                        user_id,
                        session_id,
                        topic,
                        summary,
                        started_at,
                        updated_at,
                        last_message_at,
                        message_count,
                        status,
                        metadata
                    """
                ).format(table=table),
                (topic_id, user_id, session_id),
            )
            row = cur.fetchone()

    return dict(row or {})


def _require_encrypted_topic_text(field_name: str, value: str) -> None:
    if not value.startswith(CHAT_TEXT_ENCRYPTION_PREFIX):
        raise ValueError(
            f"conversation topic {field_name} must be encrypted with chat text encryption"
        )


def insert_chat_message(
    *,
    user_id: str,
    session_id: str,
    conversation_id: str,
    question: str,
    answer: str,
    user_name: str | None = None,
    topic_id: UUID | str | None = None,
    table_name: str | None = None,
) -> dict[str, Any]:
    """Insert one chat message row and return the inserted values."""

    table_name = table_name or settings.postgres_chat_table
    table = sql.Identifier(table_name)

    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {table} (
                        user_id,
                        user_name,
                        session_id,
                        conversation_id,
                        topic_id,
                        question,
                        answer
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING
                        message_id,
                        user_id,
                        user_name,
                        session_id,
                        conversation_id,
                        topic_id,
                        create_time,
                        question,
                        answer
                    """
                ).format(table=table),
                (
                    user_id,
                    user_name,
                    session_id,
                    conversation_id,
                    topic_id,
                    question,
                    answer,
                ),
            )
            row = cur.fetchone()

    return dict(row or {})


def create_external_chat_message(
    *,
    service_id: str,
    user_id: str,
    session_id: str,
    conversation_id: str,
    question: str,
    table_name: str | None = None,
) -> dict[str, Any]:
    """Create one pending external-API chat turn and return its stable id."""

    table_name = table_name or settings.postgres_external_chat_table
    table = sql.Identifier(table_name)
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {table} (
                        service_id,
                        user_id,
                        session_id,
                        conversation_id,
                        question
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING
                        message_id,
                        service_id,
                        user_id,
                        session_id,
                        conversation_id,
                        topic_id,
                        create_time,
                        question,
                        answer
                    """
                ).format(table=table),
                (service_id, user_id, session_id, conversation_id, question),
            )
            row = cur.fetchone()
    return dict(row or {})


def update_external_chat_answer(
    *,
    message_id: UUID | str,
    service_id: str,
    answer: str,
    topic_id: UUID | str | None = None,
    table_name: str | None = None,
) -> dict[str, Any]:
    """Complete the exact external chat turn created for this request."""

    table_name = table_name or settings.postgres_external_chat_table
    table = sql.Identifier(table_name)
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    UPDATE {table}
                    SET answer = %s,
                        topic_id = COALESCE(%s::uuid, topic_id)
                    WHERE message_id = %s
                      AND service_id = %s
                      AND answer = ''
                    RETURNING
                        message_id,
                        service_id,
                        user_id,
                        session_id,
                        conversation_id,
                        topic_id,
                        create_time,
                        question,
                        answer
                    """
                ).format(table=table),
                (answer, topic_id, message_id, service_id),
            )
            row = cur.fetchone()
    return dict(row or {})


def create_chat_message(
    *,
    user_id: str,
    session_id: str,
    conversation_id: str,
    question: str,
    user_name: str | None = None,
    source_message_id: str | None = None,
    table_name: str | None = None,
) -> dict[str, Any]:
    """Create the initial chat row before the QA workflow returns an answer."""

    table_name = table_name or settings.postgres_chat_table
    table = sql.Identifier(table_name)

    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {table} (
                        user_id,
                        user_name,
                        source_message_id,
                        session_id,
                        conversation_id,
                        question
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (source_message_id) WHERE source_message_id IS NOT NULL
                    DO UPDATE SET user_name = COALESCE(EXCLUDED.user_name, {table}.user_name)
                    RETURNING
                        message_id,
                        source_message_id,
                        user_id,
                        user_name,
                        session_id,
                        conversation_id,
                        topic_id,
                        create_time,
                        question,
                        answer
                    """
                ).format(table=table),
                (
                    user_id,
                    user_name,
                    source_message_id,
                    session_id,
                    conversation_id,
                    question,
                ),
            )
            row = cur.fetchone()

    return dict(row or {})


def update_chat_answer(
    *,
    message_id: UUID | str | None = None,
    user_id: str,
    session_id: str,
    conversation_id: str,
    answer: str,
    user_name: str | None = None,
    topic_id: UUID | str | None = None,
    table_name: str | None = None,
) -> dict[str, Any]:
    """Attach an answer to one stable chat row.

    ``message_id`` is required by all new call sites. The legacy selector is
    retained only so old rollback scripts can still complete pre-migration
    rows; it must not be used for concurrent request processing.
    """

    table_name = table_name or settings.postgres_chat_table
    table = sql.Identifier(table_name)

    with postgres_connection() as conn:
        with conn.cursor() as cur:
            if message_id is not None:
                cur.execute(
                    sql.SQL(
                        """
                    UPDATE {table}
                    SET answer = CASE WHEN answer = '' THEN %s ELSE answer END,
                        user_name = COALESCE(%s, user_name),
                        topic_id = COALESCE(%s::uuid, topic_id)
                    WHERE message_id = %s
                    RETURNING
                        message_id,
                        user_id,
                        user_name,
                        session_id,
                        conversation_id,
                        topic_id,
                        create_time,
                        question,
                        answer
                        """
                    ).format(table=table),
                    (answer, user_name, topic_id, message_id),
                )
            else:
                cur.execute(
                    sql.SQL(
                        """
                    UPDATE {table}
                    SET answer = %s,
                        user_name = COALESCE(%s, user_name),
                        topic_id = COALESCE(%s::uuid, topic_id)
                    WHERE ctid = (
                        SELECT ctid
                        FROM {table}
                        WHERE user_id = %s
                          AND session_id = %s
                          AND conversation_id = %s
                          AND answer = ''
                        ORDER BY create_time DESC, ctid DESC
                        LIMIT 1
                    )
                    RETURNING
                        message_id,
                        user_id,
                        user_name,
                        session_id,
                        conversation_id,
                        topic_id,
                        create_time,
                        question,
                        answer
                        """
                    ).format(table=table),
                    (answer, user_name, topic_id, user_id, session_id, conversation_id),
                )
            row = cur.fetchone()

    return dict(row or {})


def assign_latest_chat_message_topic(
    *,
    user_id: str,
    session_id: str,
    conversation_id: str,
    topic_id: UUID | str,
    table_name: str | None = None,
) -> dict[str, Any]:
    """Attach the latest matching chat row to a virtual topic."""

    table_name = table_name or settings.postgres_chat_table
    table = sql.Identifier(table_name)

    user_name_projection = (
        sql.SQL("NULL::TEXT AS user_name")
        if table_name == settings.postgres_external_chat_table
        else sql.Identifier("user_name")
    )
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    UPDATE {table}
                    SET topic_id = %s
                    WHERE ctid = (
                        SELECT ctid
                        FROM {table}
                        WHERE user_id = %s
                          AND session_id = %s
                          AND conversation_id = %s
                        ORDER BY create_time DESC, ctid DESC
                        LIMIT 1
                    )
                    RETURNING
                        user_id,
                        {user_name_projection},
                        session_id,
                        conversation_id,
                        topic_id,
                        create_time,
                        question,
                        answer
                    """
                ).format(table=table, user_name_projection=user_name_projection),
                (topic_id, user_id, session_id, conversation_id),
            )
            row = cur.fetchone()

    return dict(row or {})


def list_chat_messages_by_topic(
    *,
    topic_id: UUID | str,
    user_id: str,
    session_id: str,
    limit: int = 10,
    table_name: str | None = None,
) -> list[dict[str, Any]]:
    """Return recent chat turns already attached to one virtual topic."""

    table_name = table_name or settings.postgres_chat_table
    table = sql.Identifier(table_name)
    limit = max(1, min(int(limit), 100))

    user_name_projection = (
        sql.SQL("NULL::TEXT AS user_name")
        if table_name == settings.postgres_external_chat_table
        else sql.Identifier("user_name")
    )
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT
                        user_id,
                        {user_name_projection},
                        session_id,
                        conversation_id,
                        topic_id,
                        create_time,
                        question,
                        answer
                    FROM {table}
                    WHERE topic_id = %s
                      AND user_id = %s
                      AND session_id = %s
                    ORDER BY create_time DESC, ctid DESC
                    LIMIT %s
                    """
                ).format(table=table, user_name_projection=user_name_projection),
                (topic_id, user_id, session_id, limit),
            )
            rows = cur.fetchall()

    return [dict(row) for row in reversed(rows)]


def ensure_feishu_answer_feedback_table() -> None:
    table = sql.Identifier(FEISHU_ANSWER_FEEDBACK_TABLE)
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {table} (
                        feedback_id VARCHAR(64) PRIMARY KEY,
                        answer TEXT NOT NULL,
                        selected VARCHAR(16),
                        create_time TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                ).format(table=table)
            )


def insert_feishu_answer_feedback_state(
    *,
    feedback_id: str,
    answer: str,
) -> dict[str, Any]:
    ensure_feishu_answer_feedback_table()
    table = sql.Identifier(FEISHU_ANSWER_FEEDBACK_TABLE)
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {table} (feedback_id, answer)
                    VALUES (%s, %s)
                    ON CONFLICT (feedback_id) DO UPDATE
                    SET answer = EXCLUDED.answer,
                        selected = NULL,
                        create_time = CURRENT_TIMESTAMP
                    RETURNING feedback_id, answer, selected, create_time
                    """
                ).format(table=table),
                (feedback_id, answer),
            )
            row = cur.fetchone()
    return dict(row or {})


def get_feishu_answer_feedback_state(feedback_id: str) -> dict[str, Any]:
    ensure_feishu_answer_feedback_table()
    table = sql.Identifier(FEISHU_ANSWER_FEEDBACK_TABLE)
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT feedback_id, answer, selected, create_time
                    FROM {table}
                    WHERE feedback_id = %s
                    """
                ).format(table=table),
                (feedback_id,),
            )
            row = cur.fetchone()
    return dict(row or {})


def update_feishu_answer_feedback_selection(
    *,
    feedback_id: str,
    selected: str,
) -> dict[str, Any]:
    ensure_feishu_answer_feedback_table()
    table = sql.Identifier(FEISHU_ANSWER_FEEDBACK_TABLE)
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    UPDATE {table}
                    SET selected = COALESCE(selected, %s)
                    WHERE feedback_id = %s
                    RETURNING feedback_id, answer, selected, create_time
                    """
                ).format(table=table),
                (selected, feedback_id),
            )
            row = cur.fetchone()
    return dict(row or {})


def delete_expired_feishu_answer_feedback_states(*, older_than_seconds: float) -> int:
    ensure_feishu_answer_feedback_table()
    table = sql.Identifier(FEISHU_ANSWER_FEEDBACK_TABLE)
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    DELETE FROM {table}
                    WHERE create_time < CURRENT_TIMESTAMP - (%s * INTERVAL '1 second')
                    """
                ).format(table=table),
                (older_than_seconds,),
            )
            return cur.rowcount or 0


def ensure_answer_job_tables() -> None:
    """Create the PostgreSQL-backed Feishu queue and shared rate buckets."""

    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {FEISHU_ANSWER_JOBS_TABLE} (
                    job_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    dedupe_key TEXT NOT NULL UNIQUE,
                    user_id TEXT NOT NULL,
                    source_session_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    available_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    leased_at TIMESTAMPTZ,
                    lease_owner TEXT,
                    cancel_requested_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    last_error TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS feishu_answer_jobs_claim_idx ON {FEISHU_ANSWER_JOBS_TABLE} (status, available_at, created_at)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS feishu_answer_jobs_session_idx ON {FEISHU_ANSWER_JOBS_TABLE} (user_id, source_session_id, status)"
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {API_RATE_LIMITS_TABLE} (
                    scope TEXT NOT NULL,
                    subject_key TEXT NOT NULL,
                    window_start TIMESTAMPTZ NOT NULL,
                    request_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (scope, subject_key, window_start)
                )
                """
            )


def enqueue_feishu_answer_job(
    *,
    dedupe_key: str,
    user_id: str,
    source_session_id: str,
    payload: str,
    max_attempts: int,
) -> tuple[dict[str, Any], bool]:
    """Atomically enqueue a message; return ``(row, created)``."""

    ensure_answer_job_tables()
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {FEISHU_ANSWER_JOBS_TABLE}
                    (dedupe_key, user_id, source_session_id, payload, max_attempts)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (dedupe_key) DO NOTHING
                RETURNING *
                """,
                (dedupe_key, user_id, source_session_id, payload, max(1, max_attempts)),
            )
            row = cur.fetchone()
            if row:
                return dict(row), True
            cur.execute(
                f"SELECT * FROM {FEISHU_ANSWER_JOBS_TABLE} WHERE dedupe_key = %s",
                (dedupe_key,),
            )
            return dict(cur.fetchone() or {}), False


def claim_feishu_answer_job(*, lease_owner: str, lease_seconds: int) -> dict[str, Any]:
    """Lease the oldest runnable job with crash-safe stale-lease recovery."""

    ensure_answer_job_tables()
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {FEISHU_ANSWER_JOBS_TABLE}
                SET status = CASE
                        WHEN cancel_requested_at IS NOT NULL THEN 'cancelled'
                        WHEN attempts >= max_attempts THEN 'failed'
                        ELSE 'queued'
                    END,
                    completed_at = CASE
                        WHEN cancel_requested_at IS NOT NULL OR attempts >= max_attempts
                        THEN CURRENT_TIMESTAMP ELSE completed_at
                    END,
                    lease_owner = NULL, leased_at = NULL,
                    available_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP,
                    last_error = COALESCE(last_error, 'worker lease expired')
                WHERE status = 'running'
                  AND leased_at < CURRENT_TIMESTAMP - (%s * INTERVAL '1 second')
                """,
                (max(30, lease_seconds),),
            )
            cur.execute(
                f"""
                WITH candidate AS (
                    SELECT queued.job_id
                    FROM {FEISHU_ANSWER_JOBS_TABLE} AS queued
                    WHERE queued.status = 'queued'
                      AND queued.attempts < queued.max_attempts
                      AND queued.available_at <= CURRENT_TIMESTAMP
                      AND queued.cancel_requested_at IS NULL
                      AND NOT EXISTS (
                        SELECT 1 FROM {FEISHU_ANSWER_JOBS_TABLE} AS running
                        WHERE running.status = 'running'
                          AND running.user_id = queued.user_id
                          AND running.source_session_id = queued.source_session_id
                      )
                    ORDER BY queued.created_at
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                UPDATE {FEISHU_ANSWER_JOBS_TABLE} AS jobs
                SET status = 'running', attempts = attempts + 1,
                    leased_at = CURRENT_TIMESTAMP, lease_owner = %s,
                    updated_at = CURRENT_TIMESTAMP
                FROM candidate
                WHERE jobs.job_id = candidate.job_id
                RETURNING jobs.*
                """,
                (lease_owner,),
            )
            return dict(cur.fetchone() or {})


def finish_feishu_answer_job(
    *, job_id: UUID | str, success: bool, error: str = ""
) -> dict[str, Any]:
    """Complete a job or requeue it with bounded exponential backoff."""

    ensure_answer_job_tables()
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            if success:
                cur.execute(
                    f"""
                    UPDATE {FEISHU_ANSWER_JOBS_TABLE}
                    SET status = CASE WHEN cancel_requested_at IS NULL THEN 'succeeded' ELSE 'cancelled' END,
                        completed_at = CURRENT_TIMESTAMP, lease_owner = NULL,
                        leased_at = NULL, last_error = NULL, updated_at = CURRENT_TIMESTAMP
                    WHERE job_id = %s RETURNING *
                    """,
                    (job_id,),
                )
            else:
                cur.execute(
                    f"""
                    UPDATE {FEISHU_ANSWER_JOBS_TABLE}
                    SET status = CASE
                            WHEN cancel_requested_at IS NOT NULL THEN 'cancelled'
                            WHEN attempts >= max_attempts THEN 'failed'
                            ELSE 'queued'
                        END,
                        available_at = CURRENT_TIMESTAMP
                            + (LEAST(60, POWER(2, GREATEST(attempts - 1, 0))) * INTERVAL '1 second'),
                        completed_at = CASE
                            WHEN cancel_requested_at IS NOT NULL OR attempts >= max_attempts
                            THEN CURRENT_TIMESTAMP ELSE NULL END,
                        lease_owner = NULL, leased_at = NULL,
                        last_error = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE job_id = %s RETURNING *
                    """,
                    (error[:1000], job_id),
                )
            return dict(cur.fetchone() or {})


def request_feishu_answer_job_cancel(job_id: UUID | str) -> bool:
    ensure_answer_job_tables()
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {FEISHU_ANSWER_JOBS_TABLE}
                SET cancel_requested_at = COALESCE(cancel_requested_at, CURRENT_TIMESTAMP),
                    status = CASE WHEN status = 'queued' THEN 'cancelled' ELSE status END,
                    completed_at = CASE WHEN status = 'queued' THEN CURRENT_TIMESTAMP ELSE completed_at END,
                    updated_at = CURRENT_TIMESTAMP
                WHERE job_id = %s AND status IN ('queued', 'running')
                RETURNING job_id
                """,
                (job_id,),
            )
            return cur.fetchone() is not None


def is_feishu_answer_job_cancelled(job_id: UUID | str) -> bool:
    ensure_answer_job_tables()
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT cancel_requested_at IS NOT NULL AS cancelled FROM {FEISHU_ANSWER_JOBS_TABLE} WHERE job_id = %s",
                (job_id,),
            )
            row = cur.fetchone() or {}
            return bool(row.get("cancelled"))


def answer_job_metrics() -> dict[str, Any]:
    ensure_answer_job_tables()
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT status, COUNT(*) AS count
                FROM {FEISHU_ANSWER_JOBS_TABLE}
                GROUP BY status
                """
            )
            counts = {str(row["status"]): int(row["count"]) for row in cur.fetchall()}
            cur.execute(
                f"SELECT EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - MIN(created_at))) AS oldest_seconds FROM {FEISHU_ANSWER_JOBS_TABLE} WHERE status = 'queued'"
            )
            oldest = cur.fetchone() or {}
    return {"counts": counts, "oldest_queued_seconds": float(oldest.get("oldest_seconds") or 0)}


def harness_session_metrics() -> dict[str, int]:
    ensure_harness_tables()
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT status, COUNT(*) AS count FROM {HARNESS_SESSIONS_TABLE} GROUP BY status"
            )
            return {str(row["status"]): int(row["count"]) for row in cur.fetchall()}


def consume_rate_limit(
    *, scope: str, subject_key: str, limit: int, burst: int = 0
) -> tuple[bool, int, int]:
    """Consume one shared fixed-minute bucket and return allowed/count/retry."""

    if limit <= 0:
        return True, 0, 0
    ensure_answer_job_tables()
    effective_limit = max(1, limit + max(0, burst))
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {API_RATE_LIMITS_TABLE}
                    (scope, subject_key, window_start, request_count)
                VALUES (%s, %s, date_trunc('minute', CURRENT_TIMESTAMP), 1)
                ON CONFLICT (scope, subject_key, window_start)
                DO UPDATE SET request_count = {API_RATE_LIMITS_TABLE}.request_count + 1
                RETURNING request_count,
                    GREATEST(1, CEIL(EXTRACT(EPOCH FROM
                        (date_trunc('minute', CURRENT_TIMESTAMP) + INTERVAL '1 minute' - CURRENT_TIMESTAMP))))::int
                        AS retry_after
                """,
                (scope, subject_key),
            )
            row = cur.fetchone() or {}
            count = int(row.get("request_count") or 0)
            retry_after = int(row.get("retry_after") or 1)
            if count == 1:
                cur.execute(
                    f"DELETE FROM {API_RATE_LIMITS_TABLE} WHERE window_start < CURRENT_TIMESTAMP - INTERVAL '10 minutes'"
                )
    return count <= effective_limit, count, retry_after


def ensure_harness_tables() -> None:
    """Create the small control-plane tables used by the Harness adapter.

    The Feishu chat id is deliberately stored as a source key, never used as
    the Harness session id: Feishu has one permanent chat window per user.
    """

    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {HARNESS_SESSIONS_TABLE} (
                    internal_session_id UUID PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    source_session_id TEXT NOT NULL,
                    harness_session_id TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_activity_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    archived_at TIMESTAMPTZ,
                    archive_error TEXT,
                    context_tokens BIGINT NOT NULL DEFAULT 0,
                    rollover_requested_at TIMESTAMPTZ,
                    archive_summary TEXT NOT NULL DEFAULT '',
                    handoff_summary TEXT NOT NULL DEFAULT '',
                    handoff_pending BOOLEAN NOT NULL DEFAULT FALSE,
                    handoff_consumed_at TIMESTAMPTZ,
                    handoff_source_session_id UUID
                )
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {HARNESS_MEMORIES_TABLE} (
                    memory_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id TEXT NOT NULL,
                    internal_session_id UUID NOT NULL REFERENCES {HARNESS_SESSIONS_TABLE}(internal_session_id),
                    topic TEXT NOT NULL,
                    keywords JSONB NOT NULL DEFAULT '[]'::jsonb,
                    summary TEXT NOT NULL DEFAULT '',
                    object_uri TEXT NOT NULL UNIQUE,
                    started_at TIMESTAMPTZ,
                    ended_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    status TEXT NOT NULL DEFAULT 'ready'
                )
                """
            )
            cur.execute(
                f"ALTER TABLE {HARNESS_MEMORIES_TABLE} ADD COLUMN IF NOT EXISTS summary TEXT NOT NULL DEFAULT ''"
            )
            for statement in (
                f"ALTER TABLE {HARNESS_SESSIONS_TABLE} ADD COLUMN IF NOT EXISTS context_tokens BIGINT NOT NULL DEFAULT 0",
                f"ALTER TABLE {HARNESS_SESSIONS_TABLE} ADD COLUMN IF NOT EXISTS rollover_requested_at TIMESTAMPTZ",
                f"ALTER TABLE {HARNESS_SESSIONS_TABLE} ADD COLUMN IF NOT EXISTS archive_summary TEXT NOT NULL DEFAULT ''",
                f"ALTER TABLE {HARNESS_SESSIONS_TABLE} ADD COLUMN IF NOT EXISTS handoff_summary TEXT NOT NULL DEFAULT ''",
                f"ALTER TABLE {HARNESS_SESSIONS_TABLE} ADD COLUMN IF NOT EXISTS handoff_pending BOOLEAN NOT NULL DEFAULT FALSE",
                f"ALTER TABLE {HARNESS_SESSIONS_TABLE} ADD COLUMN IF NOT EXISTS handoff_consumed_at TIMESTAMPTZ",
                f"ALTER TABLE {HARNESS_SESSIONS_TABLE} ADD COLUMN IF NOT EXISTS handoff_source_session_id UUID",
            ):
                cur.execute(statement)
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS harness_sessions_idle_idx ON {HARNESS_SESSIONS_TABLE} (status, last_activity_at)"
            )
            cur.execute(
                f"CREATE UNIQUE INDEX IF NOT EXISTS harness_sessions_one_active_idx ON {HARNESS_SESSIONS_TABLE} (user_id, source_session_id) WHERE status = 'active'"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS harness_memories_user_idx ON {HARNESS_MEMORIES_TABLE} (user_id, created_at DESC)"
            )


def get_or_create_harness_session(*, user_id: str, source_session_id: str) -> dict[str, Any]:
    ensure_harness_tables()
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (f"harness-session:{user_id}:{source_session_id}",),
            )
            cur.execute(
                f"""
                SELECT * FROM {HARNESS_SESSIONS_TABLE}
                WHERE user_id = %s AND source_session_id = %s AND status = 'active'
                FOR UPDATE
                """,
                (user_id, source_session_id),
            )
            row = cur.fetchone()
            if row:
                cur.execute(
                    f"UPDATE {HARNESS_SESSIONS_TABLE} SET last_activity_at = CURRENT_TIMESTAMP WHERE internal_session_id = %s RETURNING *",
                    (row["internal_session_id"],),
                )
                return dict(cur.fetchone() or row)
            internal_id = uuid4()
            harness_id = f"mtsco-{internal_id}"
            cur.execute(
                f"""
                SELECT internal_session_id, archive_summary, rollover_requested_at
                FROM {HARNESS_SESSIONS_TABLE}
                WHERE user_id = %s AND source_session_id = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (user_id, source_session_id),
            )
            previous = cur.fetchone() or {}
            is_rollover = previous.get("rollover_requested_at") is not None
            handoff_summary = (
                str(previous.get("archive_summary") or "")[:8000] if is_rollover else ""
            )
            handoff_source_session_id = (
                previous.get("internal_session_id") if is_rollover else None
            )
            cur.execute(
                f"""
                INSERT INTO {HARNESS_SESSIONS_TABLE}
                (internal_session_id, user_id, source_session_id, harness_session_id,
                 handoff_summary, handoff_pending, handoff_source_session_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING *
                """,
                (
                    internal_id,
                    user_id,
                    source_session_id,
                    harness_id,
                    handoff_summary,
                    is_rollover,
                    handoff_source_session_id,
                ),
            )
            return dict(cur.fetchone() or {})


def update_harness_context_pressure(
    *, internal_session_id: UUID | str, context_tokens: int, archive_threshold: int
) -> bool:
    """Persist provider pressure and request a between-turn rollover once."""

    ensure_harness_tables()
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {HARNESS_SESSIONS_TABLE}
                SET context_tokens = GREATEST(context_tokens, %s),
                    rollover_requested_at = CASE
                        WHEN %s >= %s THEN COALESCE(rollover_requested_at, CURRENT_TIMESTAMP)
                        ELSE rollover_requested_at END
                WHERE internal_session_id = %s AND status = 'active'
                RETURNING rollover_requested_at IS NOT NULL AS requested
                """,
                (max(0, context_tokens), max(0, context_tokens), max(1, archive_threshold), internal_session_id),
            )
            row = cur.fetchone() or {}
            return bool(row.get("requested"))


def get_harness_handoff_summary(*, internal_session_id: UUID | str) -> str:
    ensure_harness_tables()
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT COALESCE(NULLIF(current.handoff_summary, ''), source.archive_summary, '') AS handoff_summary
                FROM {HARNESS_SESSIONS_TABLE} AS current
                LEFT JOIN {HARNESS_SESSIONS_TABLE} AS source
                  ON source.internal_session_id = current.handoff_source_session_id
                 AND source.status = 'archived'
                WHERE current.internal_session_id = %s
                  AND current.handoff_pending = TRUE
                """,
                (internal_session_id,),
            )
            row = cur.fetchone() or {}
            return str(row.get("handoff_summary") or "")


def mark_harness_handoff_consumed(*, internal_session_id: UUID | str) -> None:
    ensure_harness_tables()
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {HARNESS_SESSIONS_TABLE}
                SET handoff_pending = FALSE, handoff_consumed_at = CURRENT_TIMESTAMP
                WHERE internal_session_id = %s AND handoff_pending = TRUE
                """,
                (internal_session_id,),
            )


def list_harness_chat_turns(*, internal_session_id: UUID | str, limit: int = 32) -> list[dict[str, Any]]:
    """Return completed Feishu turns for one generated Harness session.

    These application records are the durable transcript.  They are written
    independently of the Harness runtime and remain reliable when a JSON-RPC
    child process exits before its JSONL writer has flushed every event.
    """

    ensure_chat_messages_table()
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT question, answer, create_time
                FROM chat_messages
                WHERE session_id = %s AND COALESCE(answer, '') <> ''
                ORDER BY create_time DESC
                LIMIT %s
                """,
                (str(internal_session_id), max(1, min(limit, 100))),
            )
            return [dict(row) for row in reversed(cur.fetchall())]


def list_expired_harness_sessions(*, idle_seconds: int, limit: int = 100) -> list[dict[str, Any]]:
    ensure_harness_tables()
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT * FROM {HARNESS_SESSIONS_TABLE}
                WHERE status = 'active'
                  AND (
                    (
                      rollover_requested_at < CURRENT_TIMESTAMP - INTERVAL '30 seconds'
                      AND last_activity_at < CURRENT_TIMESTAMP - INTERVAL '30 seconds'
                    )
                    OR last_activity_at < CURRENT_TIMESTAMP - (%s * INTERVAL '1 second')
                  )
                ORDER BY last_activity_at
                LIMIT %s FOR UPDATE SKIP LOCKED
                """,
                (idle_seconds, limit),
            )
            rows = [dict(row) for row in cur.fetchall()]
            for row in rows:
                cur.execute(
                    f"UPDATE {HARNESS_SESSIONS_TABLE} SET status = 'archiving' WHERE internal_session_id = %s",
                    (row["internal_session_id"],),
                )
            return rows


def list_live_harness_session_ids() -> set[str]:
    """Return sessions whose temporary attachments must not be TTL-cleaned."""
    ensure_harness_tables()
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT internal_session_id FROM {HARNESS_SESSIONS_TABLE} WHERE status IN ('active', 'archiving')"
            )
            return {str(row["internal_session_id"]) for row in cur.fetchall()}


def complete_harness_archive(
    *, internal_session_id: UUID | str, error: str = "", summary: str = ""
) -> None:
    ensure_harness_tables()
    status = "active" if error else "archived"
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""UPDATE {HARNESS_SESSIONS_TABLE}
                SET status = %s, archived_at = CASE WHEN %s = '' THEN CURRENT_TIMESTAMP ELSE NULL END,
                    archive_error = NULLIF(%s, ''),
                    archive_summary = CASE WHEN %s = '' THEN archive_summary ELSE %s END
                WHERE internal_session_id = %s""",
                (status, error, error, summary, summary[:8000], internal_session_id),
            )


def insert_harness_memory(
    *,
    user_id: str,
    internal_session_id: UUID | str,
    topic: str,
    keywords: list[str],
    object_uri: str,
    summary: str,
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
) -> None:
    ensure_harness_tables()
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""INSERT INTO {HARNESS_MEMORIES_TABLE}
                (user_id, internal_session_id, topic, keywords, summary, object_uri, started_at, ended_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (object_uri) DO NOTHING""",
                (user_id, internal_session_id, topic, Jsonb(keywords), summary, object_uri, started_at, ended_at),
            )


def list_harness_memories(
    *,
    user_id: str,
    query: str = "",
    limit: int = 8,
    start_at: datetime | None = None,
    end_at: datetime | None = None,
) -> list[dict[str, Any]]:
    """Return only the requesting user's memory catalogue (never cross-user)."""
    ensure_harness_tables()
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""SELECT memory_id, internal_session_id, topic, keywords, summary, object_uri, started_at, ended_at, created_at
                FROM {HARNESS_MEMORIES_TABLE}
                WHERE user_id = %s AND status = 'ready'
                  AND (%s = '' OR topic ILIKE '%%' || %s || '%%')
                  AND (%s::timestamptz IS NULL OR COALESCE(ended_at, created_at) >= %s::timestamptz)
                  AND (%s::timestamptz IS NULL OR COALESCE(started_at, created_at) < %s::timestamptz)
                ORDER BY created_at DESC LIMIT %s""",
                (
                    user_id,
                    query.strip(),
                    query.strip(),
                    start_at, start_at,
                    end_at, end_at,
                    max(1, min(limit, 20)),
                ),
            )
            return [dict(row) for row in cur.fetchall()]


def check_postgres_health() -> dict[str, Any]:
    """Return basic PostgreSQL connectivity details without mutating data."""

    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    current_database() AS database,
                    current_user AS user,
                    current_schema() AS schema,
                    version() AS version
                """
            )
            result = cur.fetchone()

    return dict(result or {})

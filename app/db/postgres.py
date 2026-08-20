from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
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


def ensure_chat_messages_table(table_name: str | None = None) -> dict[str, Any]:
    """Create the chat message table if it does not already exist."""

    table_name = table_name or settings.postgres_chat_table
    table = sql.Identifier(table_name)

    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {table} (
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
                        session_id,
                        conversation_id,
                        question
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING
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
                (user_id, user_name, session_id, conversation_id, question),
            )
            row = cur.fetchone()

    return dict(row or {})


def update_chat_answer(
    *,
    user_id: str,
    session_id: str,
    conversation_id: str,
    answer: str,
    user_name: str | None = None,
    topic_id: UUID | str | None = None,
    table_name: str | None = None,
) -> dict[str, Any]:
    """Attach an answer to the latest matching chat row."""

    table_name = table_name or settings.postgres_chat_table
    table = sql.Identifier(table_name)

    with postgres_connection() as conn:
        with conn.cursor() as cur:
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
                    archive_error TEXT
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
                    object_uri TEXT NOT NULL UNIQUE,
                    started_at TIMESTAMPTZ,
                    ended_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    status TEXT NOT NULL DEFAULT 'ready'
                )
                """
            )
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
                INSERT INTO {HARNESS_SESSIONS_TABLE}
                (internal_session_id, user_id, source_session_id, harness_session_id)
                VALUES (%s, %s, %s, %s) RETURNING *
                """,
                (internal_id, user_id, source_session_id, harness_id),
            )
            return dict(cur.fetchone() or {})


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
                  AND last_activity_at < CURRENT_TIMESTAMP - (%s * INTERVAL '1 second')
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


def complete_harness_archive(*, internal_session_id: UUID | str, error: str = "") -> None:
    ensure_harness_tables()
    status = "active" if error else "archived"
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""UPDATE {HARNESS_SESSIONS_TABLE}
                SET status = %s, archived_at = CASE WHEN %s = '' THEN CURRENT_TIMESTAMP ELSE NULL END,
                    archive_error = NULLIF(%s, '')
                WHERE internal_session_id = %s""",
                (status, error, error, internal_session_id),
            )


def insert_harness_memory(*, user_id: str, internal_session_id: UUID | str, topic: str, keywords: list[str], object_uri: str) -> None:
    ensure_harness_tables()
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""INSERT INTO {HARNESS_MEMORIES_TABLE}
                (user_id, internal_session_id, topic, keywords, object_uri)
                VALUES (%s, %s, %s, %s, %s) ON CONFLICT (object_uri) DO NOTHING""",
                (user_id, internal_session_id, topic, Jsonb(keywords), object_uri),
            )


def list_harness_memories(*, user_id: str, query: str = "", limit: int = 8) -> list[dict[str, Any]]:
    """Return only the requesting user's memory catalogue (never cross-user)."""
    ensure_harness_tables()
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""SELECT memory_id, topic, keywords, object_uri, created_at
                FROM {HARNESS_MEMORIES_TABLE}
                WHERE user_id = %s AND status = 'ready'
                  AND (%s = '' OR topic ILIKE '%%' || %s || '%%')
                ORDER BY created_at DESC LIMIT %s""",
                (user_id, query.strip(), query.strip(), max(1, min(limit, 20))),
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

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import psycopg
from psycopg import Connection, sql
from psycopg.rows import dict_row

from app.core.config import settings


CHAT_MESSAGES_TABLE_COMMENT = "MTSCO knowledge base encrypted chat messages"
FEISHU_ANSWER_FEEDBACK_TABLE = "feishu_answer_feedback_states"


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
                        session_id VARCHAR(128) NOT NULL,
                        conversation_id VARCHAR(128) NOT NULL,
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


def insert_chat_message(
    *,
    user_id: str,
    session_id: str,
    conversation_id: str,
    question: str,
    answer: str,
    user_name: str | None = None,
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
                        question,
                        answer
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING
                        user_id,
                        user_name,
                        session_id,
                        conversation_id,
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
                    question,
                    answer,
                ),
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
                        user_name = COALESCE(%s, user_name)
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
                        create_time,
                        question,
                        answer
                    """
                ).format(table=table),
                (answer, user_name, user_id, session_id, conversation_id),
            )
            row = cur.fetchone()

    return dict(row or {})


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

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import psycopg
from psycopg import Connection, sql
from psycopg.rows import dict_row

from app.core.config import settings


CHAT_MESSAGES_TABLE_COMMENT = "MTSCO knowledge base chat messages and fallback records"


def get_postgres_connection() -> Connection[Any]:
    """Create a PostgreSQL connection from environment-backed settings."""

    if settings.database_url:
        return psycopg.connect(settings.database_url, row_factory=dict_row)

    return psycopg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
        row_factory=dict_row,
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
                        session_id VARCHAR(128) NOT NULL,
                        conversation_id VARCHAR(128) NOT NULL,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL DEFAULT '',
                        fallback BOOLEAN NOT NULL DEFAULT FALSE,
                        reason TEXT NOT NULL DEFAULT ''
                    )
                    """
                ).format(table=table)
            )
            cur.execute(
                sql.SQL("ALTER TABLE {table} ALTER COLUMN answer SET DEFAULT ''").format(
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
    fallback: bool,
    reason: str = "",
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
                        session_id,
                        conversation_id,
                        question,
                        answer,
                        fallback,
                        reason
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING
                        user_id,
                        session_id,
                        conversation_id,
                        question,
                        answer,
                        fallback,
                        reason
                    """
                ).format(table=table),
                (
                    user_id,
                    session_id,
                    conversation_id,
                    question,
                    answer,
                    fallback,
                    reason,
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
                        session_id,
                        conversation_id,
                        question
                    )
                    VALUES (%s, %s, %s, %s)
                    RETURNING
                        user_id,
                        session_id,
                        conversation_id,
                        question,
                        answer,
                        fallback,
                        reason
                    """
                ).format(table=table),
                (user_id, session_id, conversation_id, question),
            )
            row = cur.fetchone()

    return dict(row or {})


def update_chat_answer(
    *,
    user_id: str,
    session_id: str,
    conversation_id: str,
    question: str,
    answer: str,
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
                    SET answer = %s
                    WHERE ctid = (
                        SELECT ctid
                        FROM {table}
                        WHERE user_id = %s
                          AND session_id = %s
                          AND conversation_id = %s
                          AND question = %s
                        ORDER BY ctid DESC
                        LIMIT 1
                    )
                    RETURNING
                        user_id,
                        session_id,
                        conversation_id,
                        question,
                        answer,
                        fallback,
                        reason
                    """
                ).format(table=table),
                (answer, user_id, session_id, conversation_id, question),
            )
            row = cur.fetchone()

    return dict(row or {})


def update_chat_fallback(
    *,
    user_id: str,
    session_id: str,
    conversation_id: str,
    question: str,
    fallback: bool,
    reason: str = "",
    table_name: str | None = None,
) -> dict[str, Any]:
    """Attach fallback evaluation to the latest matching chat row."""

    table_name = table_name or settings.postgres_chat_table
    table = sql.Identifier(table_name)

    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    UPDATE {table}
                    SET fallback = %s,
                        reason = %s
                    WHERE ctid = (
                        SELECT ctid
                        FROM {table}
                        WHERE user_id = %s
                          AND session_id = %s
                          AND conversation_id = %s
                          AND question = %s
                        ORDER BY ctid DESC
                        LIMIT 1
                    )
                    RETURNING
                        user_id,
                        session_id,
                        conversation_id,
                        question,
                        answer,
                        fallback,
                        reason
                    """
                ).format(table=table),
                (fallback, reason, user_id, session_id, conversation_id, question),
            )
            row = cur.fetchone()

    return dict(row or {})


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

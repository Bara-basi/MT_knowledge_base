"""Standalone watchdog for idle Harness sessions.

Run this module in its own process/container.  It is deliberately polling
PostgreSQL rather than relying on an in-memory FastAPI task, so the 7-hour
cutoff is honoured even when no new Feishu message arrives.
"""
from __future__ import annotations

import json
import time
from datetime import datetime

from app.core.config import settings
from app.db.minio import build_minio_uri, ensure_bucket, get_minio_client
from app.db.postgres import (
    complete_harness_archive,
    insert_harness_memory,
    list_expired_harness_sessions,
    postgres_connection,
)
from app.services.privacy import decrypt_chat_text


def _turns(user_id: str, session_id: str) -> list[dict[str, str]]:
    with postgres_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT question, answer FROM chat_messages
                WHERE user_id = %s AND session_id = %s AND answer <> ''
                ORDER BY create_time""",
                (user_id, session_id),
            )
            return [
                {"question": decrypt_chat_text(str(row["question"])), "answer": decrypt_chat_text(str(row["answer"]))}
                for row in cur.fetchall()
            ]


def _memory_payload(turns: list[dict[str, str]]) -> tuple[str, list[str], str]:
    # A deterministic fallback keeps archiving reliable if the model service is
    # temporarily unavailable.  Later retrieval can still use this complete,
    # user-private transcript summary.
    keywords = []
    for turn in turns:
        keywords.extend(turn["question"].replace("，", " ").split()[:5])
    return "历史对话", list(dict.fromkeys(keywords))[:20], json.dumps({"turns": turns}, ensure_ascii=False, indent=2)


def archive_once() -> int:
    count = 0
    for session in list_expired_harness_sessions(idle_seconds=settings.harness_idle_seconds):
        try:
            turns = _turns(session["user_id"], str(session["internal_session_id"]))
            topic, keywords, body = _memory_payload(turns)
            day = datetime.now().strftime("%Y-%m-%d")
            object_name = f"{session['user_id']}/{session['internal_session_id']}/{day} {topic}.md"
            bucket = ensure_bucket(settings.harness_memory_bucket)
            markdown = f"# {topic}\n\n{body}\n".encode("utf-8")
            get_minio_client().put_object(bucket, object_name, data=__import__("io").BytesIO(markdown), length=len(markdown), content_type="text/markdown; charset=utf-8")
            uri = build_minio_uri(bucket, object_name)
            insert_harness_memory(user_id=session["user_id"], internal_session_id=session["internal_session_id"], topic=topic, keywords=keywords, object_uri=uri)
            complete_harness_archive(internal_session_id=session["internal_session_id"])
            count += 1
        except Exception as exc:  # keep it eligible for a later watchdog retry
            complete_harness_archive(internal_session_id=session["internal_session_id"], error=str(exc)[:1000])
    return count


def main() -> None:
    while True:
        archive_once()
        time.sleep(max(30, settings.harness_scheduler_interval_seconds))


if __name__ == "__main__":
    main()

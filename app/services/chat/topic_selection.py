from __future__ import annotations

import threading
import time
from typing import Any
from uuid import UUID


_topic_selection_ttl_seconds = 1800.0
_topic_selection_lock = threading.Lock()
_topic_selections: dict[tuple[str, str], dict[str, Any]] = {}


def remember_topic_selection(
    *,
    user_id: str,
    session_id: str,
    topic_id: UUID | str,
) -> None:
    """Remember the topic n8n selected while handling the current Feishu turn."""

    key = (_normalize_key_part(user_id), _normalize_key_part(session_id))
    if not key[0] or not key[1]:
        return
    with _topic_selection_lock:
        _topic_selections[key] = {
            "topic_id": str(topic_id),
            "expires_at": time.monotonic() + _topic_selection_ttl_seconds,
        }


def consume_topic_selection(
    *,
    user_id: str | None,
    session_id: str | None,
) -> str | None:
    """Return and clear the most recent selected topic for one user/session."""

    key = (_normalize_key_part(user_id), _normalize_key_part(session_id))
    if not key[0] or not key[1]:
        return None
    now = time.monotonic()
    with _topic_selection_lock:
        _drop_expired_locked(now)
        selection = _topic_selections.pop(key, None)
    if not selection:
        return None
    return str(selection["topic_id"])


def _drop_expired_locked(now: float) -> None:
    expired_keys = [
        key
        for key, selection in _topic_selections.items()
        if float(selection.get("expires_at") or 0) <= now
    ]
    for key in expired_keys:
        _topic_selections.pop(key, None)


def _normalize_key_part(value: str | None) -> str:
    return str(value).strip() if value is not None else ""

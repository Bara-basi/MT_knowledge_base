from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet

from app.core.config import settings


ENCRYPTION_PREFIX = "fernet:v1:"


def encrypt_chat_text(value: str) -> str:
    text = value or ""
    return f"{ENCRYPTION_PREFIX}{_chat_cipher().encrypt(text.encode('utf-8')).decode('ascii')}"


def decrypt_chat_text(value: str) -> str:
    if not value.startswith(ENCRYPTION_PREFIX):
        return value
    token = value.removeprefix(ENCRYPTION_PREFIX).encode("ascii")
    return _chat_cipher().decrypt(token).decode("utf-8")


def _chat_cipher() -> Fernet:
    return Fernet(_normalize_fernet_key(settings.chat_message_encryption_key))


def _normalize_fernet_key(raw_key: str) -> bytes:
    key = raw_key.strip()
    if not key:
        raise ValueError("CHAT_MESSAGE_ENCRYPTION_KEY must not be empty.")
    try:
        Fernet(key.encode("ascii"))
        return key.encode("ascii")
    except Exception:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return base64.urlsafe_b64encode(digest)

from __future__ import annotations

import asyncio

from app.services import chat_records
from app.services.privacy import decrypt_chat_text, encrypt_chat_text


def test_chat_text_encryption_is_reversible_and_not_plaintext() -> None:
    plaintext = "客户问题：报价里包含手机号 13800000000"

    ciphertext = encrypt_chat_text(plaintext)

    assert ciphertext != plaintext
    assert "13800000000" not in ciphertext
    assert decrypt_chat_text(ciphertext) == plaintext


def test_decrypt_chat_text_keeps_legacy_plaintext() -> None:
    assert decrypt_chat_text("legacy plaintext") == "legacy plaintext"


def test_chat_record_encrypts_question_and_answer(monkeypatch) -> None:
    created_rows: list[dict] = []
    answered_rows: list[dict] = []

    def fake_ensure_table() -> None:
        return None

    def fake_create_message(**kwargs):
        created_rows.append(kwargs)
        return kwargs

    def fake_update_answer(**kwargs):
        answered_rows.append(kwargs)
        return kwargs

    monkeypatch.setattr(chat_records, "ensure_chat_messages_table", fake_ensure_table)
    monkeypatch.setattr(chat_records, "create_chat_message", fake_create_message)
    monkeypatch.setattr(chat_records, "update_chat_answer", fake_update_answer)

    asyncio.run(
        chat_records.create_chat_record(
            user_id="user-1",
            user_name="张三",
            session_id="chat-1",
            conversation_id="chat-1",
            question="我的手机号是 13800000000",
        )
    )
    asyncio.run(
        chat_records.record_chat_answer(
            user_id="user-1",
            user_name="张三",
            session_id="chat-1",
            conversation_id="chat-1",
            question="我的手机号是 13800000000",
            answer="已收到",
        )
    )

    assert created_rows[0]["user_name"] == "张三"
    assert "13800000000" not in created_rows[0]["question"]
    assert decrypt_chat_text(created_rows[0]["question"]) == "我的手机号是 13800000000"
    assert answered_rows[0]["user_name"] == "张三"
    assert "question" not in answered_rows[0]
    assert answered_rows[0]["answer"] != "已收到"
    assert decrypt_chat_text(answered_rows[0]["answer"]) == "已收到"

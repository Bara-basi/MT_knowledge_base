from __future__ import annotations

from app.api.v1.documents import resolve_raw_document_object
from app.services.agent_documents import (
    prepare_agent_text_payload,
    resolve_agent_document_reference,
    resolve_standard_text_reference,
)
from app.db.minio import RawDocumentObject


def test_resolve_raw_document_object_converts_legacy_data_raw_path() -> None:
    resolved = resolve_raw_document_object(
        r"data\raw\category\demo.docx",
        bucket="test-bucket",
    )

    assert resolved.bucket == "test-bucket"
    assert resolved.object_name == "category/demo.docx"


def test_resolve_raw_document_object_accepts_minio_uri() -> None:
    resolved = resolve_raw_document_object(
        "minio://knowledge-raw-docs/category/demo%20file.docx",
    )

    assert resolved.bucket == "knowledge-raw-docs"
    assert resolved.object_name == "category/demo file.docx"


def test_agent_document_reference_accepts_knowledge_buckets() -> None:
    raw = resolve_agent_document_reference("minio://knowledge-raw-docs/category/demo.pdf")
    asset = resolve_agent_document_reference("minio://knowledge-standard-assets/tables/demo.png")

    assert raw.bucket == "knowledge-raw-docs"
    assert asset.bucket == "knowledge-standard-assets"


def test_agent_document_reference_rejects_local_paths() -> None:
    try:
        resolve_agent_document_reference(r"data\raw\demo.pdf")
    except ValueError as exc:
        assert "explicit minio://" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("local path should not be available to the QA agent")


def test_agent_document_reference_rejects_unknown_bucket() -> None:
    try:
        resolve_agent_document_reference("minio://private-bucket/demo.pdf")
    except ValueError as exc:
        assert "not available" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("unknown bucket should not be available to the QA agent")


def test_resolve_standard_text_reference_uses_same_basename(monkeypatch) -> None:
    expected = (
        "产品标准/ASME-Sec-II-A-Vol1-2023(切分版)/"
        "SA-213 demo/SA-213 demo.txt"
    )

    class Client:
        def stat_object(self, bucket, object_name):
            assert bucket == "knowledge-standard-assets"
            assert object_name == expected
            return object()

    monkeypatch.setattr("app.services.agent_documents.get_minio_client", lambda: Client())
    resolved = resolve_standard_text_reference(
        RawDocumentObject(
            "knowledge-raw-docs",
            "产品标准/ASME-Sec-II-A-Vol1-2023(切分版)/SA-213 demo.pdf",
        )
    )
    assert resolved == RawDocumentObject("knowledge-standard-assets", expected)


def test_prepare_agent_text_payload_reads_utf8_text(monkeypatch) -> None:
    class Response:
        def read(self, _size):
            return "7.1 chemical composition".encode()

        def close(self):
            pass

        def release_conn(self):
            pass

    class Stat:
        size = 24
        content_type = "text/plain; charset=utf-8"

    class Client:
        def stat_object(self, _bucket, _object_name):
            return Stat()

        def get_object(self, _bucket, _object_name):
            return Response()

    monkeypatch.setattr("app.services.agent_documents.get_minio_client", lambda: Client())
    payload = prepare_agent_text_payload(
        "minio://knowledge-standard-assets/demo/demo.txt"
    )
    assert payload.content == "7.1 chemical composition"
    assert payload.asset_path == "minio://knowledge-standard-assets/demo/demo.txt"

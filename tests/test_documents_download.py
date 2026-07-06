from __future__ import annotations

from app.api.v1.documents import resolve_raw_document_object


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

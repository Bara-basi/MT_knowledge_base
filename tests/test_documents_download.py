from __future__ import annotations

from app.api.v1.documents import resolve_raw_document_path


def test_resolve_raw_document_path_uses_direct_data_raw_path(tmp_path) -> None:
    raw_root = tmp_path / "data" / "raw"
    document = raw_root / "category" / "demo.docx"
    document.parent.mkdir(parents=True)
    document.write_text("demo", encoding="utf-8")

    resolved = resolve_raw_document_path(
        "data\\raw\\category\\demo.docx",
        raw_root=raw_root,
    )

    assert resolved == document.resolve()


def test_resolve_raw_document_path_falls_back_to_filename_search(tmp_path) -> None:
    raw_root = tmp_path / "data" / "raw"
    document = raw_root / "new-folder" / "销售工具包 - 订单谈判.docx"
    document.parent.mkdir(parents=True)
    document.write_text("demo", encoding="utf-8")

    resolved = resolve_raw_document_path(
        "data/raw/old-folder/销售工具包%20-%20订单谈判.docx",
        raw_root=raw_root,
    )

    assert resolved == document.resolve()

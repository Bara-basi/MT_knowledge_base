from pathlib import Path

from app.services.processed_document_assets import processed_document_prefix, processed_document_uri
from scripts.storage.migrate_processed_assets_to_minio import (
    index_processing_directories,
    select_processing_dir,
)


def test_processed_archive_prefix_keeps_raw_extension() -> None:
    source = "minio://knowledge-raw-docs/team/manual.docx"

    assert processed_document_prefix(source) == "team/manual.docx"
    assert processed_document_uri(source, bucket="knowledge-processed-docs") == (
        "minio://knowledge-processed-docs/team/manual.docx"
    )


def test_legacy_migration_prefers_path_matching_raw_object(tmp_path: Path) -> None:
    root = tmp_path / "processing"
    expected = root / "team" / "manual"
    expected.mkdir(parents=True)
    (root / "another" / "manual").mkdir(parents=True)

    selected, reason = select_processing_dir(
        "team/manual.docx",
        processing_root=root,
        name_index=index_processing_directories(root),
    )

    assert selected == expected
    assert reason is None


def test_legacy_migration_reports_ambiguous_stem_fallback(tmp_path: Path) -> None:
    root = tmp_path / "processing"
    (root / "one" / "manual").mkdir(parents=True)
    (root / "two" / "manual").mkdir(parents=True)

    selected, reason = select_processing_dir(
        "other/manual.docx",
        processing_root=root,
        name_index=index_processing_directories(root),
    )

    assert selected is None
    assert reason and "ambiguous" in reason

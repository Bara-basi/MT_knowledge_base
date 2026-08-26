from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.harness import mcp_server
from app.services import harness_attachments
from app.services.parser import parser as document_parser
from app.services.parser import paths as parser_paths


def _configure_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "attachments"
    monkeypatch.setattr(harness_attachments, "settings", SimpleNamespace(
        harness_attachment_root=str(root),
        harness_attachment_max_bytes=1024 * 1024,
        harness_attachment_ttl_seconds=86400,
    ))
    return root


def test_attachment_ids_are_user_and_session_scoped(monkeypatch, tmp_path) -> None:
    _configure_root(monkeypatch, tmp_path)
    record = harness_attachments.save_attachment(
        user_id="user-a", internal_session_id="session-a",
        filename="report.docx", content=b"document",
    )
    assert harness_attachments.list_attachments(
        user_id="user-a", internal_session_id="session-a"
    )[0]["attachment_id"] == record["attachment_id"]
    assert harness_attachments.list_attachments(
        user_id="user-b", internal_session_id="session-a"
    ) == []
    with pytest.raises(FileNotFoundError):
        harness_attachments.read_attachment(
            user_id="user-b", internal_session_id="session-a",
            attachment_id=record["attachment_id"],
        )


def test_legacy_xls_is_accepted_and_upgraded_before_parsing(monkeypatch, tmp_path) -> None:
    _configure_root(monkeypatch, tmp_path)
    converted: list[Path] = []
    parsed_paths: list[Path] = []

    def fake_convert(_source: Path, target: Path) -> None:
        target.write_bytes(b"xlsx")
        converted.append(target)

    def fake_parse(path: Path, **_kwargs):
        parsed_paths.append(Path(path))
        return [{"type": "paragraph", "style": "正文", "text": "旧表格已解析"}]

    monkeypatch.setattr(harness_attachments, "_convert_xls_to_xlsx", fake_convert)
    monkeypatch.setattr(document_parser, "parse_document", fake_parse)
    record = harness_attachments.save_attachment(
        user_id="user-a",
        internal_session_id="session-a",
        filename="出库明细.xls",
        content=b"legacy-xls",
    )

    parsed = harness_attachments.parse_attachment(
        user_id="user-a",
        internal_session_id="session-a",
        attachment_id=record["attachment_id"],
    )

    assert converted[0].suffix == ".xlsx"
    assert parsed_paths[0] == converted[0]
    assert parsed["normalized_filename"] == "出库明细.xlsx"
    assert parsed["normalized_from"] == ".xls"
    assert "旧表格已解析" in parsed["preview"]


def test_python_xls_converter_preserves_values_sheets_and_merges(monkeypatch, tmp_path) -> None:
    import xlrd
    from openpyxl import load_workbook

    class FakeCell:
        def __init__(self, value, ctype):
            self.value = value
            self.ctype = ctype

    class FakeSheet:
        name = "旧/表"
        nrows = 2
        ncols = 2
        merged_cells = [(0, 1, 0, 2)]
        cells = [
            [FakeCell("标题", xlrd.XL_CELL_TEXT), FakeCell("", xlrd.XL_CELL_EMPTY)],
            [FakeCell(12.0, xlrd.XL_CELL_NUMBER), FakeCell(True, xlrd.XL_CELL_BOOLEAN)],
        ]

        def cell(self, row: int, column: int):
            return self.cells[row][column]

    class FakeBook:
        datemode = 0

        def sheets(self):
            return [FakeSheet()]

        def release_resources(self):
            return None

    monkeypatch.setattr(xlrd, "open_workbook", lambda *_args, **_kwargs: FakeBook())
    source = tmp_path / "legacy.xls"
    source.write_bytes(b"legacy")
    target = tmp_path / "modern.xlsx"

    harness_attachments._convert_xls_to_xlsx(source, target)

    workbook = load_workbook(target, data_only=False)
    try:
        sheet = workbook["旧_表"]
        assert sheet["A1"].value == "标题"
        assert sheet["A2"].value == 12
        assert sheet["B2"].value is True
        assert str(next(iter(sheet.merged_cells.ranges))) == "A1:B1"
    finally:
        workbook.close()


def test_relative_attachment_root_is_stable_across_process_workdirs(monkeypatch, tmp_path) -> None:
    project_root = tmp_path / "project"
    other_cwd = tmp_path / "harness-cwd"
    other_cwd.mkdir()
    monkeypatch.setattr(harness_attachments, "_PROJECT_ROOT", project_root)
    monkeypatch.setattr(harness_attachments, "settings", SimpleNamespace(
        harness_attachment_root="data/harness_attachments",
        harness_attachment_max_bytes=1024 * 1024,
        harness_attachment_ttl_seconds=86400,
    ))
    record = harness_attachments.save_attachment(
        user_id="same-user", internal_session_id="same-session",
        filename="image.png", content=b"image",
    )

    monkeypatch.chdir(other_cwd)

    assert harness_attachments.harness_attachment_root() == (
        project_root / "data" / "harness_attachments"
    ).resolve()
    assert harness_attachments.list_attachments(
        user_id="same-user", internal_session_id="same-session"
    )[0]["attachment_id"] == record["attachment_id"]


def test_mcp_lists_same_attachment_from_different_workdir(monkeypatch, tmp_path) -> None:
    _configure_root(monkeypatch, tmp_path)
    record = harness_attachments.save_attachment(
        user_id="mcp-user", internal_session_id="mcp-session",
        filename="input.png", content=b"image",
    )
    other_cwd = tmp_path / "mcp-cwd"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    monkeypatch.setattr(mcp_server, "USER_ID", "mcp-user")
    monkeypatch.setattr(mcp_server, "INTERNAL_SESSION_ID", "mcp-session")

    result = mcp_server.call("user_attachment_list", {})
    payload = json.loads(result["content"][0]["text"])

    assert result["isError"] is False
    assert payload[0]["attachment_id"] == record["attachment_id"]


def test_model_parse_returns_preview_and_bounded_search(monkeypatch, tmp_path) -> None:
    root = _configure_root(monkeypatch, tmp_path)
    record = harness_attachments.save_attachment(
        user_id="user-a", internal_session_id="session-a",
        filename="report.docx", content=b"document",
    )
    monkeypatch.setattr(document_parser, "parse_document", lambda *_args, **_kwargs: [
        {"type": "paragraph", "style": "正文", "text": "alpha " * 800},
        {"type": "paragraph", "style": "正文", "text": "关键结论是耐腐蚀。"},
    ])
    parsed = harness_attachments.parse_attachment(
        user_id="user-a", internal_session_id="session-a",
        attachment_id=record["attachment_id"],
    )
    assert parsed["status"] == "parsed"
    assert len(parsed["preview"]) <= 2400
    assert "path" not in json.dumps(parsed)
    result = harness_attachments.read_attachment(
        user_id="user-a", internal_session_id="session-a",
        attachment_id=record["attachment_id"], query="耐腐蚀",
    )
    assert "耐腐蚀" in result["chunks"][0]["text"]
    assert len(result["chunks"]) <= 4
    assert str(root) not in json.dumps(result, ensure_ascii=False)


def test_parser_artifacts_stay_inside_attachment_boundary(monkeypatch, tmp_path) -> None:
    root = _configure_root(monkeypatch, tmp_path)
    source = root / "user-key" / "session-key" / "attachment-key" / "original" / "same.docx"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"x")
    assert parser_paths.processing_document_dir(source) == (
        root / "user-key" / "session-key" / "attachment-key" / "parsed"
    )


def test_model_parser_does_not_publish_into_knowledge_base(monkeypatch, tmp_path) -> None:
    from app.services.parser import word_parser

    source = tmp_path / "model.docx"
    source.write_bytes(b"placeholder")
    monkeypatch.setattr(word_parser, "parse_word_document", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        document_parser,
        "synchronize_processed_assets",
        lambda *_args, **_kwargs: pytest.fail("model attachment must not be published"),
    )
    assert document_parser.parse_document(source, source="model") == []


def test_delete_session_attachments_does_not_touch_other_users(monkeypatch, tmp_path) -> None:
    _configure_root(monkeypatch, tmp_path)
    harness_attachments.save_attachment(
        user_id="user-a", internal_session_id="session-a", filename="a.pdf", content=b"a"
    )
    harness_attachments.save_attachment(
        user_id="user-b", internal_session_id="session-a", filename="b.pdf", content=b"b"
    )
    assert harness_attachments.delete_session_attachments(
        user_id="user-a", internal_session_id="session-a"
    ) == 1
    assert harness_attachments.list_attachments(
        user_id="user-a", internal_session_id="session-a"
    ) == []
    assert len(harness_attachments.list_attachments(
        user_id="user-b", internal_session_id="session-a"
    )) == 1

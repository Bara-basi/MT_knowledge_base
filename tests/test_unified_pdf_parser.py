from __future__ import annotations

import pytest

from app.db.minio import RawDocumentObject
from app.services.llm import LLMTimeoutError
from app.services.parser import unified_pdf_parser
from app.services.parser.unified_pdf_parser import (
    PdfExtractionError,
    PdfLine,
    PdfPageKind,
    PdfPageProfile,
    PdfVisionSession,
    _apply_vision_line_corrections,
    _assign_table_blocks,
    _assign_heading_styles,
    _build_pdf_items,
    _classify_page_kind,
    _chat_json_object,
    _clean_pdf_lines,
    _coerce_model_table_rows,
    _consolidate_local_table_lines,
    _extract_page_with_vision_model,
    _extract_tables_with_vision_model,
    _filter_raster_table_regions,
    _is_obvious_garbage,
    _native_text_quality,
    _normalize_pdf_text,
    _parse_table_text_protocol,
    _parse_table_text_protocol_payload,
    _replace_lines_with_visual_tables,
    _should_use_vision,
    _should_use_full_page_vision,
    resolve_pdf_document_sources,
)


def _line(
    text: str,
    *,
    block_type: str = "paragraph",
    font_size: float = 10.0,
    bold: bool = False,
    alignment: str = "left",
    order: int = 0,
    x0: float = 72,
    x1: float = 420,
    y0: float | None = None,
) -> PdfLine:
    top = y0 if y0 is not None else 72 + order * 20
    return PdfLine(
        text=text,
        page_number=1,
        page_width=600,
        page_height=800,
        x0=x0,
        y0=top,
        x1=x1,
        y1=top + 14,
        font_size=font_size,
        bold=bold,
        order=order,
        alignment=alignment,
        block_type=block_type,
    )


def test_page_classification_covers_native_scanned_hybrid_and_blank() -> None:
    assert _classify_page_kind(
        native_chars=500,
        native_quality=1.0,
        image_coverage=0.0,
        ink_ratio=0.0,
        image_count=0,
        drawing_count=0,
    ) == PdfPageKind.NATIVE_TEXT
    assert _classify_page_kind(
        native_chars=0,
        native_quality=0.0,
        image_coverage=0.9,
        ink_ratio=0.2,
        image_count=1,
        drawing_count=0,
    ) == PdfPageKind.SCANNED
    assert _classify_page_kind(
        native_chars=500,
        native_quality=1.0,
        image_coverage=0.7,
        ink_ratio=0.0,
        image_count=1,
        drawing_count=0,
    ) == PdfPageKind.HYBRID
    assert _classify_page_kind(
        native_chars=0,
        native_quality=0.0,
        image_coverage=0.0,
        ink_ratio=0.0,
        image_count=0,
        drawing_count=0,
    ) == PdfPageKind.BLANK


def test_bad_native_font_mapping_is_routed_to_ocr() -> None:
    quality = _native_text_quality("犐犆犛犌犅犜犠犲犾犱犲犱狊狋犪犻狀犾犲狊狊")
    assert quality < 0.8
    assert _classify_page_kind(
        native_chars=200,
        native_quality=quality,
        image_coverage=0.0,
        ink_ratio=0.0,
        image_count=0,
        drawing_count=0,
    ) == PdfPageKind.SCANNED


def test_broken_embedded_latin_font_is_repaired_before_model_input() -> None:
    assert _normalize_pdf_text(
        "犌犅/犜 ２２３．８４ 犠=0.024犛(犇-犛) 犘=2犛犚/犇 犃/%",
        font_name="E-HZ9-PK7483a5",
    ) == "GB/T 223.84 W=0.024S(D-S) p=2SR/D A/%"
    # The same rare Chinese character in an unrelated font is preserved.
    assert _normalize_pdf_text("犛", font_name="SimSun") == "犛"


def test_structure_labels_emit_chunker_compatible_items() -> None:
    lines = [
        _line("统一解析测试文档", font_size=20, bold=True, alignment="center", order=0),
        _line("1 范围", font_size=12, bold=True, order=1),
        _line("正文内容", order=2),
        _line("牌号  成分  数值", block_type="table", order=3),
    ]
    _assign_heading_styles(lines)
    items = _build_pdf_items(lines)

    assert items[0]["style"] == "标题"
    assert items[1]["style"].startswith("标题")
    assert items[2]["style"] == "正文"
    assert items[3]["type"] == "table"
    assert items[3]["style"] == "表格"


def test_date_line_is_not_promoted_to_heading() -> None:
    line = _line("1989-01-01实施", font_size=12)
    _assign_heading_styles([line])
    assert line.style == "正文"


def test_two_column_prose_is_not_inferred_as_table() -> None:
    lines: list[PdfLine] = []
    for row in range(3):
        top = 100 + row * 20
        lines.extend(
            [
                _line("left column prose", x0=40, x1=260, y0=top),
                _line("right column prose", x0=330, x1=550, y0=top),
            ]
        )
    _assign_table_blocks(lines, [])
    assert all(line.block_type == "paragraph" for line in lines)


def test_note_is_body_text_and_measurement_range_is_not_heading() -> None:
    note = _line(
        "注：试样尺寸有特殊要求时，按双方协议执行。",
        font_size=12,
        bold=True,
        order=0,
    )
    measurement = _line(
        "100~140",
        font_size=16,
        bold=True,
        alignment="center",
        order=1,
    )

    _assign_table_blocks([note, measurement], [])
    _assign_heading_styles([note, measurement])

    assert note.block_type == "paragraph"
    assert note.style == "正文"
    assert measurement.block_type == "paragraph"
    assert measurement.style == "正文"


def test_table_title_without_description_is_preserved() -> None:
    caption = _line("表 1", order=0)
    _assign_table_blocks([caption], [])
    items = _build_pdf_items([caption])

    assert len(items) == 1
    assert items[0]["type"] == "table"
    assert items[0]["style"] == "表标题"
    assert items[0]["text"] == "表 1"
    assert items[0]["page_number"] == 1


def test_series_prefix_table_title_is_preserved() -> None:
    caption = _line("37系列表2", order=0)
    _assign_table_blocks([caption], [])
    _assign_heading_styles([caption])
    item = _build_pdf_items([caption])[0]

    assert item["type"] == "table"
    assert item["style"] == "表标题"


def test_raster_table_filter_rejects_diagram_and_keeps_text_grid() -> None:
    diagram = unified_pdf_parser.fitz.Rect(300, 50, 500, 350)
    table = unified_pdf_parser.fitz.Rect(40, 400, 560, 650)
    lines = [
        _line("R", x0=350, x1=360, y0=100),
        _line("X", x0=400, x1=410, y0=200),
        _line("p", x0=430, x1=440, y0=300),
    ]
    for row in range(3):
        for column in range(3):
            lines.append(
                _line(
                    f"cell-{row}-{column}",
                    x0=80 + column * 150,
                    x1=150 + column * 150,
                    y0=430 + row * 50,
                )
            )

    assert _filter_raster_table_regions([diagram, table], lines) == [table]


def test_visual_table_rows_drop_repeated_header_and_duplicate_rows() -> None:
    rows = _coerce_model_table_rows(
        {
            "headers": ["序号", "牌号"],
            "rows": [
                {"序号": "序号", "牌号": "牌号"},
                {"序号": "1", "牌号": "S30210"},
                {"序号": "1", "牌号": "S30210"},
            ],
        }
    )
    assert rows == [{"序号": "1", "牌号": "S30210"}]


def test_visual_table_model_returns_embedding_compatible_json() -> None:
    class FakeClient:
        def chat(self, messages, **kwargs) -> str:
            content = messages[0]["content"]
            assert content[0]["type"] == "text"
            assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
            assert kwargs["stream"] is False
            assert "response_format" not in kwargs["extra_body"]
            return "HEADER\t序号\t牌号\nROW\t序号\t牌号\nROW\t1\tS30210"

    document = unified_pdf_parser.fitz.open()
    page = document.new_page(width=600, height=800)
    region = unified_pdf_parser.fitz.Rect(60, 160, 540, 640)
    client = FakeClient()
    tables = _extract_tables_with_vision_model(
        page,
        page_number=1,
        table_regions=[region],
        draft_lines=[_line("散列表格", block_type="table", y0=200)],
        llm_client=client,
    )

    assert len(tables) == 1
    assert tables[0].extraction_method == "vision_table_tsv"
    assert tables[0].text == '[{"序号":"1","牌号":"S30210"}]'
    document.close()


def test_plain_line_correction_preserves_layout_and_missing_records() -> None:
    first = _line("焊按接头", y0=100)
    second = _line("保留原行", y0=130)
    output = _apply_vision_line_corrections(
        "L1\t焊接接头\n模型解释应被忽略",
        [first, second],
    )

    assert [line.text for line in output] == ["焊接接头", "保留原行"]
    assert output[0].y0 == first.y0
    assert output[0].font_size == first.font_size
    assert output[0].extraction_method == "vision_line_correction"


def test_line_correction_api_failure_aborts_document(monkeypatch) -> None:
    class FakePage:
        rect = unified_pdf_parser.fitz.Rect(0, 0, 600, 800)

    def fail_request(*_args, **_kwargs):
        raise unified_pdf_parser.LLMAPIError(
            "Failed to upload LLM file: [WinError 10054]"
        )

    monkeypatch.setattr(
        unified_pdf_parser,
        "_request_pdf_vision_text",
        fail_request,
    )

    with pytest.raises(
        PdfExtractionError,
        match=r"Required vision line correction failed on page 11",
    ):
        _extract_page_with_vision_model(
            FakePage(),
            object(),
            page_number=11,
            ocr_lines=[_line("必须经过模型校对的 OCR 文本")],
            llm_client=object(),
        )


def test_table_text_protocol_deduplicates_headers_and_rows() -> None:
    rows = _parse_table_text_protocol(
        "TITLE\t表1\n"
        "HEADER\t牌号\t牌号\n"
        "ROW\t牌号\t牌号\n"
        "ROW\tS30210\tA\n"
        "ROW\tS30210\tA\n"
    )
    assert rows == [{"牌号": "S30210", "牌号_2": "A"}]


def test_table_text_protocol_preserves_table_title() -> None:
    title, rows = _parse_table_text_protocol_payload(
        "TITLE\t表 2 化学成分\n"
        "HEADER\t牌号\t碳\n"
        "ROW\tS30210\t0.08\n"
    )

    assert title == "表 2 化学成分"
    assert rows == [{"牌号": "S30210", "碳": "0.08"}]


def test_table_text_protocol_accepts_markdown_fallback_without_json() -> None:
    rows = _parse_table_text_protocol(
        "| 项目 | 数值 |\n"
        "| --- | --- |\n"
        "| 屈服强度 | 205 |\n"
    )
    assert rows == [{"项目": "屈服强度", "数值": "205"}]


def test_visual_table_replaces_only_lines_inside_detected_region() -> None:
    body = _line("正文", y0=50)
    table_fragment = _line("散列表格", block_type="table", y0=200)
    visual_table = _line(
        '[{"列":"值"}]',
        block_type="table",
        y0=180,
        x0=50,
        x1=500,
    )
    region = unified_pdf_parser.fitz.Rect(40, 170, 520, 300)
    output = _replace_lines_with_visual_tables(
        [body, table_fragment],
        [visual_table],
        table_regions=[region],
    )
    assert [line.text for line in output] == ["正文", '[{"列":"值"}]']


def test_low_value_watermark_and_publisher_lines_are_removed() -> None:
    assert _is_obvious_garbage("www.newMaker.com")
    assert _is_obvious_garbage("版权专有不得翻印")
    assert _is_obvious_garbage("新华书店北京发行所发行各地新华书店经售")
    assert not _is_obvious_garbage("正文参见 https://example.com/spec")


def test_colophon_pollution_is_removed_without_deleting_standard_identity() -> None:
    kept = _clean_pdf_lines(
        [
            _line("源自网络", order=0),
            _line("中华人民共和国", order=1),
            _line("国家标准", order=2),
            _line("钢的脱碳层深度测定法", order=3),
            _line("GB224-87", order=4),
            _line(
                "中国标准出版社出版（北京复外三里河）中国标准出版社北京印刷厂印刷",
                order=5,
            ),
            _line("新华书店北京发行所发行各地新华书店经售", order=6),
            _line("版权专有不得翻印", order=7),
            _line("普", order=8),
            _line("开本880×1230 1/16 印张1/2 字数6000", order=9),
            _line("1988年7月第一版1988年7月第一次印刷", order=10),
            _line("印数1-5000", order=11),
            _line("标目94—8", order=12),
        ]
    )

    assert [line.text for line in kept] == [
        "中华人民共和国",
        "国家标准",
        "钢的脱碳层深度测定法",
        "GB224-87",
    ]


def test_table_title_and_local_cells_match_cleaner_and_chunker_contract() -> None:
    lines = [
        _line("2 技术要求", font_size=14, bold=True, order=0, y0=60),
        _line("表 2 化学成分", order=1, y0=100),
        _line("牌号", block_type="table", order=2, x0=60, x1=150, y0=130),
        _line("碳", block_type="table", order=3, x0=200, x1=260, y0=130),
        _line("S30210", block_type="table", order=4, x0=60, x1=150, y0=155),
        _line("0.08", block_type="table", order=5, x0=200, x1=260, y0=155),
    ]
    table_region = unified_pdf_parser.fitz.Rect(40, 120, 300, 180)
    _assign_table_blocks(lines, [table_region])
    _assign_heading_styles(lines)
    items = _build_pdf_items(_consolidate_local_table_lines(lines))

    assert items[1]["type"] == "table"
    assert items[1]["style"] == "表标题"
    assert items[2]["type"] == "table"
    assert items[2]["text"] == '[["牌号","碳"],["S30210","0.08"]]'

    from app.services.chunking.splitter import split_items
    from app.services.data_clean import clean_items

    cleaned = clean_items(items)
    assert cleaned[1]["text"] == "表 2 化学成分"
    assert cleaned[2]["text"] == '[["牌号","碳"],["S30210","0.08"]]'

    chunks = split_items(cleaned)
    table_chunks = [
        chunk for chunk in chunks if chunk.metadata.get("chunk_type") == "table"
    ]
    assert len(table_chunks) == 1
    assert table_chunks[0].metadata["path"].endswith("表 2 化学成分")
    assert table_chunks[0].content == '[["牌号","碳"],["S30210","0.08"]]'


def test_auto_model_mode_reviews_scanned_page_even_with_high_ocr_confidence() -> None:
    profile = PdfPageProfile(
        page_number=1,
        kind=PdfPageKind.SCANNED,
        native_text_chars=0,
        native_text_quality=0,
        image_coverage=1,
        ink_ratio=0.1,
        image_count=1,
        drawing_count=0,
        width=600,
        height=800,
        rotation=0,
    )
    line = _line("高置信度但可能有错字")
    line.confidence = 0.99
    assert _should_use_vision("auto", profile, [line])
    assert not _should_use_full_page_vision(
        "auto",
        profile,
        [line],
        table_regions=[unified_pdf_parser.fitz.Rect(10, 10, 100, 100)],
    )


def test_json_model_request_retries_empty_response_without_json_mode() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def chat(self, *_args, **kwargs) -> str:
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return ""
            return '```json\n{"lines":[]}\n```'

    client = FakeClient()
    payload = _chat_json_object(
        client,
        [{"role": "user", "content": "return JSON"}],
        max_tokens=100,
        purpose="test",
    )

    assert payload == {"lines": []}
    assert "response_format" in client.calls[0]["extra_body"]
    assert "response_format" not in client.calls[1]["extra_body"]
    assert client.calls[0]["max_tokens"] == 100
    assert client.calls[1]["max_tokens"] == 100


def test_json_model_timeout_does_not_retry_and_opens_task_circuit_after_threshold() -> None:
    class TimeoutClient:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, *_args, **_kwargs) -> str:
            self.calls += 1
            raise LLMTimeoutError("read timed out")

    client = TimeoutClient()
    session = PdfVisionSession()

    try:
        _chat_json_object(
            client,
            [{"role": "user", "content": "return JSON"}],
            max_tokens=100,
            purpose="page 3 table region 1",
            task="table",
            read_timeout=12,
            vision_session=session,
        )
    except LLMTimeoutError:
        pass
    else:
        raise AssertionError("timeout should propagate to the local fallback")

    assert client.calls == 1
    assert session.available("table")
    assert session.available("full_page")
    assert session.timeout_purposes == ["page 3 table region 1"]

    try:
        _chat_json_object(
            client,
            [{"role": "user", "content": "return JSON"}],
            max_tokens=100,
            purpose="page 4 table region 1",
            task="table",
            read_timeout=12,
            vision_session=session,
        )
    except LLMTimeoutError:
        pass
    else:
        raise AssertionError("second timeout should propagate to the local fallback")

    assert client.calls == 2
    assert not session.available("table")
    assert session.available("full_page")


def test_output_budget_failure_opens_only_affected_vision_circuit() -> None:
    session = PdfVisionSession()

    session.mark_failure(
        "table",
        "page 3 table region 1",
        "finish_reason='length', reasoning_length=4285",
        terminal=True,
    )

    assert not session.available("table")
    assert session.available("full_page")
    assert session.failure_purposes == ["page 3 table region 1"]


def test_auto_full_page_vision_skips_sparse_scan() -> None:
    profile = PdfPageProfile(
        page_number=1,
        kind=PdfPageKind.SCANNED,
        native_text_chars=0,
        native_text_quality=0,
        image_coverage=1,
        ink_ratio=0.01,
        image_count=1,
        drawing_count=0,
        width=600,
        height=800,
        rotation=0,
    )
    assert not _should_use_full_page_vision(
        "auto",
        profile,
        [_line("仅有页眉")],
        table_regions=[],
    )
    assert _should_use_full_page_vision(
        "always",
        profile,
        [_line("仅有页眉")],
        table_regions=[],
    )


def test_asme_parent_prefers_existing_split_minio_objects(monkeypatch) -> None:
    section = RawDocumentObject(
        "knowledge-raw-docs",
        "产品标准/ASME-Sec-II-A-Vol1-2023(切分版)/SA-105.pdf",
    )
    monkeypatch.setattr(
        unified_pdf_parser,
        "list_raw_document_objects",
        lambda *_args, **_kwargs: [section],
    )

    resolved = resolve_pdf_document_sources(
        "minio://knowledge-raw-docs/产品标准/ASME-Sec-II-A-Vol1-2023.pdf",
        split_if_missing=False,
    )

    assert resolved == [section.uri]

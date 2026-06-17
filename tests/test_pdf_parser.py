from __future__ import annotations

from app.services.parser.pdf_parser import (
    PdfLine,
    _assign_page_alignment,
    _assign_heading_styles,
    _is_bold_font,
    _clean_pdf_lines,
    _is_heading_candidate,
    _link_items,
)


def _line(
    text: str,
    *,
    page: int = 1,
    alignment: str = "left",
    bold: bool = False,
    font_size: float = 12.0,
    order: int = 1,
    x0: float = 72,
    x1: float = 240,
    y0: float = 72,
    y1: float = 90,
) -> PdfLine:
    return PdfLine(
        text=text,
        page_number=page,
        page_width=600,
        page_height=800,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        font_size=font_size,
        bold=bold,
        order=order,
        alignment=alignment,
    )


def test_pdf_alignment_uses_center_point_for_centered_text() -> None:
    line = _line("考勤及休假管理", x0=246.24, x1=318.47, font_size=9.95)

    _assign_page_alignment([line])

    assert line.alignment == "center"
    assert _is_heading_candidate(line)


def test_pdf_date_prefix_is_rejected_even_when_not_left_aligned() -> None:
    line = _line("2025-09-15 实施", x0=369.48, x1=493.42, bold=True, font_size=17.5)

    _assign_page_alignment([line])

    assert line.alignment == "right"
    assert not _is_heading_candidate(line)


def test_pdf_heading_candidate_uses_alignment_numbering_bold_and_date_signals() -> None:
    assert _is_heading_candidate(_line("1. 标题", alignment="left", bold=False))
    assert not _is_heading_candidate(_line("2024-06-16", alignment="left", bold=True))
    assert not _is_heading_candidate(_line("2025-09-15 实施", alignment="left", bold=True))
    assert not _is_heading_candidate(_line("- 列表项", alignment="left", bold=True))
    assert _is_heading_candidate(_line("居中标题", alignment="center", bold=True))
    assert _is_heading_candidate(_line("居中非粗体", alignment="center", bold=False))
    assert not _is_heading_candidate(_line("右侧标题", alignment="right", bold=True))


def test_pdf_heading_levels_follow_numbering_hierarchy_before_visual_score() -> None:
    lines = [
        _line("一、一级标题", alignment="left", bold=False, font_size=12, order=1),
        _line("（一）二级标题", alignment="left", bold=False, font_size=20, order=2),
        _line("1. 三级标题", alignment="left", bold=True, font_size=22, order=3),
        _line("（1）四级标题", alignment="left", bold=True, font_size=18, order=4),
        _line("① 五级标题", alignment="left", bold=True, font_size=18, order=5),
        _line("居中章标题", alignment="center", bold=True, font_size=24, order=6),
    ]

    _assign_heading_styles(lines)

    assert [line.style for line in lines] == [
        "标题 2",
        "标题 3",
        "标题 4",
        "标题 5",
        "标题 6",
        "标题",
    ]


def test_pdf_numbered_headings_keep_original_levels_without_center_bold_top_heading() -> None:
    lines = [
        _line("一、一级标题", alignment="left", bold=False, font_size=12, order=1),
        _line("（一）二级标题", alignment="left", bold=False, font_size=20, order=2),
        _line("1. 三级标题", alignment="left", bold=True, font_size=22, order=3),
        _line("（1）四级标题", alignment="left", bold=True, font_size=18, order=4),
        _line("① 五级标题", alignment="left", bold=True, font_size=18, order=5),
    ]

    _assign_heading_styles(lines)

    assert [line.style for line in lines] == [
        "标题",
        "标题 2",
        "标题 3",
        "标题 4",
        "标题 5",
    ]


def test_pdf_long_lowest_level_heading_is_demoted_only_for_that_line() -> None:
    lines = [
        _line("1. 短标题", alignment="left", bold=True, font_size=16, order=1),
        _line("2. 这是一个超过二十个字符的最低级长标题需要退化", alignment="left", bold=True, font_size=16, order=2),
        _line("3. 另一个短标题", alignment="left", bold=True, font_size=16, order=3),
    ]

    _assign_heading_styles(lines)

    assert lines[0].style == "标题 3"
    assert lines[1].style == "正文"
    assert lines[2].style == "标题 3"


def test_pdf_bold_signal_splits_same_size_visual_heading_levels() -> None:
    lines = [
        _line("居中粗体标题", alignment="center", bold=True, font_size=18, order=1),
        _line("另一居中粗体标题", alignment="center", bold=True, font_size=18, order=2),
        _line("其它位置粗体标题", alignment="other", bold=True, font_size=18, order=3),
    ]

    _assign_heading_styles(lines)

    assert lines[0].style == "标题"
    assert lines[1].style == "标题"
    assert lines[2].style == "标题 2"


def test_pdf_centered_heading_without_bold_signal_is_top_level() -> None:
    lines = [
        _line("考勤及休假管理", alignment="center", bold=False, font_size=9.95, order=1),
        _line("一、考勤管理", alignment="left", bold=True, font_size=12, order=2),
        _line("1、打卡要求", alignment="left", bold=True, font_size=12, order=3),
    ]

    _assign_heading_styles(lines)

    assert lines[0].style == "标题"
    assert lines[1].style == "标题 2"
    assert lines[2].style == "标题 4"


def test_pdf_centered_cover_metadata_and_table_values_are_not_headings() -> None:
    lines = [
        _line("MT-WI-HR-002", alignment="right", bold=False, font_size=21.5, order=1, x0=420, x1=550, y0=139, y1=161),
        _line("规章制度", alignment="center", bold=False, font_size=26.05, order=2, x0=245.3, x1=348.6, y0=207.5, y1=233.5),
        _line("版次： A/0 版", alignment="center", bold=True, font_size=20.05, order=3, x0=233.8, x1=369.1, y0=246.6, y1=274.4),
        _line("编 制：", alignment="left", bold=False, font_size=17.5, order=4, y0=310.3, y1=328.5),
        _line("版本号", alignment="left", bold=False, font_size=14.05, order=5, y0=553.9, y1=568.5),
        _line("A/0", alignment="center", bold=False, font_size=14.05, order=6, x0=295.4, x1=315.9, y0=553.9, y1=568.5),
        _line("发放编号", alignment="left", bold=False, font_size=14.05, order=7, y0=584.9, y1=599.5),
        _line("条件", alignment="left", bold=False, font_size=9.95, order=8, y0=620, y1=630),
        _line("可享年假天数", alignment="other", bold=False, font_size=9.95, order=9, y0=620, y1=630),
        _line("入职工龄满一年", alignment="left", bold=False, font_size=9.95, order=10, y0=643.5, y1=653.5),
        _line("5天", alignment="center", bold=False, font_size=9.95, order=11, x0=308.5, x1=323.8, y0=643.5, y1=653.5),
        _line("入职工龄两年（含）以上", alignment="left", bold=False, font_size=9.95, order=12, y0=666.9, y1=676.9),
        _line("6天", alignment="center", bold=False, font_size=9.95, order=13, x0=310.3, x1=325.4, y0=666.9, y1=676.9),
    ]

    _assign_heading_styles(lines)

    assert lines[1].style == "标题"
    assert lines[2].style == "正文"
    assert lines[5].style == "正文"
    assert lines[10].style == "正文"
    assert lines[12].style == "正文"


def test_pdf_consecutive_same_level_sentence_like_headings_are_demoted() -> None:
    lines = [
        _line("（一）累计工作满1年不满10年的职工，请病假累计2个月以上的。", alignment="left", bold=True, font_size=16, order=1),
        _line("（二）累计工作满10年不满20年的职工，请病假累计3个月以上的。", alignment="left", bold=True, font_size=16, order=2),
        _line("（三）累计工作满20年以上的职工，请病假累计4个月以上的。", alignment="left", bold=True, font_size=16, order=3),
    ]

    _assign_heading_styles(lines)

    assert [line.style for line in lines] == ["正文", "正文", "正文"]


def test_pdf_bold_font_detection_is_conservative() -> None:
    assert _is_bold_font("SourceHanSansCN-Bold", 0)
    assert _is_bold_font("Arial-Black", 0)
    assert not _is_bold_font("SourceHanSansCN-Regular", 16)
    assert not _is_bold_font("ArialMT", 0)


def test_pdf_clean_lines_removes_page_numbers_and_repeated_standalone_text() -> None:
    lines = [
        _line("嘉兴迈拓不锈钢有限公司", page=1, order=1),
        _line("1. 正文标题", page=1, order=2),
        _line("1", page=1, order=3),
        _line("嘉兴迈拓不锈钢有限公司", page=2, order=4),
        _line("第二页正文", page=2, order=5),
        _line("2", page=2, order=6),
    ]

    cleaned = _clean_pdf_lines(lines)

    assert [line.text for line in cleaned] == ["1. 正文标题", "第二页正文"]


def test_pdf_link_items_use_visible_urls_and_annotation_urls() -> None:
    line = _line("查看在线样册：https://example.com/a", order=7)
    line.urls = ["https://example.com/a", "https://example.com/annotated"]

    links = _link_items(line)

    assert [item["url"] for item in links] == [
        "https://example.com/a",
        "https://example.com/annotated",
    ]
    assert links[0]["description"] == "https://example.com/a"
    assert links[1]["description"] == line.text

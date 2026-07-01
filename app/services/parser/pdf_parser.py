from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

try:
    import fitz
except ModuleNotFoundError:  # pragma: no cover - exercised only in misconfigured envs
    fitz = None

try:
    from app.services.data_clean import clean_items
    from app.services.parser.paths import processing_subdir
    from app.services.parser.word_parser import format_extracted_items
except ModuleNotFoundError:
    from data_clean import clean_items
    from paths import processing_subdir
    from word_parser import format_extracted_items


BODY_STYLE = "正文"
LINK_STYLE = "链接"
HEADING_STYLE = "标题"
MAX_HEADING_LEVEL = 6
MAX_LOWEST_HEADING_CHARS = 15
ROW_TOP_TOLERANCE = 3.0
HEADER_FOOTER_REPEAT_RATIO = 0.35
MIN_HEADER_FOOTER_REPEATS = 2
BOLD_SCORE_BOOST = 2.0
CENTER_SCORE_BOOST = 2.0
HEADING_SCORE_CLUSTER_TOLERANCE = 0.4
LARGE_FONT_BOOST = 4.0
LARGE_GAP_MULTIPLIER = 2.0

NUMBERED_HEADING_PATTERN = re.compile(
    r"^\s*(?:"
    r"(?:\d{1,3}(?:\.\d{1,3}){0,5}|[一二三四五六七八九十百千万]{1,8}|[IVXLCDM]{1,8})"
    r"\s*[\.、．)]|"
    r"(?:第\s*[\d一二三四五六七八九十百千万]{1,8}\s*[章节篇部分条])|"
    r"(?:part|chapter|section)\s*[\dIVXLCDM一二三四五六七八九十百千万]+"
    r")",
    re.IGNORECASE,
)
UNORDERED_LIST_PATTERN = re.compile(r"^\s*(?:[-*•·●○◆◇■□▪▫▶▷◦]|[（(]?[a-zA-Z][）)])\s+")
DATE_LINE_PATTERN = re.compile(
    r"^\s*(?:"
    r"\d{4}\s*[-/.年]\s*\d{1,2}\s*(?:[-/.月]\s*\d{1,2}\s*日?)?|"
    r"\d{1,2}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{2,4}|"
    r"\d{4}\s*年\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?"
    r")\s*$"
)
DATE_PREFIX_PATTERN = re.compile(
    r"^\s*\d{4}\s*[-/.年]\s*\d{1,2}\s*(?:[-/.月]\s*\d{1,2}\s*日?)?"
)
PAGE_NUMBER_PATTERN = re.compile(r"^\s*(?:第\s*)?\d{1,3}\s*(?:页)?\s*$")
URL_PATTERN = re.compile(r"https?://[^\s<>()]+", re.IGNORECASE)
CENTER_METADATA_PATTERN = re.compile(
    r"^\s*(?:"
    r"(?:版次|版本|版本号|受控状态|发放编号|编\s*制|审\s*核|批\s*准)\s*[:：]?.*|"
    r"[A-Z]/\d(?:\s*版)?|"
    r"\d+(?:\.\d+)?\s*(?:天|年|月|日|分|元|%)"
    r")\s*$",
    re.IGNORECASE,
)


@dataclass
class PdfLine:
    text: str
    page_number: int
    page_width: float
    page_height: float
    x0: float
    y0: float
    x1: float
    y1: float
    font_size: float
    bold: bool
    order: int
    alignment: str = "left"
    style: str = BODY_STYLE
    urls: list[str] = field(default_factory=list)


def parse_pdf_document(
    file_path: str | Path,
    *,
    image_analysis_workers: int = 3,
) -> list[dict[str, Any]]:
    """Extract text-first PDFs into paragraph/link items compatible with splitter.py."""
    del image_analysis_workers
    started_at = time.perf_counter()
    if fitz is None:
        raise ModuleNotFoundError("pymupdf is required to parse PDF documents")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF document not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Only .pdf files are supported: {path}")
    if path.name.startswith("~$"):
        raise ValueError(f"Temporary PDF lock files are not supported: {path}")

    _log(f"start parsing: {path}")
    stage_started_at = time.perf_counter()
    lines = _extract_lines(path)
    _log(f"extracted lines: total={len(lines)} ({time.perf_counter() - stage_started_at:.2f}s)")

    stage_started_at = time.perf_counter()
    lines = _clean_pdf_lines(lines)
    _log(f"cleaned pdf lines: total={len(lines)} ({time.perf_counter() - stage_started_at:.2f}s)")

    stage_started_at = time.perf_counter()
    _assign_heading_styles(lines)
    items = clean_items(_build_items(lines))
    _log_item_summary("built pdf items", items, stage_started_at)

    stage_started_at = time.perf_counter()
    output_path = write_items_to_txt(items, path)
    _log(f"wrote parsed txt: {output_path} ({time.perf_counter() - stage_started_at:.2f}s)")
    _log(f"finished parsing: {path.name} ({time.perf_counter() - started_at:.2f}s)")
    return items


def _extract_lines(path: Path) -> list[PdfLine]:
    output: list[PdfLine] = []
    order = 0
    with fitz.open(path) as document:
        for page_index, page in enumerate(document, start=1):
            page_width = float(page.rect.width)
            page_height = float(page.rect.height)
            blocks = page.get_text("dict", sort=True).get("blocks", [])
            page_lines: list[PdfLine] = []
            page_links = _page_uri_links(page)
            for block in blocks:
                if block.get("type") != 0:
                    continue
                for raw_line in block.get("lines", []):
                    line = _line_from_spans(raw_line.get("spans", []), page_index, page_width, page_height, order)
                    if line is None:
                        continue
                    order += 1
                    page_lines.append(line)

            page_lines = _merge_same_row_lines(page_lines)
            _assign_line_links(page_lines, page_links)
            _assign_page_alignment(page_lines)
            output.extend(page_lines)
    return output


def _line_from_spans(
    spans: list[dict[str, Any]],
    page_number: int,
    page_width: float,
    page_height: float,
    order: int,
) -> PdfLine | None:
    text_parts: list[str] = []
    sizes: list[float] = []
    bold = False
    boxes: list[tuple[float, float, float, float]] = []

    for span in spans:
        text = _clean_line_text(str(span.get("text") or ""))
        if not text:
            continue
        text_parts.append(text)
        size = float(span.get("size") or 0)
        if size > 0:
            sizes.append(size)
        font = str(span.get("font") or "")
        flags = int(span.get("flags") or 0)
        bold = bold or _is_bold_font(font, flags)
        bbox = span.get("bbox") or (0, 0, 0, 0)
        boxes.append(tuple(float(value) for value in bbox))

    text = _clean_line_text(" ".join(text_parts))
    if not text or not boxes:
        return None

    return PdfLine(
        text=text,
        page_number=page_number,
        page_width=page_width,
        page_height=page_height,
        x0=min(box[0] for box in boxes),
        y0=min(box[1] for box in boxes),
        x1=max(box[2] for box in boxes),
        y1=max(box[3] for box in boxes),
        font_size=max(sizes, default=0.0),
        bold=bold,
        order=order,
    )


def _merge_same_row_lines(lines: list[PdfLine]) -> list[PdfLine]:
    if len(lines) < 2:
        return lines

    rows: list[list[PdfLine]] = []
    for line in sorted(lines, key=lambda item: (item.y0, item.x0, item.order)):
        for row in rows:
            if abs(line.y0 - row[0].y0) <= ROW_TOP_TOLERANCE:
                row.append(line)
                break
        else:
            rows.append([line])

    output: list[PdfLine] = []
    for row in rows:
        row = sorted(row, key=lambda item: item.x0)
        merged = row[0]
        for line in row[1:]:
            gap = line.x0 - merged.x1
            if gap > max(12.0, merged.font_size * 1.2):
                output.append(merged)
                merged = line
                continue
            merged = PdfLine(
                text=_clean_line_text(f"{merged.text} {line.text}"),
                page_number=merged.page_number,
                page_width=merged.page_width,
                page_height=merged.page_height,
                x0=min(merged.x0, line.x0),
                y0=min(merged.y0, line.y0),
                x1=max(merged.x1, line.x1),
                y1=max(merged.y1, line.y1),
                font_size=max(merged.font_size, line.font_size),
                bold=merged.bold or line.bold,
                order=min(merged.order, line.order),
                urls=_dedupe_texts([*merged.urls, *line.urls]),
            )
        output.append(merged)
    return sorted(output, key=lambda item: item.order)


def _page_uri_links(page: Any) -> list[tuple[tuple[float, float, float, float], str]]:
    links: list[tuple[tuple[float, float, float, float], str]] = []
    for link in page.get_links() or []:
        uri = str(link.get("uri") or "").strip()
        rect = link.get("from")
        if not uri or rect is None:
            continue
        links.append(((float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)), uri))
    return links


def _assign_line_links(lines: list[PdfLine], links: list[tuple[tuple[float, float, float, float], str]]) -> None:
    if not links:
        return
    for line in lines:
        line_box = (line.x0, line.y0, line.x1, line.y1)
        line.urls = _dedupe_texts(
            [
                *line.urls,
                *(url for link_box, url in links if _boxes_intersect(line_box, link_box)),
            ]
        )


def _boxes_intersect(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    return not (
        left[2] < right[0]
        or right[2] < left[0]
        or left[3] < right[1]
        or right[3] < left[1]
    )


def _assign_page_alignment(lines: list[PdfLine]) -> None:
    if not lines:
        return
    body_candidates = [line.x0 for line in lines if len(line.text) >= 8 and line.x0 < line.page_width * 0.45]
    left_margin = float(median(body_candidates)) if body_candidates else min(line.x0 for line in lines)

    for line in lines:
        line_width = max(1.0, line.x1 - line.x0)
        center = (line.x0 + line.x1) / 2
        left_gap = line.x0
        right_gap = line.page_width - line.x1
        center_tolerance = max(18.0, line.page_width * 0.06)
        left_tolerance = max(8.0, line.page_width * 0.025)
        right_tolerance = max(8.0, line.page_width * 0.025)

        if abs(center - line.page_width / 2) <= center_tolerance:
            line.alignment = "center"
        elif right_gap <= right_tolerance or line.x0 >= line.page_width * 0.55:
            line.alignment = "right"
        elif abs(line.x0 - left_margin) <= left_tolerance or line.x0 <= line.page_width * 0.18:
            line.alignment = "left"
        else:
            line.alignment = "other"


def _clean_pdf_lines(lines: list[PdfLine]) -> list[PdfLine]:
    without_page_numbers = [line for line in lines if not _is_page_number_line(line.text)]
    repeated_text = _repeated_standalone_texts(without_page_numbers)
    return [line for line in without_page_numbers if _normalized_text(line.text) not in repeated_text]


def _repeated_standalone_texts(lines: list[PdfLine]) -> set[str]:
    page_count = len({line.page_number for line in lines})
    if page_count < 2:
        return set()
    counts = Counter(_normalized_text(line.text) for line in lines if _normalized_text(line.text))
    threshold = max(MIN_HEADER_FOOTER_REPEATS, int(page_count * HEADER_FOOTER_REPEAT_RATIO + 0.999))
    return {
        text
        for text, count in counts.items()
        if count >= threshold and len(text) <= 80
    }


def _assign_heading_styles(lines: list[PdfLine]) -> None:
    candidates = [line for line in lines if _is_heading_candidate(line)]
    if not candidates:
        return

    supported_center_headings = _supported_center_heading_ids(lines, candidates)
    visual_candidates = [line for line in candidates if _heading_marker_level(line.text) is None]
    clusters = _heading_score_clusters(visual_candidates)
    has_center_top_heading = bool(supported_center_headings)
    for line in candidates:
        marker_level = _heading_marker_level(line.text)
        if marker_level is None:
            level = 1 if id(line) in supported_center_headings else _score_level(_heading_score(line), clusters) if clusters else 1
        else:
            level = marker_level + 1 if has_center_top_heading else marker_level
        line.style = _heading_style(level)

    _demote_unsupported_center_headings(candidates, supported_center_headings)
    _demote_consecutive_same_level_headings(lines)
    _demote_long_lowest_headings(candidates)


def _is_heading_candidate(line: PdfLine) -> bool:
    text = line.text.strip()
    if not text or len(text) > 120:
        return False
    if line.alignment == "right":
        return False
    if _looks_like_unordered_list(text) or _looks_like_date(text) or _starts_with_date(text):
        return False
    if line.alignment not in {"left", "center"} and not line.bold:
        return False
    if line.alignment == "center":
        return _text_length(text) <= 30
    if line.alignment != "left" and not line.bold:
        return False
    if line.alignment == "left" and not _looks_like_numbered_heading(text):
        return False
    return True


def _heading_score(line: PdfLine) -> float:
    score = float(line.font_size or 0)
    if line.alignment == "center":
        score += CENTER_SCORE_BOOST
    if line.bold:
        score += BOLD_SCORE_BOOST
    return score


def _is_center_bold_heading(line: PdfLine) -> bool:
    return line.alignment == "center" and line.bold


def _is_center_heading(line: PdfLine) -> bool:
    return line.alignment == "center" and _text_length(line.text) <= 30


def _supported_center_heading_ids(lines: list[PdfLine], candidates: list[PdfLine]) -> set[int]:
    median_gap = _median_positive_line_gap(lines)
    median_font = _median_font_size(lines)
    candidate_ids = {id(line) for line in candidates}
    supported: set[int] = set()
    for index, line in enumerate(lines):
        if id(line) not in candidate_ids or not _is_center_heading(line):
            continue
        if _looks_like_center_metadata(line.text):
            continue
        if line.bold:
            supported.add(id(line))
            continue
        if _next_line_starts_lower_heading(lines, index):
            supported.add(id(line))
            continue
        if _has_large_heading_spacing(line, lines, index, median_gap, median_font):
            supported.add(id(line))
            continue
    return supported


def _demote_unsupported_center_headings(lines: list[PdfLine], supported_center_headings: set[int]) -> None:
    for line in lines:
        if _heading_level(line.style) is not None and _is_center_heading(line) and id(line) not in supported_center_headings:
            line.style = BODY_STYLE


def _next_line_starts_lower_heading(lines: list[PdfLine], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    current = lines[index]
    next_line = lines[index + 1]
    if next_line.page_number != current.page_number:
        return False
    next_marker = _heading_marker_level(next_line.text)
    return next_marker is not None and next_marker >= 1


def _has_large_heading_spacing(
    line: PdfLine,
    lines: list[PdfLine],
    index: int,
    median_gap: float,
    median_font: float,
) -> bool:
    if line.font_size < median_font + LARGE_FONT_BOOST:
        return False
    if index == 0 or lines[index - 1].page_number != line.page_number:
        return True
    previous = lines[index - 1]
    gap = line.y0 - previous.y1
    return gap >= max(line.font_size, median_gap * LARGE_GAP_MULTIPLIER)


def _heading_score_clusters(lines: list[PdfLine]) -> list[float]:
    scores = sorted((_heading_score(line) for line in lines), reverse=True)
    clusters: list[list[float]] = []
    for score in scores:
        if clusters and abs(score - clusters[-1][-1]) <= HEADING_SCORE_CLUSTER_TOLERANCE:
            clusters[-1].append(score)
        else:
            clusters.append([score])
    return [sum(cluster) / len(cluster) for cluster in clusters[:MAX_HEADING_LEVEL]]


def _score_level(score: float, clusters: list[float]) -> int:
    nearest = min(range(len(clusters)), key=lambda index: abs(score - clusters[index]))
    return nearest + 1


def _heading_marker_level(text: str) -> int | None:
    stripped = text.strip()
    if re.match(r"^[一二三四五六七八九十百千万]{1,8}\s*、", stripped):
        return 1
    if re.match(r"^[（(]\s*[一二三四五六七八九十百千万]{1,8}\s*[）)]", stripped):
        return 2
    if re.match(r"^\d{1,3}(?:\.\d{1,3})+\s+", stripped):
        return 3
    if re.match(r"^\d{1,3}\s*[\.、．]", stripped):
        return 3
    if re.match(r"^[（(]\s*\d{1,3}\s*[）)]", stripped):
        return 4
    if re.match(r"^[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]", stripped):
        return 5
    return None


def _heading_style(level: int) -> str:
    normalized_level = max(1, min(level, MAX_HEADING_LEVEL))
    return HEADING_STYLE if normalized_level == 1 else f"{HEADING_STYLE} {normalized_level}"


def _heading_level(style: str) -> int | None:
    if style == HEADING_STYLE:
        return 1
    if not style.startswith(f"{HEADING_STYLE} "):
        return None
    match = re.search(r"\d+", style)
    return int(match.group(0)) if match else None


def _demote_long_lowest_headings(lines: list[PdfLine]) -> None:
    heading_levels = [
        level
        for level in (_heading_level(line.style) for line in lines)
        if level is not None
    ]
    if not heading_levels:
        return

    lowest_level = max(heading_levels)
    for line in lines:
        if _heading_level(line.style) == lowest_level and _text_length(line.text) > MAX_LOWEST_HEADING_CHARS:
            line.style = BODY_STYLE


def _demote_consecutive_same_level_headings(lines: list[PdfLine]) -> None:
    run: list[PdfLine] = []

    def flush() -> None:
        if len(run) >= 3 and _looks_like_body_list_run(run):
            for item in run:
                item.style = BODY_STYLE
        run.clear()

    for line in sorted(lines, key=lambda item: item.order):
        if _heading_level(line.style) is None:
            flush()
            continue
        if _heading_marker_level(line.text) is None:
            flush()
            continue
        if run and _heading_level(run[-1].style) != _heading_level(line.style):
            flush()
        run.append(line)
    flush()


def _looks_like_body_list_run(lines: list[PdfLine]) -> bool:
    if not lines:
        return False
    marker_levels = {_heading_marker_level(line.text) for line in lines}
    if len(marker_levels) != 1:
        return False
    return all(_text_length(line.text) > MAX_LOWEST_HEADING_CHARS for line in lines) or all(
        _ends_like_sentence(line.text) for line in lines
    )


def _build_items(lines: list[PdfLine]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line in sorted(lines, key=lambda item: item.order):
        item = {
            "type": "paragraph",
            "style": line.style,
            "text": line.text,
            "source": f"page:{line.page_number}:line:{line.order}",
        }
        items.append(item)
        items.extend(_link_items(line))
    return items


def _link_items(line: PdfLine) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    urls = [match.group(0).rstrip("。；;，,)") for match in URL_PATTERN.finditer(line.text)]
    urls.extend(line.urls)
    for url in _dedupe_texts(urls):
        items.append(
            {
                "type": "link_ref",
                "style": LINK_STYLE,
                "text": url,
                "url": url,
                "description": line.text if url not in line.text else url,
                "source": f"page:{line.page_number}:line:{line.order}:link",
            }
        )
    return items


def _dedupe_texts(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def _is_bold_font(font_name: str, flags: int) -> bool:
    normalized = font_name.lower().replace("-", "").replace("_", "").replace(" ", "")
    positive_tokens = ("bold", "black", "heavy", "semibold", "demibold", "extrabold", "ultrabold")
    negative_tokens = ("regular", "normal", "book", "light", "thin", "medium", "roman")
    if any(token in normalized for token in positive_tokens):
        return True
    if any(token in normalized for token in negative_tokens):
        return False
    return bool(flags & 16)


def _clean_line_text(text: str) -> str:
    value = str(text or "").replace("\u00a0", " ").replace("\u3000", " ")
    value = re.sub(r"[ \t]+", " ", value)
    return value.strip()


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).strip()


def _text_length(text: str) -> int:
    return len(_normalized_text(text))


def _median_font_size(lines: list[PdfLine]) -> float:
    sizes = [line.font_size for line in lines if line.font_size > 0]
    return float(median(sizes)) if sizes else 0.0


def _median_positive_line_gap(lines: list[PdfLine]) -> float:
    gaps: list[float] = []
    for previous, current in zip(lines, lines[1:]):
        if previous.page_number != current.page_number:
            continue
        gap = current.y0 - previous.y1
        if gap > 0:
            gaps.append(gap)
    return float(median(gaps)) if gaps else 0.0


def _is_page_number_line(text: str) -> bool:
    return bool(PAGE_NUMBER_PATTERN.match(str(text or "").strip()))


def _looks_like_numbered_heading(text: str) -> bool:
    return _heading_marker_level(text) is not None or bool(NUMBERED_HEADING_PATTERN.match(text.strip()))


def _looks_like_unordered_list(text: str) -> bool:
    return bool(UNORDERED_LIST_PATTERN.match(text.strip()))


def _looks_like_date(text: str) -> bool:
    return bool(DATE_LINE_PATTERN.match(text.strip()))


def _starts_with_date(text: str) -> bool:
    return bool(DATE_PREFIX_PATTERN.match(text.strip()))


def _ends_like_sentence(text: str) -> bool:
    return bool(re.search(r"[。！？.!?]\s*$", text.strip()))


def _looks_like_center_metadata(text: str) -> bool:
    return bool(CENTER_METADATA_PATTERN.match(text.strip()))


def write_items_to_txt(items: list[dict[str, Any]], source_file: str | Path) -> Path:
    source_path = Path(source_file)
    txt_dir = processing_subdir(source_path, "txt")
    txt_dir.mkdir(parents=True, exist_ok=True)

    output_path = txt_dir / f"{source_path.stem}.txt"
    output_path.write_text(format_extracted_items(items), encoding="utf-8")
    return output_path


def _log(message: str) -> None:
    print(f"[pdf_parser] {message}", flush=True)


def _log_item_summary(message: str, items: list[dict[str, Any]], started_at: float) -> None:
    counts = Counter(item.get("type", "unknown") for item in items)
    type_summary = ", ".join(f"{item_type}={count}" for item_type, count in sorted(counts.items()))
    _log(f"{message}: total={len(items)} ({type_summary or 'no items'}) ({time.perf_counter() - started_at:.2f}s)")


if __name__ == "__main__":
    target_file = Path("data") / "raw" / "迈拓思学院" / "公司相关"/"嘉兴迈拓不锈钢有限公司 规章制度.pdf"
    extracted_items = parse_pdf_document(target_file)
    txt_path = Path("data") / "processing" / target_file.stem / "txt" / f"{target_file.stem}.txt"

    print(f"File: {target_file}")
    print(f"Extracted text blocks: {len(extracted_items)}")
    print(f"Text output: {txt_path}")

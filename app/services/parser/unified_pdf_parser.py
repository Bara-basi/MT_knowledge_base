from __future__ import annotations

import base64
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from statistics import median
import time
from typing import Any, Iterable
import unicodedata
from urllib.parse import unquote, urlparse

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - optional in reduced deployments
    cv2 = None

try:
    import fitz
except ModuleNotFoundError:  # pragma: no cover - exercised only in misconfigured envs
    fitz = None

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - exercised only in misconfigured envs
    np = None

try:
    from rapidocr_onnxruntime import RapidOCR
except ModuleNotFoundError:  # pragma: no cover - OCR is validated before use
    RapidOCR = None

try:
    import jieba
except ModuleNotFoundError:  # pragma: no cover - optional text quality signal
    jieba = None

from app.db.minio import (
    RawDocumentObject,
    build_minio_uri,
    list_raw_document_objects,
    parse_raw_document_reference,
)
from app.services.data_clean import clean_items
from app.services.llm import (
    LLMAPIError,
    LLMConfigError,
    LLMClient,
    LLMTimeoutError,
    build_non_thinking_extra_body,
    get_llm_client,
)
from app.services.parser.img_parser import request_multimodal_text
from app.services.parser.paths import processing_document_dir, processing_subdir
from app.services.parser.word_parser import format_extracted_items


BODY_STYLE = "正文"
TABLE_STYLE = "表格"
TABLE_TITLE_STYLE = "表标题"
HEADING_STYLE = "标题"
LINK_STYLE = "链接"

OCR_DPI = int(os.getenv("PDF_OCR_DPI", "300"))
VISION_DPI = int(os.getenv("PDF_VISION_DPI", "160"))
TABLE_VISION_DPI = int(os.getenv("PDF_TABLE_VISION_DPI", "220"))
VISION_MAX_PIXELS = int(os.getenv("PDF_VISION_MAX_PIXELS", "3000000"))
TABLE_VISION_MAX_PIXELS = int(os.getenv("PDF_TABLE_VISION_MAX_PIXELS", "2500000"))
VISION_READ_TIMEOUT = float(os.getenv("PDF_VISION_READ_TIMEOUT", "120"))
TABLE_VISION_READ_TIMEOUT = float(os.getenv("PDF_TABLE_VISION_READ_TIMEOUT", "120"))
VISION_MIN_DRAFT_CHARS = int(os.getenv("PDF_VISION_MIN_DRAFT_CHARS", "40"))
VISION_CIRCUIT_BREAKER_TIMEOUTS = max(
    1,
    int(os.getenv("PDF_VISION_CIRCUIT_BREAKER_TIMEOUTS", "2")),
)
VISION_CIRCUIT_BREAKER_FAILURES = max(
    1,
    int(os.getenv("PDF_VISION_CIRCUIT_BREAKER_FAILURES", "2")),
)
# Retained by the legacy JSON helpers and shared by the active img_parser
# transport. In ``auto`` mode Moonshot uses temporary file references while
# other OpenAI-compatible providers use Data URLs.
VISION_IMAGE_TRANSPORT = os.getenv("PDF_VISION_IMAGE_TRANSPORT", "auto").strip().lower()
VISION_JPEG_QUALITY = int(os.getenv("PDF_VISION_JPEG_QUALITY", "85"))
VISION_FILE_TIMEOUT = float(os.getenv("PDF_VISION_FILE_TIMEOUT", "30"))
VISION_FILE_CLEANUP_TIMEOUT = float(os.getenv("PDF_VISION_FILE_CLEANUP_TIMEOUT", "5"))
FULL_PAGE_MAX_OUTPUT_TOKENS = int(os.getenv("PDF_FULL_PAGE_MAX_OUTPUT_TOKENS", "2200"))
TABLE_MAX_OUTPUT_TOKENS = int(os.getenv("PDF_TABLE_MAX_OUTPUT_TOKENS", "3000"))
MIN_NATIVE_TEXT_CHARS = int(os.getenv("PDF_MIN_NATIVE_TEXT_CHARS", "40"))
HYBRID_IMAGE_COVERAGE = float(os.getenv("PDF_HYBRID_IMAGE_COVERAGE", "0.45"))
BLANK_INK_RATIO = float(os.getenv("PDF_BLANK_INK_RATIO", "0.003"))
MODEL_MODE_ENV = "PDF_MODEL_CLEANUP_MODE"
DEFAULT_MODEL_MODE = "auto"
EXTERNAL_VISION_OPT_IN_ENV = "PDF_ALLOW_EXTERNAL_VISION"
VISION_MODEL = os.getenv("PDF_VISION_MODEL") or None
TABLE_VISION_ENABLED = os.getenv("PDF_TABLE_VISION_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MAX_HEADING_LEVEL = 6
HEADER_FOOTER_REPEAT_RATIO = 0.35
MIN_HEADER_FOOTER_REPEATS = 2
NATIVE_ROW_TOLERANCE = 3.0
OCR_ROW_TOLERANCE = 5.0

URL_PATTERN = re.compile(r"https?://[^\s<>()]+", re.IGNORECASE)
PAGE_NUMBER_PATTERN = re.compile(
    r"^\s*(?:(?:第\s*)?\d{1,4}\s*(?:页)?|[IVXLCDM]{1,8})\s*$",
    re.IGNORECASE,
)
STANDARD_CODE_PATTERN = re.compile(
    r"^(?:"
    r"(?:GB\s*/?\s*T?|SA|SB|SF)\s*[-+]?\s*\d[\w./-]*"
    r"|ASME\s+.+"
    r")$",
    re.IGNORECASE,
)
SUSPICIOUS_TEXT_PATTERN = re.compile(r"[\ufffd�]|(?:[|_~=]{3,})")
KNOWN_WATERMARK_PATTERN = re.compile(
    r"^(?:用户慎之|用[户者]慎之|仅供学习|仅供参考|严禁复制)$"
)
TABLE_TEXT_PATTERN = re.compile(
    r"(?:\S+\s{2,}){2,}\S+",
    re.IGNORECASE,
)
TABLE_TITLE_PATTERN = re.compile(
    r"^(?:"
    r"(?:表|TABLE)\s*[A-Z]?\d+(?:[.\-—]\d+)*(?:\s*\S.*)?"
    r"|[A-Z]?\d+(?:[.\-—]\d+)*\s*系列\s*(?:表|TABLE)\s*\d+"
    r")$",
    re.IGNORECASE,
)
BROKEN_LATIN_FONT_PATTERN = re.compile(r"^E-(?:HZ|BX)9-", re.IGNORECASE)
BROKEN_LATIN_GLYPH_MAP = str.maketrans(
    {
        # Confirmed against the rendered GB/T 12771-2019 pages.  These are
        # incorrect ToUnicode mappings in two embedded Latin fonts.
        "犃": "A",
        "犅": "B",
        "犆": "C",
        "犇": "D",
        "犌": "G",
        "犎": "H",
        "犐": "I",
        "犘": "p",
        "犚": "R",
        "犛": "S",
        "犜": "T",
        "犠": "W",
        "犪": "a",
        "犱": "d",
        "犲": "e",
        "犳": "f",
        "犻": "i",
        "犾": "l",
        "狀": "n",
        "狅": "o",
        "狆": "p",
        "狉": "r",
        "狊": "s",
        "狋": "t",
        "狌": "u",
    }
)
LOW_VALUE_TEXT_PATTERNS = (
    re.compile(r"^(?:www\.)?newmaker\.com/?$", re.IGNORECASE),
    re.compile(r"^(?:源自|来源于?)(?:网络|互联网)$"),
    re.compile(r"^(?:仅供|只供).*(?:学习|交流|参考).*(?:请勿|不得)?(?:商用|传播|转载)?$"),
    re.compile(r"^版权(?:专有|所有)\s*(?:不得|严禁)(?:翻印|复制|转载)$"),
    re.compile(r"^(?:扫描全能王|CS\s*CamScanner)$", re.IGNORECASE),
    re.compile(r"中国标准出版社.*(?:出版|印刷厂印刷)"),
    re.compile(r"新华书店.*(?:发行|经售)"),
    re.compile(r"^印数\s*\d+\s*[-—~至]\s*\d+"),
    re.compile(r"^标目\s*\d+\s*[-—]\s*\d+"),
    re.compile(r"^(?:开本|印张|字数)\s*[\d./×xX—\- ]+(?:(?:开本|印张|字数)\s*[\d./×xX—\- ]+)*$"),
    re.compile(r"^\d{4}\s*年.*第[一二三四五六七八九十百\d]+\s*版.*第[一二三四五六七八九十百\d]+\s*次印刷$"),
    re.compile(r"^(?:ISBN|统一书号|书号|CIP数据|定价)\s*[:：]?\s*\S+", re.IGNORECASE),
    re.compile(r"^(?:责任编辑|封面设计|责任校对|责任印制)\s*[:：]?\s*\S+"),
)


class PdfPageKind(str, Enum):
    NATIVE_TEXT = "native_text"
    SCANNED = "scanned"
    HYBRID = "hybrid"
    BLANK = "blank"


@dataclass(frozen=True)
class PdfPageProfile:
    page_number: int
    kind: PdfPageKind
    native_text_chars: int
    native_text_quality: float
    image_coverage: float
    ink_ratio: float
    image_count: int
    drawing_count: int
    width: float
    height: float
    rotation: int


@dataclass
class PdfLine:
    """Canonical line/block representation shared by native and OCR extraction."""

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
    confidence: float = 1.0
    extraction_method: str = "native_pdf"
    block_type: str = "paragraph"
    block_id: int = 0
    bold_score: float = 0.0


class PdfExtractionError(RuntimeError):
    """Raised when a nonblank PDF page cannot produce auditable text."""


@dataclass
class PdfVisionSession:
    """Document-scoped circuit breakers for independent visual workloads."""

    disabled_tasks: set[str] = field(default_factory=set)
    timeout_purposes: list[str] = field(default_factory=list)
    consecutive_timeouts: dict[str, int] = field(default_factory=dict)
    failure_purposes: list[str] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)
    consecutive_failures: dict[str, int] = field(default_factory=dict)

    def available(self, task: str) -> bool:
        return task not in self.disabled_tasks

    def mark_timeout(self, task: str, purpose: str) -> None:
        self.timeout_purposes.append(purpose)
        timeout_count = self.consecutive_timeouts.get(task, 0) + 1
        self.consecutive_timeouts[task] = timeout_count
        if timeout_count >= VISION_CIRCUIT_BREAKER_TIMEOUTS:
            self.disabled_tasks.add(task)
            _log(
                "vision circuit opened: "
                f"task={task}; reason=consecutive_timeouts; count={timeout_count}; "
                f"last_purpose={purpose}"
            )

    def mark_success(self, task: str) -> None:
        self.consecutive_timeouts[task] = 0
        self.consecutive_failures[task] = 0

    def mark_failure(
        self,
        task: str,
        purpose: str,
        reason: str,
        *,
        terminal: bool = False,
    ) -> None:
        self.failure_purposes.append(purpose)
        self.failure_reasons.append(reason)
        failure_count = self.consecutive_failures.get(task, 0) + 1
        self.consecutive_failures[task] = failure_count
        if terminal or failure_count >= VISION_CIRCUIT_BREAKER_FAILURES:
            self.disabled_tasks.add(task)
            _log(
                "vision circuit opened: "
                f"task={task}; reason={'terminal_failure' if terminal else 'consecutive_failures'}; "
                f"count={failure_count}; last_purpose={purpose}; detail={reason}"
            )


_RAPID_OCR_ENGINE: Any | None = None


def parse_unified_pdf_document(
    file_path: str | Path,
    *,
    image_analysis_workers: int = 3,
    source_reference: str | Path | None = None,
    model_cleanup_mode: str | None = None,
) -> list[dict[str, Any]]:
    """Parse every PDF through the unified native/OCR/layout pipeline."""

    del image_analysis_workers  # Reserved for bounded visual fallbacks.
    _ensure_pdf_dependencies()
    started_at = time.perf_counter()
    path = Path(file_path)
    _validate_pdf_path(path)
    mode = _normalize_model_mode(model_cleanup_mode)
    external_vision_enabled = _env_bool(EXTERNAL_VISION_OPT_IN_ENV, True)
    output_dir = processing_document_dir(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = processing_subdir(path, "img")
    image_dir.mkdir(parents=True, exist_ok=True)

    _log(
        "document start: "
        f"path={path}; cleanup_mode={mode}; ocr_dpi={OCR_DPI}; "
        f"vision_dpi={VISION_DPI}; table_vision_dpi={TABLE_VISION_DPI}; "
        f"vision_transport={VISION_IMAGE_TRANSPORT}; "
        f"external_vision_enabled={external_vision_enabled}"
    )
    profiles = classify_pdf_pages(path)
    kind_counts = Counter(profile.kind.value for profile in profiles)
    _log(
        "document classified: "
        f"pages={len(profiles)}; kinds={dict(sorted(kind_counts.items()))}; "
        f"elapsed={time.perf_counter() - started_at:.2f}s"
    )
    profile_by_page = {profile.page_number: profile for profile in profiles}
    extracted_by_page: dict[int, list[PdfLine]] = {}
    page_warnings: list[dict[str, Any]] = []
    page_metrics: list[dict[str, Any]] = []
    vision_session = PdfVisionSession()

    with fitz.open(path) as document:
        for page_index, page in enumerate(document):
            page_started_at = time.perf_counter()
            page_number = page_index + 1
            profile = profile_by_page[page_number]
            _log(
                "page start: "
                f"page={page_number}/{len(profiles)}; kind={profile.kind.value}; "
                f"native_chars={profile.native_text_chars}; "
                f"native_quality={profile.native_text_quality:.3f}; "
                f"image_coverage={profile.image_coverage:.3f}; ink_ratio={profile.ink_ratio:.3f}"
            )
            local_started_at = time.perf_counter()
            native_lines = _extract_native_lines_from_page(page, page_number=page_number)
            native_table_regions = _native_table_regions(page)
            table_regions = list(native_table_regions)
            ocr_lines: list[PdfLine] = []
            selected_lines: list[PdfLine]
            warning = ""
            local_elapsed = 0.0
            vision_attempted = False
            vision_used = False
            table_vision_attempted = False
            table_vision_used = False
            pixmap: Any | None = None

            if profile.kind == PdfPageKind.BLANK:
                selected_lines = []
                local_elapsed = time.perf_counter() - local_started_at
            elif profile.kind == PdfPageKind.NATIVE_TEXT:
                selected_lines = native_lines
                local_elapsed = time.perf_counter() - local_started_at
                _log(
                    "page native extraction complete: "
                    f"page={page_number}; native_lines={len(native_lines)}; "
                    f"table_regions={len(table_regions)}; "
                    f"elapsed={time.perf_counter() - local_started_at:.2f}s"
                )
            else:
                render_started_at = time.perf_counter()
                pixmap = _render_page(page, dpi=OCR_DPI)
                render_elapsed = time.perf_counter() - render_started_at
                table_detect_started_at = time.perf_counter()
                ocr_table_regions = _raster_table_regions(pixmap, page)
                table_detect_elapsed = time.perf_counter() - table_detect_started_at
                raw_table_regions = _merge_rectangles(
                    [*native_table_regions, *ocr_table_regions]
                )
                ocr_started_at = time.perf_counter()
                ocr_lines = _extract_ocr_lines(
                    page,
                    pixmap,
                    page_number=page_number,
                    table_regions=raw_table_regions,
                )
                ocr_elapsed = time.perf_counter() - ocr_started_at
                ocr_table_regions = _filter_raster_table_regions(
                    ocr_table_regions,
                    ocr_lines,
                )
                table_regions = _merge_rectangles(
                    [*native_table_regions, *ocr_table_regions]
                )
                for ocr_line in ocr_lines:
                    ocr_line.block_type = (
                        "table"
                        if _rect_center_in_any(_line_rect(ocr_line), table_regions)
                        else "paragraph"
                    )
                if profile.kind == PdfPageKind.HYBRID:
                    selected_lines = _merge_native_and_ocr_lines(native_lines, ocr_lines)
                else:
                    selected_lines = ocr_lines
                local_elapsed = time.perf_counter() - local_started_at
                _log(
                    "page local OCR complete: "
                    f"page={page_number}; native_lines={len(native_lines)}; "
                    f"ocr_lines={len(ocr_lines)}; table_regions={len(table_regions)}; "
                    f"render_elapsed={render_elapsed:.2f}s; "
                    f"table_detect_elapsed={table_detect_elapsed:.2f}s; "
                    f"ocr_elapsed={ocr_elapsed:.2f}s; "
                    f"elapsed={time.perf_counter() - local_started_at:.2f}s"
                )

                # A detected table page is handled by the table-only visual
                # route below. Sending it through the full-page transcriber
                # first duplicates work and encourages whole-page JSON output.
                if _should_use_full_page_vision(
                    mode,
                    profile,
                    selected_lines,
                    table_regions=table_regions,
                ) and external_vision_enabled:
                    if vision_session.available("full_page"):
                        vision_attempted = True
                        vision_pixmap = _render_model_image(
                            page,
                            dpi=VISION_DPI,
                            max_pixels=VISION_MAX_PIXELS,
                        )
                        visual_lines = _extract_page_with_vision_model(
                            page,
                            vision_pixmap,
                            page_number=page_number,
                            ocr_lines=selected_lines,
                            vision_session=vision_session,
                        )
                    else:
                        visual_lines = []
                        _log(
                            "model request skipped: "
                            f"page={page_number}; task=full_page; reason=circuit_open"
                        )
                    if visual_lines:
                        selected_lines = visual_lines
                        vision_used = True
                    else:
                        warning = (
                            "full-page visual cleanup unavailable; "
                            "kept local OCR extraction"
                        )

            if profile.kind in {PdfPageKind.NATIVE_TEXT, PdfPageKind.HYBRID}:
                selected_lines.extend(
                    _extract_figure_blocks(
                        page,
                        page_number=page_number,
                        image_dir=image_dir,
                    )
                )
            if table_regions and TABLE_VISION_ENABLED and external_vision_enabled:
                table_vision_attempted = vision_session.available("table")
                if table_vision_attempted:
                    visual_tables = _extract_tables_with_vision_model(
                        page,
                        page_number=page_number,
                        table_regions=table_regions,
                        draft_lines=selected_lines,
                        vision_session=vision_session,
                    )
                else:
                    visual_tables = []
                    _log(
                        "model request skipped: "
                        f"page={page_number}; task=table; reason=circuit_open; "
                        f"regions={len(table_regions)}"
                    )
                if visual_tables:
                    selected_lines = _replace_lines_with_visual_tables(
                        selected_lines,
                        visual_tables,
                        table_regions=table_regions,
                    )
                    table_vision_used = True
                else:
                    warning = warning or "table detected but visual table extraction returned no data"
            _assign_table_blocks(selected_lines, table_regions)
            _assign_page_alignment(selected_lines)
            extracted_by_page[page_number] = selected_lines

            if profile.kind != PdfPageKind.BLANK and not selected_lines:
                raise PdfExtractionError(
                    f"Nonblank PDF page produced no text: {path.name} page {page_number} "
                    f"(kind={profile.kind.value})"
                )

            suspicious_ratio = _suspicious_text_ratio(selected_lines)
            mean_confidence = _mean_confidence(selected_lines)
            if profile.kind in {PdfPageKind.SCANNED, PdfPageKind.HYBRID}:
                if suspicious_ratio >= 0.08:
                    warning = warning or f"high suspicious OCR text ratio: {suspicious_ratio:.3f}"
                elif mean_confidence and mean_confidence < 0.85:
                    warning = warning or f"low OCR confidence: {mean_confidence:.3f}"
            if warning:
                page_warnings.append({"page_number": page_number, "warning": warning})
            page_elapsed = time.perf_counter() - page_started_at
            page_metrics.append(
                {
                    "page_number": page_number,
                    "kind": profile.kind.value,
                    "native_line_count": len(native_lines),
                    "ocr_line_count": len(ocr_lines),
                    "selected_line_count": len(selected_lines),
                    "mean_confidence": round(mean_confidence, 4),
                    "suspicious_text_ratio": round(suspicious_ratio, 4),
                    "vision_attempted": vision_attempted,
                    "vision_used": vision_used,
                    "table_vision_attempted": table_vision_attempted,
                    "table_vision_used": table_vision_used,
                    "local_elapsed_seconds": round(local_elapsed, 3),
                    "page_elapsed_seconds": round(page_elapsed, 3),
                }
            )
            _log(
                "page complete: "
                f"page={page_number}/{len(profiles)}; selected_lines={len(selected_lines)}; "
                f"vision_attempted={vision_attempted}; vision_used={vision_used}; "
                f"table_vision_attempted={table_vision_attempted}; "
                f"table_vision_used={table_vision_used}; "
                f"mean_confidence={mean_confidence:.3f}; suspicious_ratio={suspicious_ratio:.3f}; "
                f"warning={warning or 'none'}; elapsed={page_elapsed:.2f}s"
            )

    lines = [
        line
        for page_number in sorted(extracted_by_page)
        for line in extracted_by_page[page_number]
    ]
    lines = _clean_pdf_lines(lines)
    _assign_heading_styles(lines)
    lines = _merge_wrapped_paragraphs(lines)
    lines = _consolidate_local_table_lines(lines)
    items = clean_items(_build_pdf_items(lines))
    text_path = write_items_to_txt(items, path)
    manifest_path = write_parse_manifest(
        path,
        source_reference=source_reference,
        profiles=profiles,
        page_metrics=page_metrics,
        warnings=page_warnings,
        item_count=len(items),
        text_path=text_path,
        model_cleanup_mode=mode,
        model_timeout_purposes=vision_session.timeout_purposes,
        model_failure_purposes=vision_session.failure_purposes,
        model_failure_reasons=vision_session.failure_reasons,
        disabled_model_tasks=sorted(vision_session.disabled_tasks),
    )
    _log(
        f"document complete: path={path.name}; pages={len(profiles)}; items={len(items)}; "
        f"timeouts={len(vision_session.timeout_purposes)}; "
        f"model_failures={len(vision_session.failure_purposes)}; "
        f"disabled_tasks={sorted(vision_session.disabled_tasks)}; "
        f"manifest={manifest_path}; elapsed={time.perf_counter() - started_at:.2f}s"
    )
    return items


def classify_pdf_pages(file_path: str | Path) -> list[PdfPageProfile]:
    """Classify each page instead of routing an entire document by path."""

    _ensure_pdf_dependencies()
    path = Path(file_path)
    _validate_pdf_path(path)
    profiles: list[PdfPageProfile] = []
    with fitz.open(path) as document:
        for page_index, page in enumerate(document):
            native_text = _normalized_text(_normalized_native_page_text(page))
            native_chars = len(native_text)
            native_quality = _native_text_quality(native_text)
            image_rects = _image_rectangles(page)
            image_coverage = _rectangle_coverage(image_rects, page.rect)
            drawing_count = len(page.get_drawings())
            ink_ratio = 0.0
            if native_chars < MIN_NATIVE_TEXT_CHARS:
                ink_ratio = _page_ink_ratio(page)
            kind = _classify_page_kind(
                native_chars=native_chars,
                native_quality=native_quality,
                image_coverage=image_coverage,
                ink_ratio=ink_ratio,
                image_count=len(image_rects),
                drawing_count=drawing_count,
            )
            profiles.append(
                PdfPageProfile(
                    page_number=page_index + 1,
                    kind=kind,
                    native_text_chars=native_chars,
                    native_text_quality=round(native_quality, 5),
                    image_coverage=round(image_coverage, 5),
                    ink_ratio=round(ink_ratio, 5),
                    image_count=len(image_rects),
                    drawing_count=drawing_count,
                    width=float(page.rect.width),
                    height=float(page.rect.height),
                    rotation=int(page.rotation or 0),
                )
            )
    return profiles


def _normalized_native_page_text(page: Any) -> str:
    parts: list[str] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for raw_line in block.get("lines", []):
            for span in raw_line.get("spans", []):
                text = _normalize_pdf_text(
                    str(span.get("text") or ""),
                    font_name=str(span.get("font") or ""),
                )
                if text:
                    parts.append(text)
    return "\n".join(parts)


def resolve_pdf_document_sources(
    source: str | Path,
    *,
    local_path: str | Path | None = None,
    split_if_missing: bool = True,
) -> list[str | Path]:
    """Resolve an ASME parent volume to existing split PDFs before parsing."""

    source_text = str(source)
    if not _is_asme_parent_pdf(source_text):
        return [source]

    existing = _existing_asme_split_sources(source)
    if existing:
        _log(f"using existing ASME split edition: parent={source_text}, sections={len(existing)}")
        return existing
    if not split_if_missing:
        return [source]
    if local_path is None:
        raise ValueError("local_path is required to split an ASME parent PDF")

    # Lazy import avoids a module cycle.  The splitter retains only the ASME
    # segmentation responsibility; all section text is parsed by this module.
    from app.services.parser.standard_pdf_splitter import split_standard_pdf_sections_only

    sections = split_standard_pdf_sections_only(
        local_path,
        source_reference=source,
    )
    resolved: list[str | Path] = [
        section.source_uri or section.output_path
        for section in sections
    ]
    if not resolved:
        raise PdfExtractionError(f"ASME parent PDF produced no split sections: {source}")
    return resolved


def _classify_page_kind(
    *,
    native_chars: int,
    native_quality: float,
    image_coverage: float,
    ink_ratio: float,
    image_count: int,
    drawing_count: int,
) -> PdfPageKind:
    if native_chars == 0 and ink_ratio <= BLANK_INK_RATIO:
        return PdfPageKind.BLANK
    if native_chars < MIN_NATIVE_TEXT_CHARS:
        if image_coverage >= 0.25 or ink_ratio > BLANK_INK_RATIO or drawing_count:
            return PdfPageKind.SCANNED
        return PdfPageKind.BLANK
    if native_quality < 0.80:
        return PdfPageKind.SCANNED
    if image_count and image_coverage >= HYBRID_IMAGE_COVERAGE:
        return PdfPageKind.HYBRID
    return PdfPageKind.NATIVE_TEXT


def _native_text_quality(text: str) -> float:
    """Estimate whether an embedded text map contains visually unrelated glyphs."""

    normalized = _normalized_text(text)
    if not normalized:
        return 0.0
    suspicious = sum(1 for char in normalized if char in "\ufffd�")
    cjk = [char for char in normalized if "\u4e00" <= char <= "\u9fff"]
    rare_ratio = 0.0
    if jieba is not None and len(cjk) >= 8:
        try:
            jieba.setLogLevel(40)
            jieba.initialize()
            rare_ratio = sum(1 for char in cjk if not jieba.get_FREQ(char)) / len(cjk)
        except Exception:
            rare_ratio = 0.0
    suspicious_ratio = suspicious / len(normalized)
    symbol_run_chars = sum(
        len(match.group(0))
        for match in re.finditer(r"""[!"#$%&'()*+,./:;<=>?@\[\]^_`{|}~-]{4,}""", normalized)
    )
    symbol_run_ratio = symbol_run_chars / len(normalized)
    return _clamp(
        1.0 - suspicious_ratio * 4.0 - rare_ratio * 1.2 - symbol_run_ratio * 3.0,
        0.0,
        1.0,
    )


def _extract_native_lines(path: Path) -> list[PdfLine]:
    output: list[PdfLine] = []
    with fitz.open(path) as document:
        for page_index, page in enumerate(document, start=1):
            output.extend(_extract_native_lines_from_page(page, page_number=page_index))
    return output


def _extract_native_lines_from_page(page: Any, *, page_number: int) -> list[PdfLine]:
    page_width = float(page.rect.width)
    page_height = float(page.rect.height)
    lines: list[PdfLine] = []
    order = 0
    blocks = page.get_text("dict", sort=True).get("blocks", [])
    for block_index, block in enumerate(blocks):
        if block.get("type") != 0:
            continue
        for raw_line in block.get("lines", []):
            line = _native_line_from_spans(
                raw_line.get("spans", []),
                page_number=page_number,
                page_width=page_width,
                page_height=page_height,
                order=order,
                block_id=block_index,
            )
            if line is None:
                continue
            lines.append(line)
            order += 1
    lines = _merge_same_row_lines(lines, tolerance=NATIVE_ROW_TOLERANCE)
    _assign_page_alignment(lines)
    _assign_native_links(page, lines)
    return lines


def _extract_figure_blocks(
    page: Any,
    *,
    page_number: int,
    image_dir: Path,
) -> list[PdfLine]:
    """Export meaningful embedded/raster figure blocks from text-bearing pages."""

    output: list[PdfLine] = []
    try:
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_IMAGES).get("blocks", [])
    except Exception:
        return output
    page_area = max(1.0, float(page.rect.get_area()))
    for block_index, block in enumerate(blocks):
        if int(block.get("type", 0)) != 1:
            continue
        bbox = fitz.Rect(block.get("bbox") or (0, 0, 0, 0)) & page.rect
        area_ratio = float(bbox.get_area()) / page_area
        if area_ratio < 0.012 or area_ratio > 0.60 or bbox.width < 36 or bbox.height < 36:
            continue
        image_path = image_dir / f"page_{page_number:04d}_figure_{block_index:03d}.png"
        try:
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(2.0, 2.0),
                clip=bbox,
                alpha=False,
            )
            pixmap.save(str(image_path))
        except Exception:
            continue
        output.append(
            PdfLine(
                text=str(image_path),
                page_number=page_number,
                page_width=float(page.rect.width),
                page_height=float(page.rect.height),
                x0=float(bbox.x0),
                y0=float(bbox.y0),
                x1=float(bbox.x1),
                y1=float(bbox.y1),
                font_size=0.0,
                bold=False,
                order=100_000 + block_index,
                alignment="center",
                style="图片",
                confidence=1.0,
                extraction_method="native_image",
                block_type="figure",
                block_id=block_index,
            )
        )
    return output


def _native_line_from_spans(
    spans: list[dict[str, Any]],
    *,
    page_number: int,
    page_width: float,
    page_height: float,
    order: int,
    block_id: int,
) -> PdfLine | None:
    parts: list[tuple[str, tuple[float, float, float, float]]] = []
    sizes: list[float] = []
    boxes: list[tuple[float, float, float, float]] = []
    bold = False
    for span in spans:
        text = _normalize_pdf_text(
            str(span.get("text") or ""),
            font_name=str(span.get("font") or ""),
        )
        if not text:
            continue
        bbox = tuple(float(value) for value in (span.get("bbox") or (0, 0, 0, 0)))
        parts.append((text, bbox))
        boxes.append(bbox)
        size = float(span.get("size") or 0.0)
        if size > 0:
            sizes.append(size)
        bold = bold or _is_bold_font(str(span.get("font") or ""), int(span.get("flags") or 0))
    if not parts:
        return None
    return PdfLine(
        text=_join_positioned_text(parts),
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
        block_id=block_id,
        extraction_method="native_pdf",
    )


def _extract_ocr_lines(
    page: Any,
    pixmap: Any,
    *,
    page_number: int,
    table_regions: list[Any],
) -> list[PdfLine]:
    engine = _rapid_ocr_engine()
    image = _pixmap_array(pixmap)
    results, _elapsed = engine(image)
    if not results:
        return []
    scale_x = float(page.rect.width) / float(pixmap.width)
    scale_y = float(page.rect.height) / float(pixmap.height)
    lines: list[PdfLine] = []
    density_scores: list[float] = []
    for order, result in enumerate(results):
        if not isinstance(result, (list, tuple)) or len(result) < 3:
            continue
        polygon, raw_text, raw_confidence = result[:3]
        text = _clean_line_text(str(raw_text or ""))
        if not text:
            continue
        points = [(float(point[0]), float(point[1])) for point in polygon]
        pixel_x0 = min(point[0] for point in points)
        pixel_y0 = min(point[1] for point in points)
        pixel_x1 = max(point[0] for point in points)
        pixel_y1 = max(point[1] for point in points)
        rect = fitz.Rect(
            pixel_x0 * scale_x,
            pixel_y0 * scale_y,
            pixel_x1 * scale_x,
            pixel_y1 * scale_y,
        )
        font_size = max(1.0, (pixel_y1 - pixel_y0) * 72.0 / OCR_DPI * 0.82)
        density = _ocr_bold_density(image, pixel_x0, pixel_y0, pixel_x1, pixel_y1)
        density_scores.append(density)
        lines.append(
            PdfLine(
                text=text,
                page_number=page_number,
                page_width=float(page.rect.width),
                page_height=float(page.rect.height),
                x0=float(rect.x0),
                y0=float(rect.y0),
                x1=float(rect.x1),
                y1=float(rect.y1),
                font_size=font_size,
                bold=False,
                order=order,
                confidence=float(raw_confidence or 0.0),
                extraction_method="ocr",
                block_type="table" if _rect_center_in_any(rect, table_regions) else "paragraph",
                bold_score=density,
            )
        )
    if density_scores:
        body_density = float(median(density_scores))
        for line in lines:
            line.bold = (
                line.bold_score >= max(0.12, body_density * 1.22)
                and _text_length(line.text) <= 120
            )
    _assign_page_alignment(lines)
    return _merge_same_row_lines(lines, tolerance=OCR_ROW_TOLERANCE)


def _merge_native_and_ocr_lines(
    native_lines: list[PdfLine],
    ocr_lines: list[PdfLine],
) -> list[PdfLine]:
    output = list(native_lines)
    for ocr_line in ocr_lines:
        if any(
            _intersection_over_smaller(_line_rect(ocr_line), _line_rect(native_line)) >= 0.55
            or (
                _normalized_text(ocr_line.text) == _normalized_text(native_line.text)
                and _normalized_text(ocr_line.text)
            )
            for native_line in native_lines
        ):
            continue
        output.append(ocr_line)
    output.sort(key=lambda line: (line.page_number, line.y0, line.x0, line.order))
    for order, line in enumerate(output):
        line.order = order
    return output


def _legacy_extract_page_with_vision_json(
    page: Any,
    pixmap: Any,
    *,
    page_number: int,
    ocr_lines: list[PdfLine],
    llm_client: LLMClient | None = None,
    vision_session: PdfVisionSession | None = None,
) -> list[PdfLine]:
    if vision_session is not None and not vision_session.available("full_page"):
        _log(
            f"vision fallback skipped on page {page_number}: "
            "full-page vision circuit is open after an earlier timeout"
        )
        return []
    if llm_client is None and not _env_bool(EXTERNAL_VISION_OPT_IN_ENV, True):
        _log(
            f"vision fallback skipped on page {page_number}: "
            f"set {EXTERNAL_VISION_OPT_IN_ENV}=true only after external-data approval"
        )
        return []
    client = llm_client
    if client is None:
        try:
            client = get_llm_client()
        except LLMConfigError:
            return []
    ocr_draft = "\n".join(f"{index + 1}. {line.text}" for index, line in enumerate(ocr_lines))
    prompt = f"""
你是企业知识库的 PDF 页面精确转录与结构清洗器。请直接阅读图片，并参考 OCR 草稿纠错。

要求：
1. 只输出图片中确实存在的文字，不补写、不总结、不改写。
2. 删除纯页码、重复页眉页脚，以及下列能够明确确认无问答价值的污染：
   - “源自网络/互联网”、下载站名或网址、扫描软件水印、“仅供学习/交流/参考、请勿商用”等来源声明；
   - 出版社、印刷厂、发行所、经售书店及其地址组成的出版说明；
   - “版权专有不得翻印”等版权页套话；
   - 开本、印张、字数、版次、印次、印数、标目、书号、ISBN、CIP、定价、责任编辑、封面设计、校对、印制等书籍印制元数据；
   - 位于上述出版信息之间、脱离上下文且无独立含义的“普”“精装”等装帧/印刷等级短字；
   - OCR 产生的孤立乱码、裁切标记和无内容装饰符号。
   不得删除文档标题、标准编号、发布日期/实施日期、发布/归口/起草单位、起草人、修订记录、正文网址、公式符号、表格单元格、图表标题、注释和脚注。
3. 允许按视觉版面重新拆行和合并行：混在一行的正文与表格要拆开，被 OCR 打散的同一段或同一表格要合并。
4. 只纠正图片能够明确证明的 OCR 错字、异常生僻字、错序和断词。原文本身的语法或事实错误不得修改。
5. 判断每一项是 paragraph、table_title 还是 table。表名/表题必须单独标为 table_title，不得混入数据行。
6. 如果表格仍以散列文字存在，合并成一个 table 项；其 text 必须是 JSON 对象数组，例如
   [{{"列1":"值1","列2":"值2"}},{{"列1":"值3","列2":"值4"}}]。
   不要把表头自身重复作为数据行。
7. 标签必须遵守下游协议：
   - 文档主标题用 paragraph+标题；章节按真实层级使用 paragraph+标题 2 至标题 6；普通文字用 paragraph+正文；
   - 同一个逻辑标题即使在图片中视觉换行，也必须合并成一个 paragraph 项，text 中不得换行；不得连续输出多个同级标题片段；
   - 表题必须用 table_title+表标题，表格数据必须用 table+表格；
   - “注/NOTE”、脚注、附加说明属于 paragraph+正文，不得标成表格；
   - 图中尺寸值、坐标轴、序号、孤立数值范围/单位、出版地点不得判为标题；
   - 不因居中、粗体或字号较大就改变语义标签，标题必须确实是文档结构节点。
   style 只能是 正文、标题、标题 2、标题 3、标题 4、标题 5、标题 6、表标题、表格。
8. bbox 使用 [x0,y0,x1,y1]，坐标范围统一为 0~1000；无法准确判断时给出近似区域。
9. bold 表示视觉上是否为粗体；font_scale 表示相对正文字号，正文约为 1.0。

OCR 草稿：
{ocr_draft or "无"}

只返回 JSON：
{{
  "lines": [
    {{
      "text": "原文",
      "type": "paragraph|table_title|table",
      "style": "正文|标题|标题 2|标题 3|标题 4|标题 5|标题 6|表标题|表格",
      "bbox": [0, 0, 1000, 1000],
      "bold": false,
      "font_scale": 1.0,
      "confidence": 0.0
    }}
  ]
}}
""".strip()
    uploaded_file_id: str | None = None
    try:
        content, uploaded_file_id = _vision_content(
            prompt,
            pixmap,
            client=client,
            purpose=f"page {page_number} full-page vision",
            draft_chars=len(ocr_draft),
            max_tokens=FULL_PAGE_MAX_OUTPUT_TOKENS,
            reference_rect=page.rect,
        )
        payload = _chat_json_object(
            client,
            [{"role": "user", "content": content}],
            max_tokens=FULL_PAGE_MAX_OUTPUT_TOKENS,
            purpose=f"page {page_number} full-page vision",
            task="full_page",
            read_timeout=VISION_READ_TIMEOUT,
            vision_session=vision_session,
        )
    except (LLMAPIError, LLMConfigError, ValueError, OSError) as exc:
        _log(f"vision fallback failed on page {page_number}: {type(exc).__name__}: {exc}")
        return []
    finally:
        _delete_uploaded_vision_file(client, uploaded_file_id)
    output: list[PdfLine] = []
    for order, item in enumerate(payload.get("lines") or []):
        if not isinstance(item, dict):
            continue
        raw_text: Any = item.get("text")
        if (
            str(item.get("type") or "").lower() == "table"
            and isinstance(item.get("rows"), list)
        ):
            raw_text = json.dumps(
                _coerce_model_table_rows(item),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        text = _clean_line_text(str(raw_text or ""))
        if not text:
            continue
        bbox = item.get("bbox") or [0, 0, 1000, 1000]
        if not isinstance(bbox, list) or len(bbox) != 4:
            bbox = [0, 0, 1000, 1000]
        x0, y0, x1, y1 = [_clamp(float(value), 0.0, 1000.0) for value in bbox]
        rect = fitz.Rect(
            x0 / 1000.0 * page.rect.width,
            y0 / 1000.0 * page.rect.height,
            x1 / 1000.0 * page.rect.width,
            y1 / 1000.0 * page.rect.height,
        )
        style = _normalize_model_style(str(item.get("style") or BODY_STYLE))
        model_type = str(item.get("type") or "").lower()
        if model_type == "table_title" or style == TABLE_TITLE_STYLE:
            block_type = "table_title"
            style = TABLE_TITLE_STYLE
        elif model_type == "table" or style == TABLE_STYLE:
            block_type = "table"
            style = TABLE_STYLE
        else:
            block_type = "paragraph"
        font_scale = _safe_float(item.get("font_scale"), 1.0)
        output.append(
            PdfLine(
                text=text,
                page_number=page_number,
                page_width=float(page.rect.width),
                page_height=float(page.rect.height),
                x0=float(rect.x0),
                y0=float(rect.y0),
                x1=float(rect.x1),
                y1=float(rect.y1),
                font_size=max(1.0, 10.0 * font_scale),
                bold=bool(item.get("bold")),
                order=order,
                alignment="left",
                style=style,
                confidence=_safe_float(item.get("confidence"), 0.75),
                extraction_method="vision_model",
                block_type=block_type,
            )
        )
    _assign_page_alignment(output)
    return output


def _legacy_extract_tables_with_vision_json(
    page: Any,
    *,
    page_number: int,
    table_regions: list[Any],
    draft_lines: list[PdfLine],
    llm_client: LLMClient | None = None,
    vision_session: PdfVisionSession | None = None,
) -> list[PdfLine]:
    """Extract detected table regions as row dictionaries using page vision."""

    if llm_client is None and not _env_bool(EXTERNAL_VISION_OPT_IN_ENV, True):
        _log(
            f"table vision skipped on page {page_number}: "
            f"{EXTERNAL_VISION_OPT_IN_ENV}=false"
        )
        return []
    client = llm_client
    if client is None:
        try:
            client = get_llm_client()
        except LLMConfigError:
            return []
    output: list[PdfLine] = []
    for region_index, region in enumerate(table_regions, start=1):
        if vision_session is not None and not vision_session.available("table"):
            _log(
                f"table vision skipped on page {page_number} region {region_index}: "
                "table vision circuit is open after an earlier timeout"
            )
            break
        clip = _expanded_table_clip(region, page.rect)
        table_pixmap = _render_model_image(
            page,
            dpi=TABLE_VISION_DPI,
            max_pixels=TABLE_VISION_MAX_PIXELS,
            clip=clip,
        )
        region_lines = [
            line
            for line in draft_lines
            if line.block_type != "figure"
            and _rect_center_in_any(_line_rect(line), [region])
        ]
        drafts = [
            {
                "text": line.text,
                "bbox": _rect_to_normalized_bbox(_line_rect(line), clip),
            }
            for line in region_lines
        ]
        prompt = _legacy_table_vision_json_prompt(drafts)
        uploaded_file_id: str | None = None
        try:
            content, uploaded_file_id = _vision_content(
                prompt,
                table_pixmap,
                client=client,
                purpose=f"page {page_number} table region {region_index}",
                draft_chars=sum(len(line.text) for line in region_lines),
                max_tokens=TABLE_MAX_OUTPUT_TOKENS,
                reference_rect=clip,
            )
            payload = _chat_json_object(
                client,
                [{"role": "user", "content": content}],
                max_tokens=TABLE_MAX_OUTPUT_TOKENS,
                purpose=f"page {page_number} table region {region_index}",
                task="table",
                read_timeout=TABLE_VISION_READ_TIMEOUT,
                vision_session=vision_session,
            )
        except LLMTimeoutError as exc:
            _log(
                f"table vision timed out on page {page_number} region {region_index}; "
                f"skipping remaining table regions on this page: {exc}"
            )
            break
        except (LLMAPIError, LLMConfigError, ValueError, OSError) as exc:
            _log(
                f"table vision failed on page {page_number} region {region_index}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        finally:
            _delete_uploaded_vision_file(client, uploaded_file_id)

        for table in payload.get("tables") or []:
            if not isinstance(table, dict):
                continue
            rows = _coerce_model_table_rows(table)
            if not rows:
                continue
            rect = _model_bbox_to_rect(table.get("bbox"), clip, fallback=region)
            output.append(
                PdfLine(
                    text=json.dumps(rows, ensure_ascii=False, separators=(",", ":")),
                    page_number=page_number,
                    page_width=float(page.rect.width),
                    page_height=float(page.rect.height),
                    x0=float(rect.x0),
                    y0=float(rect.y0),
                    x1=float(rect.x1),
                    y1=float(rect.y1),
                    font_size=10.0,
                    bold=False,
                    order=len(output),
                    alignment="left",
                    style=TABLE_STYLE,
                    confidence=1.0,
                    extraction_method="vision_table",
                    block_type="table",
                )
            )
    return output


def _legacy_table_vision_json_prompt(drafts: list[dict[str, Any]]) -> str:
    return f"""
你是 PDF 复杂表格精确转录器。输入图片已经裁剪为单个表格区域，不是 PDF 全页。

只解析图片中的表格，不返回图片外的正文、标题、页眉页脚、水印或工程示意图。
要求：
1. 支持跨页表格片段、合并单元格、竖排/旋转文字和无边框单元格。
2. rows 必须是 JSON 对象数组；每一行对象重复使用相同字段名。
3. 合并单元格的值向下填充到适用数据行。
4. 不要把表头自身再次作为数据行；不要重复完全相同的数据行。
5. 无法确认的单元格使用空字符串，不推测。
6. bbox 使用当前裁剪图片的 0~1000 坐标。

区域内 OCR 草稿，仅用于辅助，不得照抄乱码：
{json.dumps(drafts, ensure_ascii=False)}

只返回 JSON：
{{
  "tables": [
    {{
      "bbox": [0, 0, 1000, 1000],
      "title": "表名，可为空",
      "headers": ["列1", "列2"],
      "rows": [{{"列1": "值", "列2": "值"}}]
    }}
  ]
}}
""".strip()


def _extract_page_with_vision_model(
    page: Any,
    pixmap: Any,
    *,
    page_number: int,
    ocr_lines: list[PdfLine],
    llm_client: LLMClient | None = None,
    vision_session: PdfVisionSession | None = None,
) -> list[PdfLine]:
    """Correct OCR text while retaining all locally measured layout fields."""

    if vision_session is not None and not vision_session.available("full_page"):
        raise PdfExtractionError(
            f"Required vision line correction is unavailable on page {page_number}: "
            "full-page vision circuit is open"
        )
    if llm_client is None and not _env_bool(EXTERNAL_VISION_OPT_IN_ENV, True):
        return []
    client = llm_client
    if client is None:
        try:
            client = get_llm_client()
        except LLMConfigError as exc:
            raise PdfExtractionError(
                f"Required vision line correction is not configured on page "
                f"{page_number}: {exc}"
            ) from exc

    prompt = _page_line_correction_prompt(ocr_lines)
    try:
        response = _request_pdf_vision_text(
            client,
            prompt,
            pixmap,
            purpose=f"page {page_number} line correction",
            draft_chars=sum(len(line.text) for line in ocr_lines),
            max_tokens=FULL_PAGE_MAX_OUTPUT_TOKENS,
            reference_rect=page.rect,
            task="full_page",
            read_timeout=VISION_READ_TIMEOUT,
            vision_session=vision_session,
        )
        output = _apply_vision_line_corrections(response, ocr_lines)
        if ocr_lines and not output:
            raise ValueError("line correction removed every OCR line")
        if vision_session is not None:
            vision_session.mark_success("full_page")
        return output
    except (LLMAPIError, LLMConfigError, ValueError, OSError) as exc:
        if vision_session is not None and not isinstance(exc, LLMTimeoutError):
            vision_session.mark_failure(
                "full_page",
                f"page {page_number} line correction",
                f"{type(exc).__name__}: {exc}",
                terminal=_is_output_budget_exhausted(exc),
            )
        _log(
            f"vision line correction failed on page {page_number}; "
            f"aborting document: {type(exc).__name__}: {exc}"
        )
        raise PdfExtractionError(
            f"Required vision line correction failed on page {page_number}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _page_line_correction_prompt(lines: list[PdfLine]) -> str:
    draft = "\n".join(
        f"L{index}\t{line.text}"
        for index, line in enumerate(lines, start=1)
    )
    return f"""
你是企业知识库 PDF 的最终逐行判定与 OCR 校对器。图片是完整页面，下方是带固定行号的 OCR 草稿。
只根据图片纠正明确的 OCR 错字、断词和异常生僻字；不润色、不改写、不总结、不补充。
保留每个行号。仅在能够明确确认对检索问答无价值时将文本写为 [DROP]。

必须删除的明确污染源：
1. 纯页码、重复页眉页脚、下载站/扫描软件水印、裁切标记、孤立乱码和无内容装饰符号。
2. “源自网络/互联网”、来源网站、下载网址、“仅供学习/交流/参考、请勿商用”等传播声明。
3. 出版社、印刷厂、发行所、经售书店及地址组成的出版说明；“版权专有不得翻印”等版权页套话。
4. 开本、印张、字数、版次、印次、印数、标目、书号、ISBN、CIP、定价、责任编辑、封面设计、校对、印制等印刷元数据。
5. 夹在上述出版信息中的“普”“精装”等孤立装帧/印刷等级短字。

必须保留的重要内容：
文档标题、标准编号、发布/实施/修订日期、发布/提出/归口/起草单位、起草人、修订记录、范围、术语、要求、公式及变量、正文网址、表格单元格、图表标题、注释、脚注和附加说明。
不能确认是否有价值时一律保留。

不要输出 JSON、Markdown、代码块、解释、标题、坐标或样式。
每行严格使用：L行号<TAB>校对后的原文
不能确认时原样返回，不得创造草稿中不存在的行号。

OCR 草稿：
{draft}
""".strip()


def _apply_vision_line_corrections(
    response: str,
    source_lines: list[PdfLine],
) -> list[PdfLine]:
    corrections: dict[int, str | None] = {}
    cleaned = _strip_model_text_artifacts(response)
    record_pattern = re.compile(
        r"^\s*L?(\d+)\s*(?:\t+|[|:：]\s*|\.\s+)(.*)$"
    )
    for raw_line in cleaned.splitlines():
        match = record_pattern.match(raw_line)
        if not match:
            continue
        index = int(match.group(1))
        if not 1 <= index <= len(source_lines):
            continue
        text = _clean_line_text(match.group(2))
        if text.upper() in {"[DROP]", "DROP", "[DELETE]"}:
            corrections[index] = None
            continue
        original = source_lines[index - 1].text
        if not text or len(text) > max(80, len(original) * 4):
            continue
        corrections[index] = text

    if not corrections:
        raise ValueError("model returned no recognized OCR line records")

    output: list[PdfLine] = []
    for index, line in enumerate(source_lines, start=1):
        if index not in corrections:
            output.append(line)
            continue
        corrected = corrections[index]
        if corrected is None:
            continue
        output.append(
            replace(
                line,
                text=corrected,
                extraction_method="vision_line_correction",
                confidence=max(line.confidence, 0.9),
            )
        )
    return output


def _extract_tables_with_vision_model(
    page: Any,
    *,
    page_number: int,
    table_regions: list[Any],
    draft_lines: list[PdfLine],
    llm_client: LLMClient | None = None,
    vision_session: PdfVisionSession | None = None,
) -> list[PdfLine]:
    """Transcribe each crop as a simple tabular protocol, then build JSON locally."""

    if llm_client is None and not _env_bool(EXTERNAL_VISION_OPT_IN_ENV, True):
        return []
    client = llm_client
    if client is None:
        try:
            client = get_llm_client()
        except LLMConfigError:
            return []

    output: list[PdfLine] = []
    for region_index, region in enumerate(table_regions, start=1):
        if vision_session is not None and not vision_session.available("table"):
            break
        clip = _expanded_table_clip(region, page.rect)
        pixmap = _render_model_image(
            page,
            dpi=TABLE_VISION_DPI,
            max_pixels=TABLE_VISION_MAX_PIXELS,
            clip=clip,
        )
        region_lines = [
            line
            for line in draft_lines
            if line.block_type != "figure"
            and _rect_center_in_any(_line_rect(line), [region])
        ]
        prompt = _table_text_protocol_prompt(region_lines)
        try:
            response = _request_pdf_vision_text(
                client,
                prompt,
                pixmap,
                purpose=f"page {page_number} table region {region_index}",
                draft_chars=sum(len(line.text) for line in region_lines),
                max_tokens=TABLE_MAX_OUTPUT_TOKENS,
                reference_rect=clip,
                task="table",
                read_timeout=TABLE_VISION_READ_TIMEOUT,
                vision_session=vision_session,
            )
            table_title, rows = _parse_table_text_protocol_payload(response)
            if vision_session is not None:
                vision_session.mark_success("table")
        except LLMTimeoutError as exc:
            _log(
                f"table vision timed out on page {page_number} region {region_index}; "
                f"kept local table OCR: {exc}"
            )
            break
        except (LLMAPIError, LLMConfigError, ValueError, OSError) as exc:
            if vision_session is not None:
                vision_session.mark_failure(
                    "table",
                    f"page {page_number} table region {region_index}",
                    f"{type(exc).__name__}: {exc}",
                    terminal=_is_output_budget_exhausted(exc),
                )
            _log(
                f"table vision failed on page {page_number} region {region_index}; "
                f"kept local table OCR: {type(exc).__name__}: {exc}"
            )
            continue

        rect = fitz.Rect(region)
        if table_title and not _has_nearby_table_title(
            draft_lines,
            table_title=table_title,
            table_rect=rect,
        ):
            output.append(
                PdfLine(
                    text=table_title,
                    page_number=page_number,
                    page_width=float(page.rect.width),
                    page_height=float(page.rect.height),
                    x0=float(rect.x0),
                    y0=max(0.0, float(rect.y0) - 14.0),
                    x1=float(rect.x1),
                    y1=max(1.0, float(rect.y0) - 2.0),
                    font_size=10.0,
                    bold=True,
                    order=len(output),
                    alignment="center",
                    style=TABLE_TITLE_STYLE,
                    confidence=1.0,
                    extraction_method="vision_table_title",
                    block_type="table_title",
                    block_id=region_index,
                )
            )
        output.append(
            PdfLine(
                text=json.dumps(rows, ensure_ascii=False, separators=(",", ":")),
                page_number=page_number,
                page_width=float(page.rect.width),
                page_height=float(page.rect.height),
                x0=float(rect.x0),
                y0=float(rect.y0),
                x1=float(rect.x1),
                y1=float(rect.y1),
                font_size=10.0,
                bold=False,
                order=len(output),
                alignment="left",
                style=TABLE_STYLE,
                confidence=1.0,
                extraction_method="vision_table_tsv",
                block_type="table",
                block_id=region_index,
            )
        )
    return output


def _table_text_protocol_prompt(lines: list[PdfLine]) -> str:
    draft = "\n".join(line.text for line in lines)
    return f"""
你是复杂 PDF 表格逐字转录器。图片只包含一个表格区域。
不要输出 JSON、Markdown 表格、代码块、解释、坐标或正文。
输出纯文本记录，字段之间必须使用 TAB：
TITLE<TAB>表名
HEADER<TAB>列名1<TAB>列名2
ROW<TAB>值1<TAB>值2
每个数据行一条 ROW。合并单元格的值向下重复；空单元格保留为空字段。
不要把表头再次作为数据行。无法确认的字符保持空白，不要猜测。

区域 OCR 草稿，仅供辅助：
{draft or "[无可用 OCR 草稿]"}
""".strip()


def _parse_table_text_protocol(response: str) -> list[dict[str, str]]:
    _, rows = _parse_table_text_protocol_payload(response)
    return rows


def _parse_table_text_protocol_payload(
    response: str,
) -> tuple[str, list[dict[str, str]]]:
    title = ""
    headers: list[str] = []
    raw_rows: list[list[str]] = []
    markdown_rows: list[list[str]] = []
    for raw_line in _strip_model_text_artifacts(response).splitlines():
        cells = _split_table_protocol_record(raw_line)
        if not cells:
            continue
        record_type = cells[0].strip().upper().rstrip(":：")
        if record_type == "TITLE" and len(cells) > 1:
            title = _clean_line_text(" ".join(cells[1:]))
        elif record_type == "HEADER" and len(cells) > 1:
            headers = _unique_table_headers(cells[1:])
        elif record_type == "ROW" and len(cells) > 1:
            raw_rows.append(cells[1:])
        elif raw_line.strip().startswith("|") and len(cells) > 1:
            markdown_rows.append(cells)
    if not headers and markdown_rows:
        headers = _unique_table_headers(markdown_rows[0])
        raw_rows.extend(
            row
            for row in markdown_rows[1:]
            if not all(re.fullmatch(r":?-{2,}:?", cell) for cell in row)
        )
    if not headers:
        raise ValueError("model returned no HEADER record")

    output: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for values in raw_rows:
        normalized = (values + [""] * len(headers))[: len(headers)]
        signature = tuple(normalized)
        if signature in seen or _is_repeated_table_header(headers, normalized):
            continue
        if not any(normalized):
            continue
        seen.add(signature)
        output.append(dict(zip(headers, normalized)))
    if not output:
        raise ValueError("model returned no usable ROW records")
    return title, output


def _has_nearby_table_title(
    lines: list[PdfLine],
    *,
    table_title: str,
    table_rect: Any,
) -> bool:
    normalized_title = _normalized_text(table_title).casefold()
    for line in lines:
        if not _is_table_title_text(line.text):
            continue
        line_rect = _line_rect(line)
        close_to_table = (
            line_rect.y1 <= table_rect.y1
            and line_rect.y0 >= table_rect.y0 - max(80.0, table_rect.height * 0.25)
        )
        if close_to_table or _normalized_text(line.text).casefold() == normalized_title:
            return True
    return False


def _split_table_protocol_record(raw_line: str) -> list[str]:
    stripped = raw_line.strip()
    if not stripped:
        return []
    if "\t" in raw_line:
        return [cell.strip() for cell in raw_line.split("\t")]
    if "|" in stripped:
        return [cell.strip() for cell in stripped.strip("|").split("|")]
    match = re.match(r"^(TITLE|HEADER|ROW)\s*[:：]\s*(.*)$", stripped, re.IGNORECASE)
    if match:
        values = [
            cell.strip()
            for cell in re.split(r"\s{2,}", match.group(2))
        ]
        return [match.group(1), *values]
    return [stripped]


def _unique_table_headers(values: list[str]) -> list[str]:
    output: list[str] = []
    counts: Counter[str] = Counter()
    for index, value in enumerate(values, start=1):
        base = _clean_line_text(value) or f"列{index}"
        counts[base] += 1
        output.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return output


def _is_repeated_table_header(headers: list[str], values: list[str]) -> bool:
    return all(
        _normalized_text(re.sub(r"_\d+$", "", header)).casefold()
        == _normalized_text(value).casefold()
        for header, value in zip(headers, values)
    )


def _strip_model_text_artifacts(text: str) -> str:
    value = re.sub(r"(?is)<think>.*?</think>", "", str(text or ""))
    value = re.sub(r"(?im)^\s*```(?:text|tsv|csv)?\s*$", "", value)
    value = re.sub(r"(?im)^\s*```\s*$", "", value)
    return value.strip()


def _request_pdf_vision_text(
    client: LLMClient,
    prompt: str,
    pixmap: Any,
    *,
    purpose: str,
    draft_chars: int,
    max_tokens: int,
    reference_rect: Any,
    task: str,
    read_timeout: float,
    vision_session: PdfVisionSession | None,
) -> str:
    jpeg_bytes = pixmap.tobytes(
        "jpeg",
        jpg_quality=max(40, min(VISION_JPEG_QUALITY, 95)),
    )
    effective_dpi = round(
        pixmap.width / max(1.0, float(reference_rect.width)) * 72.0,
        1,
    )
    _log(
        "model input prepared: "
        f"step={purpose}; requested_transport={VISION_IMAGE_TRANSPORT}; "
        f"stream=false; format=text; dpi={effective_dpi}; "
        f"pixels={pixmap.width}x{pixmap.height}; "
        f"megapixels={pixmap.width * pixmap.height / 1_000_000:.2f}; "
        f"jpeg_kb={len(jpeg_bytes) / 1024:.1f}; "
        f"prompt_chars={len(prompt)}; draft_chars={draft_chars}; "
        f"max_output_tokens={max_tokens}; read_timeout={read_timeout}"
    )
    try:
        response = request_multimodal_text(
            client,
            prompt=prompt,
            image_bytes=jpeg_bytes,
            content_type="image/jpeg",
            model=VISION_MODEL,
            max_tokens=max_tokens,
            read_timeout=read_timeout,
            json_mode=False,
            purpose=purpose,
            image_transport=VISION_IMAGE_TRANSPORT,
        )
    except LLMTimeoutError:
        if vision_session is not None:
            vision_session.mark_timeout(task, purpose)
        raise
    if not str(response or "").strip():
        raise ValueError("multimodal response is empty")
    return response


def _expanded_table_clip(region: Any, page_rect: Any) -> Any:
    source = fitz.Rect(region)
    padding_x = max(4.0, source.width * 0.015)
    # Include a nearby table caption and unit line without sending the page.
    padding_top = max(12.0, source.height * 0.10)
    padding_bottom = max(5.0, source.height * 0.025)
    clip = fitz.Rect(
        source.x0 - padding_x,
        source.y0 - padding_top,
        source.x1 + padding_x,
        source.y1 + padding_bottom,
    )
    clip.intersect(page_rect)
    return clip


def _coerce_model_table_rows(table: dict[str, Any]) -> list[dict[str, str]]:
    raw_rows = table.get("rows")
    if not isinstance(raw_rows, list):
        raw_rows = table.get("data")
    if not isinstance(raw_rows, list):
        return []
    headers = [
        _clean_line_text(str(header))
        for header in (table.get("headers") or [])
        if _clean_line_text(str(header))
    ]
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw_row in raw_rows:
        if isinstance(raw_row, dict):
            row = {
                _clean_line_text(str(key)): _clean_line_text(str(value or ""))
                for key, value in raw_row.items()
                if _clean_line_text(str(key))
            }
        elif isinstance(raw_row, list) and headers:
            row = {
                header: _clean_line_text(str(raw_row[index] or ""))
                if index < len(raw_row)
                else ""
                for index, header in enumerate(headers)
            }
        else:
            continue
        if not row or not any(value for value in row.values()):
            continue
        if _is_model_header_row(row):
            continue
        signature = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if signature in seen:
            continue
        seen.add(signature)
        rows.append(row)
    return rows


def _is_model_header_row(row: dict[str, str]) -> bool:
    populated = [(key, value) for key, value in row.items() if _normalized_text(value)]
    if not populated:
        return False
    matches = sum(
        _normalized_text(key).casefold() == _normalized_text(value).casefold()
        for key, value in populated
    )
    return matches / len(populated) >= 0.8


def _replace_lines_with_visual_tables(
    lines: list[PdfLine],
    visual_tables: list[PdfLine],
    *,
    table_regions: list[Any],
) -> list[PdfLine]:
    kept: list[PdfLine] = []
    for line in lines:
        if line.block_type == "figure":
            kept.append(line)
            continue
        rect = _line_rect(line)
        if line.block_type == "table" or _rect_center_in_any(rect, table_regions):
            continue
        kept.append(line)
    output = [*kept, *visual_tables]
    output.sort(key=lambda line: (line.y0, line.x0, line.order))
    for order, line in enumerate(output):
        line.order = order
    return output


def _rect_to_normalized_bbox(rect: Any, page_rect: Any) -> list[int]:
    return [
        round(_clamp((float(rect.x0) - page_rect.x0) / max(1.0, float(page_rect.width)) * 1000, 0, 1000)),
        round(_clamp((float(rect.y0) - page_rect.y0) / max(1.0, float(page_rect.height)) * 1000, 0, 1000)),
        round(_clamp((float(rect.x1) - page_rect.x0) / max(1.0, float(page_rect.width)) * 1000, 0, 1000)),
        round(_clamp((float(rect.y1) - page_rect.y0) / max(1.0, float(page_rect.height)) * 1000, 0, 1000)),
    ]


def _model_bbox_to_rect(value: Any, page_rect: Any, *, fallback: Any) -> Any:
    if not isinstance(value, list) or len(value) != 4:
        return fitz.Rect(fallback)
    try:
        x0, y0, x1, y1 = [_clamp(float(item), 0, 1000) for item in value]
    except (TypeError, ValueError):
        return fitz.Rect(fallback)
    rect = fitz.Rect(
        page_rect.x0 + x0 / 1000 * page_rect.width,
        page_rect.y0 + y0 / 1000 * page_rect.height,
        page_rect.x0 + x1 / 1000 * page_rect.width,
        page_rect.y0 + y1 / 1000 * page_rect.height,
    )
    return rect if not rect.is_empty else fitz.Rect(fallback)


def _clean_pdf_lines(lines: list[PdfLine]) -> list[PdfLine]:
    cleaned: list[PdfLine] = []
    margin_candidates: list[PdfLine] = []
    colophon_pages = {
        line.page_number
        for line in lines
        if re.search(
            r"出版社|印刷厂|新华书店|开本|印张|字数|第.+版.+印刷|"
            r"印数|标目|ISBN|CIP|定价",
            _clean_line_text(line.text),
            re.IGNORECASE,
        )
    }
    for line in lines:
        line.text = _clean_line_text(line.text)
        if (
            line.page_number in colophon_pages
            and line.text in {"普", "精装", "平装"}
        ):
            continue
        if not line.text or _is_obvious_garbage(line.text):
            continue
        if _is_margin_line(line):
            margin_candidates.append(line)
        cleaned.append(line)

    page_count = len({line.page_number for line in cleaned})
    threshold = max(
        MIN_HEADER_FOOTER_REPEATS,
        math.ceil(page_count * HEADER_FOOTER_REPEAT_RATIO),
    )
    margin_counts = Counter(
        _normalized_text(line.text)
        for line in margin_candidates
        if _normalized_text(line.text)
    )
    repeated_margin_text = {
        text
        for text, count in margin_counts.items()
        if count >= threshold and len(text) <= 100
    }
    output = []
    for line in cleaned:
        normalized = _normalized_text(line.text)
        if normalized in repeated_margin_text and _is_margin_line(line):
            continue
        if PAGE_NUMBER_PATTERN.match(line.text) and _is_margin_line(line):
            continue
        output.append(line)
    output.sort(key=lambda line: (line.page_number, line.y0, line.x0, line.order))
    for order, line in enumerate(output):
        line.order = order
    return output


def _assign_heading_styles(lines: list[PdfLine]) -> None:
    if not lines:
        return
    body_sizes = [
        line.font_size
        for line in lines
        if line.block_type == "paragraph"
        and not line.bold
        and 12 <= _text_length(line.text) <= 240
        and line.font_size > 0
    ]
    body_size = float(median(body_sizes)) if body_sizes else float(
        median([line.font_size for line in lines if line.font_size > 0])
    )
    has_document_title = any(
        line.style == HEADING_STYLE
        or (
            line.alignment == "center"
            and line.font_size >= body_size * 1.35
            and _text_length(line.text) <= 80
        )
        for line in lines[: min(30, len(lines))]
    )
    first_page = min(line.page_number for line in lines)
    first_page_sizes = [
        line.font_size
        for line in lines
        if line.page_number == first_page and line.font_size > 0
    ]
    first_page_median_size = (
        float(median(first_page_sizes)) if first_page_sizes else body_size
    )
    for line in lines:
        if line.block_type == "figure":
            line.style = "图片"
            continue
        if line.block_type == "table_title":
            line.style = TABLE_TITLE_STYLE
            continue
        if line.block_type == "table":
            line.style = TABLE_STYLE
            continue
        if line.extraction_method == "vision_model" and line.style.startswith(HEADING_STYLE):
            continue
        if (
            line.page_number == first_page
            and line.alignment == "center"
            and line.y0 <= line.page_height * 0.58
            and line.font_size >= max(first_page_median_size * 1.28, body_size * 0.95)
            and 4 <= _text_length(line.text) <= 100
            and not _is_non_heading_value(line.text)
        ):
            line.style = HEADING_STYLE
            has_document_title = True
            continue
        marker_level = _numbered_heading_level(line.text)
        if marker_level is not None and _looks_like_numbered_heading_line(
            line,
            body_size=body_size,
        ):
            if has_document_title:
                marker_level += 1
            line.style = _heading_style(marker_level)
            continue
        if _is_visual_heading(line, body_size=body_size):
            ratio = line.font_size / max(1.0, body_size)
            if line.alignment == "center" and ratio >= 1.25:
                level = 1
            elif ratio >= 1.45:
                level = 2 if has_document_title else 1
            elif ratio >= 1.18 or line.bold:
                level = 3 if has_document_title else 2
            else:
                continue
            line.style = _heading_style(level)
        else:
            line.style = BODY_STYLE


def _looks_like_numbered_heading_line(line: PdfLine, *, body_size: float) -> bool:
    length = _text_length(line.text)
    if length == 0 or length > 72:
        return False
    if re.search(r"[。！？.!?；;]\s*$", line.text):
        return False
    if length <= 32:
        return True
    return line.bold or line.font_size >= body_size * 1.08


def _merge_wrapped_paragraphs(lines: list[PdfLine]) -> list[PdfLine]:
    if not lines:
        return []
    output: list[PdfLine] = []
    current: PdfLine | None = None
    for line in sorted(lines, key=lambda item: item.order):
        if current is None:
            current = line
            continue
        if _can_merge_paragraph_lines(current, line):
            current = _merge_line_group([current, line], style=current.style)
            continue
        output.append(current)
        current = line
    if current is not None:
        output.append(current)
    for order, line in enumerate(output):
        line.order = order
    return output


def _can_merge_paragraph_lines(previous: PdfLine, current: PdfLine) -> bool:
    if previous.page_number != current.page_number:
        return False
    if previous.block_type != "paragraph" or current.block_type != "paragraph":
        return False
    if previous.style != BODY_STYLE or current.style != BODY_STYLE:
        return False
    if previous.extraction_method != current.extraction_method:
        return False
    if previous.alignment == "center" or current.alignment == "center":
        return False
    if abs(previous.x0 - current.x0) > max(12.0, previous.font_size * 1.5):
        return False
    gap = current.y0 - previous.y1
    if gap < -2 or gap > max(8.0, previous.font_size * 1.35):
        return False
    if re.search(r"[。！？.!?；;：:]\s*$", previous.text):
        return False
    if _numbered_heading_level(current.text) is not None:
        return False
    return True


def _build_pdf_items(lines: list[PdfLine]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line in sorted(lines, key=lambda item: item.order):
        if line.block_type == "figure":
            item_type = "image"
            style = "图片"
        elif line.block_type == "table_title":
            item_type = "table"
            style = TABLE_TITLE_STYLE
        elif line.block_type == "table":
            item_type = "table"
            style = TABLE_STYLE
        else:
            item_type = "paragraph"
            style = line.style
        items.append(
            {
                "type": item_type,
                "style": style,
                "text": line.text,
                "source": (
                    f"page:{line.page_number}:line:{line.order}:"
                    f"{line.extraction_method}"
                ),
                "page_number": line.page_number,
                "bbox": [line.x0, line.y0, line.x1, line.y1],
                "confidence": line.confidence,
                "extraction_method": line.extraction_method,
                "font_size": line.font_size,
                "bold": line.bold,
                **({"path": line.text} if item_type == "image" else {}),
            }
        )
        items.extend(_link_items(line))
    return items


def write_items_to_txt(items: list[dict[str, Any]], source_file: str | Path) -> Path:
    source_path = Path(source_file)
    txt_dir = processing_subdir(source_path, "txt")
    txt_dir.mkdir(parents=True, exist_ok=True)
    output_path = txt_dir / f"{source_path.stem}.txt"
    output_path.write_text(format_extracted_items(items), encoding="utf-8")
    return output_path


def write_parse_manifest(
    source_file: str | Path,
    *,
    source_reference: str | Path | None,
    profiles: list[PdfPageProfile],
    page_metrics: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    item_count: int,
    text_path: Path,
    model_cleanup_mode: str,
    model_timeout_purposes: list[str] | None = None,
    model_failure_purposes: list[str] | None = None,
    model_failure_reasons: list[str] | None = None,
    disabled_model_tasks: list[str] | None = None,
) -> Path:
    source_path = Path(source_file)
    json_dir = processing_subdir(source_path, "json")
    json_dir.mkdir(parents=True, exist_ok=True)
    output_path = json_dir / "pdf_parse_manifest.json"
    counts = Counter(profile.kind.value for profile in profiles)
    payload = {
        "pipeline": "unified_pdf_parser",
        "source_pdf": str(source_path),
        "source_reference": str(source_reference or source_file),
        "page_count": len(profiles),
        "page_kind_counts": dict(counts),
        "pages": [
            {
                **asdict(profile),
                "kind": profile.kind.value,
            }
            for profile in profiles
        ],
        "page_metrics": page_metrics,
        "warnings": warnings,
        "item_count": item_count,
        "text_path": str(text_path),
        "ocr_dpi": OCR_DPI,
        "vision_dpi": VISION_DPI,
        "table_vision_dpi": TABLE_VISION_DPI,
        "vision_max_pixels": VISION_MAX_PIXELS,
        "table_vision_max_pixels": TABLE_VISION_MAX_PIXELS,
        "vision_read_timeout": VISION_READ_TIMEOUT,
        "table_vision_read_timeout": TABLE_VISION_READ_TIMEOUT,
        "vision_min_draft_chars": VISION_MIN_DRAFT_CHARS,
        "vision_circuit_breaker_timeouts": VISION_CIRCUIT_BREAKER_TIMEOUTS,
        "vision_circuit_breaker_failures": VISION_CIRCUIT_BREAKER_FAILURES,
        "vision_image_transport": VISION_IMAGE_TRANSPORT,
        "external_vision_enabled": _env_bool(EXTERNAL_VISION_OPT_IN_ENV, True),
        "vision_streaming": False,
        "vision_response_format": "text_protocol",
        "vision_jpeg_quality": VISION_JPEG_QUALITY,
        "full_page_max_output_tokens": FULL_PAGE_MAX_OUTPUT_TOKENS,
        "table_max_output_tokens": TABLE_MAX_OUTPUT_TOKENS,
        "model_cleanup_mode": model_cleanup_mode,
        "model_timeout_purposes": list(model_timeout_purposes or []),
        "model_failure_purposes": list(model_failure_purposes or []),
        "model_failure_reasons": list(model_failure_reasons or []),
        "disabled_model_tasks": list(disabled_model_tasks or []),
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def _existing_asme_split_sources(source: str | Path) -> list[str | Path]:
    source_text = str(source)
    parsed = urlparse(source_text.replace("\\", "/"))
    decoded = unquote(source_text.replace("\\", "/"))
    if parsed.scheme in {"minio", "s3"}:
        reference = parse_raw_document_reference(source)
        source_path = PurePosixPath(reference.object_name)
        prefix = source_path.parent / f"{source_path.stem}(切分版)"
        objects = list_raw_document_objects(
            build_minio_uri(reference.bucket, prefix.as_posix()),
            recursive=True,
        )
        return [
            item.uri
            for item in objects
            if PurePosixPath(item.object_name).suffix.lower() == ".pdf"
        ]

    path = Path(decoded)
    split_dir = path.parent / f"{path.stem}(切分版)"
    if not split_dir.is_dir():
        return []
    return sorted(child for child in split_dir.rglob("*.pdf") if child.is_file())


def _is_asme_parent_pdf(value: str | Path) -> bool:
    decoded = unquote(str(value).replace("\\", "/"))
    parsed = urlparse(decoded)
    path = PurePosixPath(parsed.path if parsed.scheme else decoded)
    return (
        path.suffix.lower() == ".pdf"
        and path.name.upper().startswith("ASME")
        and "(切分版)" not in decoded
    )


def _is_generated_asme_section(value: str | Path) -> bool:
    return "(切分版)" in unquote(str(value).replace("\\", "/"))


def _native_table_regions(page: Any) -> list[Any]:
    output: list[Any] = []
    finder = getattr(page, "find_tables", None)
    if finder is None:
        return output
    try:
        tables = finder().tables
    except Exception:
        return output
    for table in tables or []:
        bbox = getattr(table, "bbox", None)
        if bbox:
            output.append(fitz.Rect(bbox))
    return output


def _raster_table_regions(pixmap: Any, page: Any) -> list[Any]:
    if cv2 is None or np is None:
        return []
    image = _pixmap_array(pixmap)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(20, pixmap.width // 30), 1),
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(20, pixmap.height // 40)),
    )
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    grid = cv2.add(horizontal, vertical)
    contours, _hierarchy = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scale_x = float(page.rect.width) / float(pixmap.width)
    scale_y = float(page.rect.height) / float(pixmap.height)
    output: list[Any] = []
    min_area = pixmap.width * pixmap.height * 0.006
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width * height < min_area or width < pixmap.width * 0.18 or height < 20:
            continue
        output.append(
            fitz.Rect(
                x * scale_x,
                y * scale_y,
                (x + width) * scale_x,
                (y + height) * scale_y,
            )
        )
    return output


def _filter_raster_table_regions(
    regions: list[Any],
    lines: list[PdfLine],
) -> list[Any]:
    """Reject engineering drawings that merely contain horizontal/vertical lines."""

    output: list[Any] = []
    for region in regions:
        inside = [
            line
            for line in lines
            if _rect_center_in_any(_line_rect(line), [region])
            and _text_length(line.text) > 0
        ]
        if len(inside) < 6:
            continue
        x_centers = sorted((line.x0 + line.x1) / 2 for line in inside)
        y_centers = sorted((line.y0 + line.y1) / 2 for line in inside)
        if _position_cluster_count(x_centers, tolerance=12.0) < 2:
            continue
        if _position_cluster_count(y_centers, tolerance=6.0) < 2:
            continue
        if sum(_text_length(line.text) for line in inside) < 12:
            continue
        output.append(region)
    return output


def _position_cluster_count(values: list[float], *, tolerance: float) -> int:
    if not values:
        return 0
    count = 1
    previous = values[0]
    for value in values[1:]:
        if value - previous > tolerance:
            count += 1
        previous = value
    return count


def _assign_table_blocks(lines: list[PdfLine], table_regions: list[Any]) -> None:
    for line in lines:
        if line.block_type == "figure":
            continue
        if line.block_type == "table_title" or _is_table_title_text(line.text):
            line.block_type = "table_title"
            line.style = TABLE_TITLE_STYLE
            continue
        matching_region = _containing_region_index(_line_rect(line), table_regions)
        if line.block_type == "table" or matching_region is not None:
            line.block_type = "table"
            line.style = TABLE_STYLE
            if matching_region is not None:
                line.block_id = 1_000_000 + matching_region
            elif line.block_id == 0:
                line.block_id = -(line.order + 1)
            continue
        if TABLE_TEXT_PATTERN.search(line.text):
            line.block_type = "table"
            line.style = TABLE_STYLE
            if line.block_id == 0:
                line.block_id = -(line.order + 1)
    # Repeated multi-column rows are deliberately not used as a table signal:
    # they are indistinguishable from normal journal / standard two-column
    # layout. Geometry, raster grid lines and explicit captions are auditable.


def _containing_region_index(rect: Any, regions: list[Any]) -> int | None:
    center = fitz.Point((rect.x0 + rect.x1) / 2.0, (rect.y0 + rect.y1) / 2.0)
    for index, region in enumerate(regions, start=1):
        if fitz.Rect(region).contains(center):
            return index
    return None


def _is_table_title_text(text: str) -> bool:
    cleaned = _clean_line_text(text)
    if not TABLE_TITLE_PATTERN.match(cleaned) or _text_length(cleaned) > 80:
        return False
    if re.fullmatch(
        r"[A-Z]?\d+(?:[.\-—]\d+)*\s*系列\s*(?:表|TABLE)\s*\d+",
        cleaned,
        re.IGNORECASE,
    ):
        return True
    match = re.match(
        r"^(?:表|TABLE)\s*[A-Z]?\d+(?:[.\-—]\d+)*\s*(?P<tail>.*)$",
        cleaned,
        re.IGNORECASE,
    )
    if match is None:
        return False
    tail = match.group("tail").strip()
    if re.search(r"[。！？!?；;：:]\s*$", cleaned):
        return False
    if not tail:
        return True
    return not re.match(r"^(?:规定|给出|列出|中|所示|见|按|参见)", tail)


def _consolidate_local_table_lines(lines: list[PdfLine]) -> list[PdfLine]:
    """Turn local table-cell fragments into one JSON matrix per table region."""

    ordered = sorted(lines, key=lambda line: line.order)
    groups: dict[tuple[int, int], list[PdfLine]] = {}
    for line in ordered:
        if line.block_type != "table" or line.extraction_method.startswith("vision_table"):
            continue
        groups.setdefault((line.page_number, line.block_id), []).append(line)

    output: list[PdfLine] = []
    emitted: set[tuple[int, int]] = set()
    for line in ordered:
        key = (line.page_number, line.block_id)
        if line.block_type != "table" or line.extraction_method.startswith("vision_table"):
            output.append(line)
            continue
        if key in emitted:
            continue
        emitted.add(key)
        group = groups.get(key) or [line]
        if len(group) == 1 and _looks_like_json_table_text(group[0].text):
            output.append(group[0])
            continue
        output.append(_local_table_group_to_json(group))

    output.sort(key=lambda line: line.order)
    for order, line in enumerate(output):
        line.order = order
    return output


def _local_table_group_to_json(lines: list[PdfLine]) -> PdfLine:
    ordered = sorted(lines, key=lambda line: (line.y0, line.x0, line.order))
    heights = [max(1.0, line.y1 - line.y0) for line in ordered]
    row_tolerance = max(OCR_ROW_TOLERANCE, float(median(heights)) * 0.55)
    row_groups: list[list[PdfLine]] = []
    row_centers: list[float] = []
    for line in ordered:
        center = (line.y0 + line.y1) / 2.0
        if row_groups and abs(center - row_centers[-1]) <= row_tolerance:
            row_groups[-1].append(line)
            row_centers[-1] = sum(
                (item.y0 + item.y1) / 2.0 for item in row_groups[-1]
            ) / len(row_groups[-1])
        else:
            row_groups.append([line])
            row_centers.append(center)
    rows = [
        [_clean_line_text(line.text) for line in sorted(row, key=lambda item: item.x0)]
        for row in row_groups
    ]
    rows = [[cell for cell in row if cell] for row in rows]
    rows = [row for row in rows if row]
    rect = fitz.Rect(
        min(line.x0 for line in ordered),
        min(line.y0 for line in ordered),
        max(line.x1 for line in ordered),
        max(line.y1 for line in ordered),
    )
    first = ordered[0]
    return replace(
        first,
        text=json.dumps(rows, ensure_ascii=False, separators=(",", ":")),
        x0=float(rect.x0),
        y0=float(rect.y0),
        x1=float(rect.x1),
        y1=float(rect.y1),
        font_size=float(median([line.font_size for line in ordered])),
        bold=any(line.bold for line in ordered),
        order=min(line.order for line in ordered),
        style=TABLE_STYLE,
        confidence=min(line.confidence for line in ordered),
        extraction_method="local_table_json",
        block_type="table",
        urls=_dedupe_texts(url for line in ordered for url in line.urls),
    )


def _looks_like_json_table_text(text: str) -> bool:
    try:
        payload = json.loads(str(text or ""))
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(payload, list)


def _merge_same_row_lines(lines: list[PdfLine], *, tolerance: float) -> list[PdfLine]:
    if len(lines) < 2:
        return lines
    rows: list[list[PdfLine]] = []
    for line in sorted(lines, key=lambda item: (item.y0, item.x0, item.order)):
        for row in rows:
            if abs(line.y0 - row[0].y0) <= tolerance:
                row.append(line)
                break
        else:
            rows.append([line])
    output: list[PdfLine] = []
    for row in rows:
        row.sort(key=lambda item: item.x0)
        current_group: list[PdfLine] = [row[0]]
        for line in row[1:]:
            previous = current_group[-1]
            gap = line.x0 - previous.x1
            if gap > max(12.0, previous.font_size * 1.4):
                output.append(_merge_line_group(current_group, style=current_group[0].style))
                current_group = [line]
            else:
                current_group.append(line)
        output.append(_merge_line_group(current_group, style=current_group[0].style))
    output.sort(key=lambda line: (line.y0, line.x0, line.order))
    for order, line in enumerate(output):
        line.order = order
    return output


def _merge_line_group(lines: list[PdfLine], *, style: str) -> PdfLine:
    if len(lines) == 1:
        line = lines[0]
        line.style = style
        return line
    ordered = sorted(lines, key=lambda item: (item.y0, item.x0, item.order))
    text_parts = [
        (
            line.text,
            (line.x0, line.y0, line.x1, line.y1),
        )
        for line in ordered
    ]
    return PdfLine(
        text=_join_positioned_text(text_parts),
        page_number=ordered[0].page_number,
        page_width=ordered[0].page_width,
        page_height=ordered[0].page_height,
        x0=min(line.x0 for line in ordered),
        y0=min(line.y0 for line in ordered),
        x1=max(line.x1 for line in ordered),
        y1=max(line.y1 for line in ordered),
        font_size=max(line.font_size for line in ordered),
        bold=any(line.bold for line in ordered),
        order=min(line.order for line in ordered),
        alignment=ordered[0].alignment,
        style=style,
        urls=_dedupe_texts(url for line in ordered for url in line.urls),
        confidence=min(line.confidence for line in ordered),
        extraction_method=(
            ordered[0].extraction_method
            if len({line.extraction_method for line in ordered}) == 1
            else "hybrid"
        ),
        block_type=(
            "table" if any(line.block_type == "table" for line in ordered) else "paragraph"
        ),
        block_id=ordered[0].block_id,
        bold_score=max(line.bold_score for line in ordered),
    )


def _assign_page_alignment(lines: list[PdfLine]) -> None:
    if not lines:
        return
    body_candidates = [
        line.x0
        for line in lines
        if _text_length(line.text) >= 8 and line.x0 < line.page_width * 0.45
    ]
    left_margin = float(median(body_candidates)) if body_candidates else min(line.x0 for line in lines)
    for line in lines:
        center = (line.x0 + line.x1) / 2
        right_gap = line.page_width - line.x1
        if abs(center - line.page_width / 2) <= max(18.0, line.page_width * 0.055):
            line.alignment = "center"
        elif right_gap <= max(8.0, line.page_width * 0.02) or line.x0 >= line.page_width * 0.62:
            line.alignment = "right"
        elif abs(line.x0 - left_margin) <= max(8.0, line.page_width * 0.025) or line.x0 <= line.page_width * 0.18:
            line.alignment = "left"
        else:
            line.alignment = "other"


def _numbered_heading_level(text: str) -> int | None:
    value = text.strip()
    if _is_non_heading_value(value):
        return None
    if re.match(r"^第\s*[一二三四五六七八九十百千万\d]+\s*[篇章部分]", value):
        return 1
    if re.match(r"^[一二三四五六七八九十百千万]{1,8}\s*[、.．]", value):
        return 1
    if re.match(r"^[（(]\s*[一二三四五六七八九十百千万]{1,8}\s*[）)]", value):
        return 2
    match = re.match(
        r"^(\d{1,3}(?:\.\d{1,3})*)(?![\d./年月-])\s*[、.．]?\s*\S+",
        value,
    )
    if match:
        return min(MAX_HEADING_LEVEL, len(match.group(1).split(".")))
    if re.match(r"^[（(]\s*\d{1,3}\s*[）)]", value):
        return 4
    if re.match(r"^[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]", value):
        return 5
    return None


def _is_visual_heading(line: PdfLine, *, body_size: float) -> bool:
    text = line.text.strip()
    if not text or _text_length(text) > 120:
        return False
    if _is_non_heading_value(text):
        return False
    if re.match(r"^(?:注|NOTE)\s*[:：]", text, re.IGNORECASE):
        return False
    if line.alignment == "right":
        return False
    if re.search(r"[。！？.!?；;]\s*$", text):
        return False
    if STANDARD_CODE_PATTERN.match(text):
        return line.alignment == "center" or line.font_size >= body_size * 1.15
    return (
        (line.alignment == "center" and line.font_size >= body_size * 1.12)
        or (line.bold and line.font_size >= body_size * 1.02)
        or line.font_size >= body_size * 1.35
    )


def _is_non_heading_value(text: str) -> bool:
    return bool(re.fullmatch(
        r"[≈~～]?\s*\d+(?:\.\d+)?(?:\s*[-—~～]\s*\d+(?:\.\d+)?)?"
        r"\s*(?:%|℃|°C|mm|cm|m|kg|MPa|N)?",
        text.strip(),
        re.IGNORECASE,
    ))


def _heading_style(level: int) -> str:
    normalized = max(1, min(int(level), MAX_HEADING_LEVEL))
    return HEADING_STYLE if normalized == 1 else f"{HEADING_STYLE} {normalized}"


def _should_use_vision(
    mode: str,
    profile: PdfPageProfile,
    lines: list[PdfLine],
) -> bool:
    if mode == "never" or profile.kind not in {PdfPageKind.SCANNED, PdfPageKind.HYBRID}:
        return False
    if mode == "always":
        return True
    # Local OCR confidence is not a semantic correctness score. In the
    # visually checked GB scan it stayed above 0.93 while several characters
    # were wrong, so auto mode must review every fully scanned page.
    if profile.kind == PdfPageKind.SCANNED:
        return True
    if not lines:
        return True
    return _mean_confidence(lines) < 0.82 or _suspicious_text_ratio(lines) >= 0.10


def _should_use_full_page_vision(
    mode: str,
    profile: PdfPageProfile,
    lines: list[PdfLine],
    *,
    table_regions: list[Any],
) -> bool:
    if table_regions or not _should_use_vision(mode, profile, lines):
        return False
    if mode == "auto" and sum(_text_length(line.text) for line in lines) < VISION_MIN_DRAFT_CHARS:
        return False
    return True


def _normalize_model_mode(mode: str | None) -> str:
    value = str(mode or os.getenv(MODEL_MODE_ENV, DEFAULT_MODEL_MODE)).strip().lower()
    if value not in {"auto", "always", "never"}:
        raise ValueError(f"Unsupported PDF model cleanup mode: {value}")
    return value


def _normalize_model_style(style: str) -> str:
    value = style.strip()
    if value in {BODY_STYLE, TABLE_STYLE, TABLE_TITLE_STYLE, HEADING_STYLE}:
        return value
    match = re.match(r"^标题\s*([2-6])$", value)
    return f"标题 {match.group(1)}" if match else BODY_STYLE


def _is_margin_line(line: PdfLine) -> bool:
    return line.y1 <= line.page_height * 0.12 or line.y0 >= line.page_height * 0.88


def _is_obvious_garbage(text: str) -> bool:
    normalized = _normalized_text(text)
    if not normalized:
        return True
    if SUSPICIOUS_TEXT_PATTERN.fullmatch(normalized):
        return True
    if KNOWN_WATERMARK_PATTERN.fullmatch(normalized):
        return True
    if any(pattern.search(_clean_line_text(text)) for pattern in LOW_VALUE_TEXT_PATTERNS):
        return True
    if len(normalized) == 1 and not normalized.isalnum() and not _contains_cjk(normalized):
        return True
    return False


def _suspicious_text_ratio(lines: list[PdfLine]) -> float:
    text = "".join(line.text for line in lines)
    if not text:
        return 0.0
    suspicious = sum(
        1
        for char in text
        if char in "\ufffd�│¦¬" or (ord(char) < 32 and char not in "\r\n\t")
    )
    suspicious += sum(len(match.group(0)) for match in SUSPICIOUS_TEXT_PATTERN.finditer(text))
    return suspicious / max(1, len(text))


def _mean_confidence(lines: list[PdfLine]) -> float:
    values = [line.confidence for line in lines if line.confidence > 0]
    return sum(values) / len(values) if values else 0.0


def _page_ink_ratio(page: Any) -> float:
    pixmap = _render_page(page, dpi=72)
    image = _pixmap_array(pixmap)
    if cv2 is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.mean(axis=2)
    return float((gray < 245).mean())


def _render_page(page: Any, *, dpi: int) -> Any:
    scale = dpi / 72.0
    return page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)


def _render_model_image(
    page: Any,
    *,
    dpi: int,
    max_pixels: int,
    clip: Any | None = None,
) -> Any:
    target = fitz.Rect(clip) if clip is not None else fitz.Rect(page.rect)
    requested_scale = max(1.0, float(dpi) / 72.0)
    requested_pixels = target.width * requested_scale * target.height * requested_scale
    if requested_pixels > max_pixels > 0:
        requested_scale *= math.sqrt(max_pixels / requested_pixels)
    return page.get_pixmap(
        matrix=fitz.Matrix(requested_scale, requested_scale),
        clip=target,
        alpha=False,
    )


def _vision_content(
    prompt: str,
    pixmap: Any,
    *,
    client: LLMClient,
    purpose: str,
    draft_chars: int,
    max_tokens: int,
    reference_rect: Any,
) -> tuple[list[dict[str, Any]], str | None]:
    jpeg_bytes = pixmap.tobytes(
        "jpeg",
        jpg_quality=max(40, min(VISION_JPEG_QUALITY, 95)),
    )
    effective_dpi = round(
        pixmap.width / max(1.0, float(reference_rect.width)) * 72.0,
        1,
    )
    transport = _vision_transport_for_client(client)
    uploaded_file_id: str | None = None
    image_url = ""
    if transport == "file":
        uploaded_file_id = client.upload_file(
            filename="pdf-page.jpg",
            content=jpeg_bytes,
            purpose="image",
            content_type="image/jpeg",
            read_timeout=VISION_FILE_TIMEOUT,
        )
        image_url = f"ms://{uploaded_file_id}"
    if transport == "data_uri":
        encoded = base64.b64encode(jpeg_bytes).decode("ascii")
        image_url = "data:image/jpeg;base64," + encoded

    _log(
        "model input: "
        f"step={purpose}; transport={transport}; dpi={effective_dpi}; "
        f"pixels={pixmap.width}x{pixmap.height}; "
        f"megapixels={pixmap.width * pixmap.height / 1_000_000:.2f}; "
        f"jpeg_kb={len(jpeg_bytes) / 1024:.1f}; "
        f"chat_image_ref_chars={len(image_url)}; "
        f"prompt_chars={len(prompt)}; draft_chars={draft_chars}; "
        f"max_output_tokens={max_tokens}"
    )
    return [
        {
            "type": "image_url",
            "image_url": {"url": image_url},
        },
        {"type": "text", "text": prompt},
    ], uploaded_file_id


def _vision_transport_for_client(client: LLMClient) -> str:
    configured = VISION_IMAGE_TRANSPORT
    if configured not in {"auto", "file", "data_uri"}:
        raise ValueError(f"Unsupported PDF vision image transport: {configured}")
    if configured != "auto":
        return configured
    settings = getattr(client, "settings", None)
    base_url = str(getattr(settings, "base_url", "")).lower()
    return "file" if "moonshot" in base_url and hasattr(client, "upload_file") else "data_uri"


def _delete_uploaded_vision_file(
    client: LLMClient,
    uploaded_file_id: str | None,
) -> None:
    if not uploaded_file_id:
        return
    try:
        client.delete_file(
            uploaded_file_id,
            read_timeout=VISION_FILE_CLEANUP_TIMEOUT,
        )
    except (AttributeError, LLMAPIError, OSError) as exc:
        _log(
            f"temporary vision file cleanup failed for {uploaded_file_id}: "
            f"{type(exc).__name__}: {exc}"
        )


def _pixmap_array(pixmap: Any) -> Any:
    if np is None:
        raise ModuleNotFoundError("numpy is required for PDF OCR")
    image = np.frombuffer(pixmap.samples, dtype=np.uint8)
    image = image.reshape(pixmap.height, pixmap.width, pixmap.n)
    if pixmap.n == 4:
        image = image[:, :, :3]
    return image


def _rapid_ocr_engine() -> Any:
    global _RAPID_OCR_ENGINE
    if RapidOCR is None:
        raise ModuleNotFoundError(
            "rapidocr-onnxruntime is required for scanned PDF pages"
        )
    if _RAPID_OCR_ENGINE is None:
        _RAPID_OCR_ENGINE = RapidOCR()
    return _RAPID_OCR_ENGINE


def _ocr_bold_density(
    image: Any,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> float:
    left = max(0, int(x0))
    top = max(0, int(y0))
    right = min(image.shape[1], int(math.ceil(x1)))
    bottom = min(image.shape[0], int(math.ceil(y1)))
    if right <= left or bottom <= top:
        return 0.0
    crop = image[top:bottom, left:right]
    if cv2 is not None:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    else:
        gray = crop.mean(axis=2)
    return float((gray < 165).mean())


def _image_rectangles(page: Any) -> list[Any]:
    output: list[Any] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 1 or not block.get("bbox"):
            continue
        rect = fitz.Rect(block["bbox"])
        rect.intersect(page.rect)
        if not rect.is_empty:
            output.append(rect)
    return output


def _rectangle_coverage(rects: list[Any], page_rect: Any) -> float:
    page_area = max(1.0, float(page_rect.width * page_rect.height))
    return min(1.0, sum(float(rect.width * rect.height) for rect in rects) / page_area)


def _merge_rectangles(rects: list[Any]) -> list[Any]:
    output: list[Any] = []
    for candidate in sorted((fitz.Rect(rect) for rect in rects), key=lambda rect: (rect.y0, rect.x0)):
        if candidate.is_empty:
            continue
        for existing in output:
            if existing.intersects(candidate):
                existing.include_rect(candidate)
                break
        else:
            output.append(candidate)
    return output


def _rect_center_in_any(rect: Any, regions: list[Any]) -> bool:
    center = fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
    return any(center in fitz.Rect(region) for region in regions)


def _intersection_over_smaller(left: Any, right: Any) -> float:
    intersection = fitz.Rect(left)
    intersection.intersect(right)
    if intersection.is_empty:
        return 0.0
    smaller = min(max(1.0, left.get_area()), max(1.0, right.get_area()))
    return intersection.get_area() / smaller


def _line_rect(line: PdfLine) -> Any:
    return fitz.Rect(line.x0, line.y0, line.x1, line.y1)


def _assign_native_links(page: Any, lines: list[PdfLine]) -> None:
    for link in page.get_links() or []:
        uri = str(link.get("uri") or "").strip()
        rect = link.get("from")
        if not uri or rect is None:
            continue
        for line in lines:
            if _line_rect(line).intersects(rect):
                line.urls = _dedupe_texts([*line.urls, uri])


def _link_items(line: PdfLine) -> list[dict[str, Any]]:
    urls = [match.group(0).rstrip("。；;，,)") for match in URL_PATTERN.finditer(line.text)]
    urls.extend(line.urls)
    return [
        {
            "type": "link_ref",
            "style": LINK_STYLE,
            "text": url,
            "url": url,
            "description": line.text if url not in line.text else url,
            "source": f"page:{line.page_number}:line:{line.order}:link",
        }
        for url in _dedupe_texts(urls)
    ]


def _join_positioned_text(
    parts: list[tuple[str, tuple[float, float, float, float]]],
) -> str:
    output = ""
    previous_box: tuple[float, float, float, float] | None = None
    for text, box in parts:
        if output and previous_box is not None:
            gap = box[0] - previous_box[2]
            needs_space = (
                gap > max(1.5, (box[3] - box[1]) * 0.16)
                and not (_ends_cjk(output) or _starts_cjk(text))
            )
            if needs_space:
                output += " "
        output += text
        previous_box = box
    return _clean_line_text(output)


def _is_bold_font(font_name: str, flags: int) -> bool:
    normalized = font_name.lower().replace("-", "").replace("_", "").replace(" ", "")
    positive = ("bold", "black", "heavy", "semibold", "demibold", "extrabold", "ultrabold")
    negative = ("regular", "normal", "book", "light", "thin", "medium", "roman")
    if any(token in normalized for token in positive):
        return True
    if any(token in normalized for token in negative):
        return False
    return bool(flags & 16)


def _chat_json_object(
    client: LLMClient,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    purpose: str,
    task: str = "generic",
    read_timeout: float | None = None,
    vision_session: PdfVisionSession | None = None,
) -> dict[str, Any]:
    settings = getattr(client, "settings", None)
    selected_model = VISION_MODEL or str(getattr(settings, "model", ""))
    non_thinking = build_non_thinking_extra_body(
        model=selected_model,
        base_url=str(getattr(settings, "base_url", "")),
    )
    attempts = [
        {
            **non_thinking,
            "response_format": {"type": "json_object"},
        },
        dict(non_thinking),
    ]
    failures: list[str] = []
    for attempt_index, extra_body in enumerate(attempts, start=1):
        try:
            response = client.chat(
                messages,
                model=VISION_MODEL,
                temperature=0,
                max_tokens=max_tokens,
                extra_body=extra_body,
                read_timeout=read_timeout,
                stream=True,
            )
            payload = _parse_json_object(response)
            if vision_session is not None:
                vision_session.mark_success(task)
            return payload
        except LLMTimeoutError:
            if vision_session is not None:
                vision_session.mark_timeout(task, purpose)
            # A second request with the same image is likely to time out again.
            # Preserve local OCR/native output. Repeated timeouts eventually
            # open the document-local circuit for this workload type.
            raise
        except (LLMAPIError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"attempt {attempt_index}: {type(exc).__name__}: {exc}")
            if attempt_index < len(attempts):
                _log(
                    f"{purpose} JSON response invalid; retrying without "
                    f"response_format ({failures[-1]})"
                )
    raise ValueError(f"{purpose} returned no valid JSON; {'; '.join(failures)}")


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = str(text or "").lstrip("\ufeff").strip()
    if not stripped:
        raise ValueError("model response content is empty")
    stripped = re.sub(
        r"^\s*<think>[\s\S]*?</think>\s*",
        "",
        stripped,
        count=1,
        flags=re.IGNORECASE,
    )
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(
            "model response does not contain a complete JSON object "
            f"(length={len(stripped)})"
        )
    stripped = stripped[start : end + 1]
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("Visual PDF response is not a JSON object")
    return payload


def _validate_pdf_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"PDF document not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Only .pdf files are supported: {path}")
    if path.name.startswith("~$"):
        raise ValueError(f"Temporary PDF lock files are not supported: {path}")


def _ensure_pdf_dependencies() -> None:
    if fitz is None:
        raise ModuleNotFoundError("pymupdf is required to parse PDF documents")
    if np is None:
        raise ModuleNotFoundError("numpy is required to parse PDF documents")


def _clean_line_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.replace("\u00a0", " ").replace("\u3000", " ")
    value = value.replace("\ue011", " ")
    value = re.sub(r"[ \t]+", " ", value)
    return value.strip()


def _normalize_pdf_text(text: str, *, font_name: str = "") -> str:
    value = str(text or "")
    if BROKEN_LATIN_FONT_PATTERN.search(font_name):
        value = value.translate(BROKEN_LATIN_GLYPH_MAP)
    return _clean_line_text(value)


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).strip()


def _text_length(text: str) -> int:
    return len(_normalized_text(text))


def _contains_cjk(text: str) -> bool:
    return any("\u3400" <= char <= "\u9fff" for char in text)


def _starts_cjk(text: str) -> bool:
    return bool(text) and "\u3400" <= text[0] <= "\u9fff"


def _ends_cjk(text: str) -> bool:
    return bool(text) and "\u3400" <= text[-1] <= "\u9fff"


def _dedupe_texts(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_output_budget_exhausted(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "finish_reason='length'" in message or (
        "content is empty" in message and "reasoning_length=" in message
    )


def _log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[unified_pdf_parser][{timestamp}] {message}", flush=True)


# Compatibility names used by the ASME splitter.  They point to the new
# canonical implementation, not to the disconnected legacy PDF parser.
_extract_lines = _extract_native_lines
_build_items = _build_pdf_items

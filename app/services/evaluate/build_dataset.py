from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.llm import LLMAPIError, LLMClient
from app.services.parser.parser import parse_document
from app.services.parser.word_parser import format_extracted_items


DEFAULT_OUTPUT_FILE = Path("data") / "dataset" / "test.json"
DEFAULT_PDF_SOURCE_DIR = Path("data") / "dataset" / "pdf_sources"
SUPPORTED_SUFFIXES = {".docx", ".pdf", ".pptx", ".xlsx"}
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".json", ".csv"}
DOCX_PDF_CONVERTERS = {"auto", "word", "libreoffice"}
GENERATION_MAX_ATTEMPTS = 3
GENERATION_RETRY_BASE_SECONDS = 5.0


def log(message: str, *, verbose: bool = True) -> None:
    if verbose:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [build_dataset] {message}", flush=True)


@dataclass(frozen=True)
class DocumentSource:
    """Text extracted from the same parsed artifact used by ingestion."""

    original_path: Path
    source_path: Path
    source_kind: str
    text: str


SYSTEM_PROMPT = """你是企业内部知识库评测数据集专家，负责根据企业文档生成高质量问答题。

重要约束：
1. 被评测的回答模型不知道问题来自哪一份文档，因此问题和参考答案都不要使用“此文档中”“上述段落”“该材料”等依赖来源上下文的代称。
2. 题目要像真实员工会问的问题：设身处地预测用户可能怎么问，避免无意义、机械摘抄式的问题。陷阱题可以故意误导，但参考答案必须纠正错误。
3. 所有题目必须能由给定文档内容支撑；如果文档没有相关信息，不要编造公司制度或事实。
4. 图片、链接等内容可能仍是占位符，不要把占位符本身当成最终业务答案。

题型可选：
- 知识问答类：针对文档中存在的知识点提问。
- 信息咨询题：询问公司内部情况，例如下班时间、公司规模、WiFi 密码等；仅在文档确有信息时生成。
- 流程问答题：请假流程、业务流程、系统操作流程等。
- 陷阱题：问题中包含与文档不符的前提或诱导，参考答案需要明确纠错。
- 超长提问题：问题题干较长，建议 300 字以上，可综合多个需求。
- 噪声题：包含口语、方言、错别字、语气词、倒叙、无效字符等，但仍应可理解。
"""


USER_PROMPT_TEMPLATE = """请基于下面这份企业内部文档生成评测问答对。

文档名称：{document_name}
文档来源形式：{source_kind}
文档字符数：{content_length}
目标题目数量：约 {target_count} 条。内容少时可以少于目标数量，内容丰富时可以略多，但不要为了凑数编造。

输出要求：
- 只输出 JSON，不要输出 Markdown 代码块。
- JSON 顶层格式为：{{"items": [ ... ]}}
- 每个 item 必须包含：
  - question_types: 字符串数组，取值来自 ["知识问答类","信息咨询题","流程问答题","陷阱题","超长提问题","噪声题"]，可以多个。
  - question: 用户问题。
  - reference_answer: 参考答案。陷阱题必须指出问题中的错误并给出符合文档的答案；如果文档不足以回答，要说明文档未提供该信息。
  - evidence: 文档中支撑答案的简短依据，不超过 200 字。
  - difficulty: "easy"、"medium" 或 "hard"。

文档内容：
{document_text}
"""


def build_dataset(
    input_dir: str | Path,
    output_file: str | Path = DEFAULT_OUTPUT_FILE,
    *,
    target_count: int = 10,
    max_document_chars: int = 30000,
    model: str | None = None,
    pdf_dir: str | Path | None = None,
    docx_pdf_converter: str = "auto",
    rebuild_pdf: bool = False,
    verbose: bool = True,
    pdf_only: bool = False,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Generate QA pairs for every supported document under an input directory."""
    input_path = Path(input_dir)
    log(f"start building dataset from: {input_path}", verbose=verbose)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_path}")

    pdf_output_dir = Path(pdf_dir) if pdf_dir is not None else DEFAULT_PDF_SOURCE_DIR
    log(f"legacy pdf source/cache directory: {pdf_output_dir}", verbose=verbose)
    log(f"output dataset file: {output_file}", verbose=verbose)
    documents = list(
        iter_document_files(
            input_path,
            pdf_output_dir=pdf_output_dir,
            pdf_only=pdf_only,
        )
    )
    log(f"found {len(documents)} supported source file(s)", verbose=verbose)
    if not documents:
        raise ValueError(
            f"No supported documents found in {input_path}. "
            f"Supported suffixes: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    log("initializing LLM client", verbose=verbose)
    llm = LLMClient()
    log(f"LLM model: {model or llm.settings.model}", verbose=verbose)
    generated_at = datetime.now(timezone.utc).isoformat()
    output_path = Path(output_file)
    all_items = [] if dry_run else load_existing_dataset(output_path, verbose=verbose)
    documents = resume_documents(
        documents,
        existing_items=all_items,
        document_root=input_path,
        verbose=verbose,
    )

    for doc_index, document_path in enumerate(documents, start=1):
        log(
            f"[{doc_index}/{len(documents)}] loading source: {document_path}",
            verbose=verbose,
        )
        source = load_document_source(
            document_path,
            document_root=input_path,
            pdf_output_dir=pdf_output_dir,
            docx_pdf_converter=docx_pdf_converter,
            rebuild_pdf=rebuild_pdf,
            verbose=verbose,
        )
        if not source.text.strip():
            log(
                f"[{doc_index}/{len(documents)}] skipped empty extracted text: {source.source_path}",
                verbose=verbose,
            )
            continue

        prompt_text = trim_document_text(source.text, max_document_chars)
        if len(prompt_text) < len(source.text):
            log(
                f"[{doc_index}/{len(documents)}] trimmed document text from {len(source.text)} to {len(prompt_text)} chars",
                verbose=verbose,
            )
        if dry_run:
            log(
                f"[{doc_index}/{len(documents)}] dry run: skipped LLM call for {source.original_path.name}",
                verbose=verbose,
            )
            continue
        log(
            f"[{doc_index}/{len(documents)}] calling generator model for {source.original_path.name}",
            verbose=verbose,
        )
        raw_items = generate_document_items(
            llm,
            document_name=source.original_path.name,
            source_kind=source.source_kind,
            document_text=prompt_text,
            content_length=len(source.text),
            target_count=target_count,
            model=model,
            verbose=verbose,
        )
        log(
            f"[{doc_index}/{len(documents)}] model returned {len(raw_items)} raw QA item(s)",
            verbose=verbose,
        )

        accepted_count = 0
        for index, item in enumerate(raw_items, start=1):
            normalized = normalize_item(
                item,
                source=source,
                document_root=input_path,
                generated_at=generated_at,
                item_index=index,
                generator_model=model or llm.settings.model,
                source_char_count=len(source.text),
            )
            if normalized is not None:
                all_items.append(normalized)
                accepted_count += 1
        log(
            f"[{doc_index}/{len(documents)}] accepted {accepted_count} QA item(s); total={len(all_items)}",
            verbose=verbose,
        )
        write_dataset_json(output_path, all_items, verbose=verbose)

    if dry_run:
        log("dry run complete; dataset file was not written", verbose=verbose)
        return all_items

    log(f"done, wrote {len(all_items)} QA item(s)", verbose=verbose)
    return all_items


def load_existing_dataset(path: Path, *, verbose: bool = True) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Existing dataset JSON is invalid and cannot be resumed: {path}") from exc
    if not isinstance(data, list):
        raise ValueError(f"Existing dataset JSON must be a list: {path}")
    items = [item for item in data if isinstance(item, dict)]
    log(f"loaded {len(items)} existing QA item(s) from {path}", verbose=verbose)
    return items


def resume_documents(
    documents: list[Path],
    *,
    existing_items: list[dict[str, Any]],
    document_root: Path,
    verbose: bool = True,
) -> list[Path]:
    if not existing_items:
        return documents

    last_item = existing_items[-1]
    last_document_path = str(last_item.get("document_path") or "").strip()
    last_document_name = str(last_item.get("document_name") or "").strip()
    if not last_document_path and not last_document_name:
        return documents

    for index, document_path in enumerate(documents):
        relative_path = relative_to_root(document_path, document_root)
        if relative_path == last_document_path or document_path.name == last_document_name:
            remaining = documents[index + 1 :]
            log(
                f"resuming after last completed document: {last_document_name or last_document_path}; "
                f"remaining documents={len(remaining)}",
                verbose=verbose,
            )
            return remaining

    log(
        f"existing dataset last document was not found in current input: "
        f"{last_document_name or last_document_path}; starting from the beginning",
        verbose=verbose,
    )
    return documents


def write_dataset_json(path: Path, items: list[dict[str, Any]], *, verbose: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)
    log(f"saved dataset progress: {path} ({len(items)} QA item(s))", verbose=verbose)


def convert_docx_folder_to_pdf(
    input_dir: str | Path,
    output_dir: str | Path = DEFAULT_PDF_SOURCE_DIR,
    *,
    converter: str = "auto",
    rebuild: bool = False,
    verbose: bool = True,
) -> list[Path]:
    """Convert all docx files under input_dir to mirrored PDF files under output_dir."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_path}")

    docx_files = [
        docx_path
        for docx_path in sorted(input_path.rglob("*.docx"))
        if not docx_path.name.startswith("~$")
    ]
    log(f"found {len(docx_files)} docx file(s) to convert", verbose=verbose)
    converted_files: list[Path] = []
    for index, docx_path in enumerate(docx_files, start=1):
        relative = docx_path.relative_to(input_path)
        pdf_path = output_path / relative.with_suffix(".pdf")
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        if pdf_path.exists() and not rebuild:
            log(f"[{index}/{len(docx_files)}] using existing PDF: {pdf_path}", verbose=verbose)
            converted_files.append(pdf_path)
            continue
        log(f"[{index}/{len(docx_files)}] converting docx to PDF: {docx_path}", verbose=verbose)
        converted_files.append(
            convert_docx_to_pdf(
                docx_path,
                pdf_path,
                converter=converter,
                verbose=verbose,
            )
        )
    return converted_files


def iter_document_files(
    input_dir: Path,
    *,
    pdf_output_dir: Path,
    pdf_only: bool = False,
) -> list[Path]:
    pdf_output_dir = pdf_output_dir.resolve()
    paths: list[Path] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("~$"):
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if pdf_only and path.suffix.lower() != ".pdf":
            continue
        try:
            if path.resolve().is_relative_to(pdf_output_dir):
                continue
        except ValueError:
            pass
        paths.append(path)
    return paths


def dedupe_pdf_pairs(paths: list[Path]) -> list[Path]:
    pdf_keys = {
        (path.parent.resolve(), path.stem.lower())
        for path in paths
        if path.suffix.lower() == ".pdf"
    }
    deduped: list[Path] = []
    for path in paths:
        key = (path.parent.resolve(), path.stem.lower())
        if path.suffix.lower() != ".pdf" and key in pdf_keys:
            continue
        deduped.append(path)
    return deduped


def load_document_source(
    path: Path,
    *,
    document_root: Path,
    pdf_output_dir: Path,
    docx_pdf_converter: str = "auto",
    rebuild_pdf: bool = False,
    verbose: bool = True,
) -> DocumentSource:
    del document_root, pdf_output_dir, docx_pdf_converter
    source_path = ensure_processing_txt_source(
        path,
        rebuild=rebuild_pdf,
        verbose=verbose,
    )
    source_kind = f"parsed_{path.suffix.lower().lstrip('.')}_txt"
    log(f"using parsed text source ({source_kind}): {source_path}", verbose=verbose)

    return DocumentSource(
        original_path=path,
        source_path=source_path,
        source_kind=source_kind,
        text=read_document_text(source_path, verbose=verbose),
    )


def ensure_processing_txt_source(
    path: Path,
    *,
    rebuild: bool = False,
    verbose: bool = True,
) -> Path:
    txt_path = processing_txt_path(path)
    if txt_path.exists() and not rebuild:
        log(f"found existing parsed txt for {path.name}: {txt_path}", verbose=verbose)
        return txt_path

    log(f"parsing source document with ingestion parser: {path}", verbose=verbose)
    parsed_items = parse_document(path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    if not txt_path.exists():
        txt_path.write_text(format_extracted_items(parsed_items), encoding="utf-8")
    return txt_path


def processing_txt_path(path: Path) -> Path:
    return Path("data") / "processing" / path.stem / "txt" / f"{path.stem}.txt"


def ensure_pdf_source(
    path: Path,
    *,
    document_root: Path,
    pdf_output_dir: Path,
    docx_pdf_converter: str = "auto",
    rebuild_pdf: bool = False,
    verbose: bool = True,
) -> Path:
    if path.suffix.lower() == ".pdf":
        log(f"input is already PDF: {path}", verbose=verbose)
        return path

    candidates = [path.with_suffix(".pdf")]
    relative = path.relative_to(document_root)
    candidates.append(pdf_output_dir / relative.with_suffix(".pdf"))

    for candidate in candidates:
        if candidate.exists() and candidate.is_file() and not rebuild_pdf:
            log(f"found existing PDF for {path.name}: {candidate}", verbose=verbose)
            return candidate

    output_path = pdf_output_dir / relative.with_suffix(".pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".docx":
        return convert_docx_to_pdf(
            path,
            output_path,
            converter=docx_pdf_converter,
            verbose=verbose,
        )
    if path.suffix.lower() in TEXT_SUFFIXES:
        return convert_text_to_pdf(path, output_path, verbose=verbose)
    raise ValueError(f"Unsupported document type for PDF conversion: {path.suffix}")


def convert_docx_to_pdf(
    source_path: Path,
    output_path: Path,
    *,
    converter: str = "auto",
    verbose: bool = True,
) -> Path:
    converter = converter.lower().strip()
    if converter not in DOCX_PDF_CONVERTERS:
        raise ValueError(
            f"Unsupported DOCX PDF converter: {converter}. "
            f"Choose from: {', '.join(sorted(DOCX_PDF_CONVERTERS))}"
        )

    if converter in {"auto", "word"}:
        try:
            return convert_docx_to_pdf_with_word(source_path, output_path, verbose=verbose)
        except Exception as exc:
            if converter == "word":
                raise
            log(f"MS Word PDF conversion unavailable, falling back to LibreOffice: {exc}", verbose=verbose)

    return convert_docx_to_pdf_with_libreoffice(source_path, output_path, verbose=verbose)


def convert_docx_to_pdf_with_word(
    source_path: Path,
    output_path: Path,
    *,
    verbose: bool = True,
) -> Path:
    if platform.system().lower() != "windows":
        raise RuntimeError("MS Word PDF conversion is only available on Windows.")

    try:
        import pythoncom
        import win32com.client
    except ImportError as exc:
        raise RuntimeError(
            "MS Word PDF conversion requires pywin32. Install it with: pip install pywin32"
        ) from exc

    source_path = source_path.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log("running converter: Microsoft Word COM", verbose=verbose)
    started = time.perf_counter()

    word = None
    document = None
    pythoncom.CoInitialize()
    try:
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0
        document = word.Documents.Open(
            str(source_path),
            ConfirmConversions=False,
            ReadOnly=True,
            AddToRecentFiles=False,
            Revert=False,
            NoEncodingDialog=True,
        )
        document.ExportAsFixedFormat(
            OutputFileName=str(output_path),
            ExportFormat=17,
            OpenAfterExport=False,
            OptimizeFor=0,
            Range=0,
            Item=0,
            IncludeDocProps=True,
            KeepIRM=True,
            CreateBookmarks=1,
            DocStructureTags=True,
            BitmapMissingFonts=True,
            UseISO19005_1=False,
        )
    finally:
        if document is not None:
            document.Close(False)
        if word is not None:
            word.Quit()
        pythoncom.CoUninitialize()

    if not output_path.exists():
        raise RuntimeError(f"MS Word did not create expected PDF: {output_path}")
    log(
        f"MS Word conversion finished in {time.perf_counter() - started:.1f}s",
        verbose=verbose,
    )
    log(f"created PDF: {output_path}", verbose=verbose)
    return output_path


def convert_docx_to_pdf_with_libreoffice(
    source_path: Path,
    output_path: Path,
    *,
    verbose: bool = True,
) -> Path:
    converter = find_libreoffice_converter()
    if converter is None:
        raise RuntimeError(
            "DOCX to PDF conversion requires MS Word on Windows or LibreOffice/soffice. "
            f"Please install one converter, add soffice to PATH, set LIBREOFFICE_PATH/SOFFICE_PATH, "
            f"or provide an existing PDF beside {source_path}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"running converter: LibreOffice ({converter})", verbose=verbose)
    started = time.perf_counter()
    subprocess.run(
        [
            converter,
            "--headless",
            "--convert-to",
            "pdf:writer_pdf_Export",
            "--outdir",
            str(output_path.parent),
            str(source_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    log(
        f"LibreOffice conversion finished in {time.perf_counter() - started:.1f}s",
        verbose=verbose,
    )

    converted_path = output_path.parent / f"{source_path.stem}.pdf"
    if converted_path != output_path and converted_path.exists():
        converted_path.replace(output_path)
    if not output_path.exists():
        raise RuntimeError(f"LibreOffice did not create expected PDF: {output_path}")
    log(f"created PDF: {output_path}", verbose=verbose)
    return output_path


def find_libreoffice_converter() -> str | None:
    env_candidates = [
        os.getenv("SOFFICE_PATH"),
        os.getenv("LIBREOFFICE_PATH"),
    ]
    path_candidates = [
        shutil.which("soffice"),
        shutil.which("libreoffice"),
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    ]

    for candidate in [*env_candidates, *path_candidates]:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if candidate_path.exists() and candidate_path.is_file():
            return str(candidate_path)
        if shutil.which(candidate):
            return str(shutil.which(candidate))
    return None


def convert_text_to_pdf(source_path: Path, output_path: Path, *, verbose: bool = True) -> Path:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise RuntimeError(
            "Text to PDF conversion requires reportlab. Install dependencies or run: pip install reportlab"
        ) from exc

    log(f"converting text file to PDF: {source_path}", verbose=verbose)
    text = source_path.read_text(encoding="utf-8", errors="ignore")
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    page_width, page_height = A4
    margin = 48
    line_height = 15
    max_chars = 48

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output_path), pagesize=A4)
    pdf.setFont("STSong-Light", 10)
    y = page_height - margin

    for raw_line in text.splitlines() or [""]:
        line = raw_line.rstrip()
        chunks = [line[i : i + max_chars] for i in range(0, len(line), max_chars)] or [""]
        for chunk in chunks:
            if y < margin:
                pdf.showPage()
                pdf.setFont("STSong-Light", 10)
                y = page_height - margin
            pdf.drawString(margin, y, chunk)
            y -= line_height

    pdf.save()
    log(f"created PDF: {output_path}", verbose=verbose)
    return output_path


def read_document_text(path: Path, *, verbose: bool = True) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        log(f"reading parsed txt: {path}", verbose=verbose)
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        return read_pdf_text(path, verbose=verbose)
    raise ValueError(f"Dataset generation only reads parsed txt or PDF sources, got: {path}")


def read_pdf_text(path: Path, *, verbose: bool = True) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "Reading PDF files requires pypdf. Install dependencies or run: pip install pypdf"
        ) from exc

    log(f"opening PDF with pypdf: {path}", verbose=verbose)
    started = time.perf_counter()
    reader = PdfReader(str(path))
    page_count = len(reader.pages)
    log(f"PDF opened, page count: {page_count}", verbose=verbose)
    pages: list[str] = []
    for page_index, page in enumerate(reader.pages, start=1):
        if page_index == 1 or page_index == page_count or page_index % 10 == 0:
            log(f"extracting PDF page {page_index}/{page_count}", verbose=verbose)
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[page {page_index}]\n{text.strip()}")
    extracted_text = "\n\n".join(pages)
    log(
        f"PDF text extraction finished in {time.perf_counter() - started:.1f}s, "
        f"chars={len(extracted_text)}",
        verbose=verbose,
    )
    return extracted_text


def trim_document_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= max_chars:
        return text

    head_chars = max_chars * 2 // 3
    tail_chars = max_chars - head_chars
    return (
        text[:head_chars]
        + "\n\n[文档过长，中间部分已省略，以下为文档末尾内容]\n\n"
        + text[-tail_chars:]
    )


def generate_document_items(
    llm: LLMClient,
    *,
    document_name: str,
    source_kind: str,
    document_text: str,
    content_length: int,
    target_count: int,
    model: str | None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    content = USER_PROMPT_TEMPLATE.format(
        document_name=document_name,
        source_kind=source_kind,
        content_length=content_length,
        target_count=target_count,
        document_text=document_text,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    started = time.perf_counter()
    try:
        log("sending LLM request with JSON response_format", verbose=verbose)
        reply = call_generation_llm_with_retries(
            llm,
            messages,
            model=model,
            temperature=0.4,
            max_tokens=6000,
            extra_body=build_generation_extra_body(
                llm,
                model=model,
                response_format=True,
            ),
            verbose=verbose,
        )
    except LLMAPIError as exc:
        if not looks_like_response_format_error(exc):
            log(f"LLM request failed before JSON fallback: {exc}", verbose=verbose)
            raise
        log(
            f"LLM JSON-mode request failed, retrying without response_format: {exc}",
            verbose=verbose,
        )
        reply = call_generation_llm_with_retries(
            llm,
            messages,
            model=model,
            temperature=0.4,
            max_tokens=6000,
            extra_body=build_generation_extra_body(
                llm,
                model=model,
                response_format=False,
            ),
            verbose=verbose,
        )
    log(
        f"LLM response received in {time.perf_counter() - started:.1f}s, "
        f"chars={len(reply)}",
        verbose=verbose,
    )
    data = parse_json_object(reply)
    items = data.get("items", [])
    if not isinstance(items, list):
        raise ValueError(f"LLM JSON field 'items' is not a list for {document_name}")
    return [item for item in items if isinstance(item, dict)]


def call_generation_llm_with_retries(
    llm: LLMClient,
    messages: list[dict[str, Any]],
    *,
    model: str | None,
    temperature: float,
    max_tokens: int,
    extra_body: dict[str, Any],
    verbose: bool = True,
) -> str:
    for attempt in range(1, GENERATION_MAX_ATTEMPTS + 1):
        try:
            return llm.chat(
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
        except LLMAPIError as exc:
            if attempt >= GENERATION_MAX_ATTEMPTS or not is_llm_timeout_error(exc):
                raise
            sleep_seconds = GENERATION_RETRY_BASE_SECONDS * attempt
            log(
                f"LLM request timed out; retrying attempt {attempt + 1}/{GENERATION_MAX_ATTEMPTS} "
                f"after {sleep_seconds:.0f}s: {exc}",
                verbose=verbose,
            )
            time.sleep(sleep_seconds)
    raise RuntimeError("unreachable")


def build_generation_extra_body(
    llm: LLMClient,
    *,
    model: str | None,
    response_format: bool,
) -> dict[str, Any]:
    selected_model = model or llm.settings.model
    extra: dict[str, Any] = {}
    if response_format:
        extra["response_format"] = {"type": "json_object"}
    if is_kimi_thinking_model(selected_model, base_url=llm.settings.base_url):
        extra["thinking"] = {"type": "disabled"}
    return extra


def is_kimi_thinking_model(model: str, *, base_url: str) -> bool:
    normalized_model = model.strip().lower()
    normalized_base_url = base_url.strip().lower()
    return normalized_model.startswith("kimi-") or "moonshot" in normalized_base_url


def is_llm_timeout_error(exc: LLMAPIError) -> bool:
    message = str(exc).lower()
    return "timed out" in message or "timeout" in message


def looks_like_response_format_error(exc: LLMAPIError) -> bool:
    message = str(exc).lower()
    return (
        "response_format" in message
        or "json_object" in message
        or "json mode" in message
    )


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if match is None:
            raise
        data = json.loads(match.group(0))

    if not isinstance(data, dict):
        raise ValueError("LLM response JSON must be an object.")
    return data


def normalize_item(
    item: dict[str, Any],
    *,
    source: DocumentSource,
    document_root: Path,
    generated_at: str,
    item_index: int,
    generator_model: str,
    source_char_count: int,
) -> dict[str, Any] | None:
    question = str(item.get("question", "")).strip()
    reference_answer = str(item.get("reference_answer", "")).strip()
    if not question or not reference_answer:
        return None

    raw_types = item.get("question_types", [])
    if isinstance(raw_types, str):
        question_types = [raw_types]
    elif isinstance(raw_types, list):
        question_types = [str(value).strip() for value in raw_types if str(value).strip()]
    else:
        question_types = []

    relative_path = relative_to_root(source.original_path, document_root)
    source_path = relative_to_root(source.source_path, document_root)
    item_id = stable_item_id(relative_path, item_index, question)
    return {
        "id": item_id,
        "document_name": source.original_path.name,
        "document_path": relative_path,
        "source_document_name": source.source_path.name,
        "source_document_path": source_path,
        "source_kind": source.source_kind,
        "question_types": question_types,
        "question": question,
        "reference_answer": reference_answer,
        "evidence": str(item.get("evidence", "")).strip(),
        "difficulty": str(item.get("difficulty", "medium")).strip() or "medium",
        "source_char_count": source_char_count,
        "generator_model": generator_model,
        "generated_at": generated_at,
    }


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def stable_item_id(document_path: str, index: int, question: str) -> str:
    digest = hashlib.sha1(f"{document_path}:{index}:{question}".encode("utf-8")).hexdigest()
    return f"qa_{digest[:16]}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a QA evaluation dataset from documents.")
    parser.add_argument("input_dir", help="Folder containing source documents.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_FILE),
        help=f"Dataset output JSON file. Default: {DEFAULT_OUTPUT_FILE}",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=5,
        help="Approximate number of QA pairs per document.",
    )
    parser.add_argument(
        "--max-document-chars",
        type=int,
        default=300000,
        help="Maximum document characters sent to the generator model.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override LLM model name. Defaults to KIMI_MODEL/LLM_MODEL or kimi-k2.6.",
    )
    parser.add_argument(
        "--pdf-dir",
        default=str(DEFAULT_PDF_SOURCE_DIR),
        help=(
            "Folder used to find or store PDF versions of source documents. "
            f"Default: {DEFAULT_PDF_SOURCE_DIR}"
        ),
    )
    parser.add_argument(
        "--docx-pdf-converter",
        choices=sorted(DOCX_PDF_CONVERTERS),
        default="auto",
        help=(
            "DOCX to PDF converter. 'auto' uses Microsoft Word on Windows first, "
            "then falls back to LibreOffice. Default: auto."
        ),
    )
    parser.add_argument(
        "--rebuild-pdf",
        action="store_true",
        help="Rebuild parsed txt sources instead of reusing existing data/processing txt files.",
    )
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Only convert docx files to PDF and do not call the dataset generation model.",
    )
    parser.add_argument(
        "--pdf-only",
        action="store_true",
        help="Only use PDF files from input_dir and ignore docx/text files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and read PDF sources only; do not call the LLM and do not write dataset JSON.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verbose = not args.quiet
    if args.convert_only:
        converted_files = convert_docx_folder_to_pdf(
            args.input_dir,
            args.pdf_dir,
            converter=args.docx_pdf_converter,
            rebuild=args.rebuild_pdf,
            verbose=verbose,
        )
        log(f"converted {len(converted_files)} docx files to PDF under {args.pdf_dir}", verbose=verbose)
        return

    items = build_dataset(
        args.input_dir,
        args.output,
        target_count=args.target_count,
        max_document_chars=args.max_document_chars,
        model=args.model,
        pdf_dir=args.pdf_dir,
        docx_pdf_converter=args.docx_pdf_converter,
        rebuild_pdf=args.rebuild_pdf,
        verbose=verbose,
        pdf_only=args.pdf_only,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        log(f"wrote {len(items)} QA items to {args.output}", verbose=verbose)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


STANDARD_VERSION = "ASME2023"
CREATE_BY_CODE = "code"
CREATE_BY_MANUAL = "manual"

TAG_RE = re.compile(r"^\[paragraph\]\s+\[(?P<tag>[^\]]+)\]\s+(?P<text>.*)$")
TOP_STANDARD_RE = re.compile(r"^(?P<code>S[AB]-\d+[A-Z]?(?:/S[AB]-\d+[A-Z]?M?)?)\s+-\s+(?P<title>.+?)\s+(?P=code)$")
REFERENCE_CODE_RE = re.compile(
    r"\b(?P<code>(?:ASTM\s+)?[AB]\s*-?\s*\d{1,4}[A-Z]?(?:\s*/\s*[AB]?\s*-?\s*\d{1,4}[A-Z]?M?)?|"
    r"(?:(?:ASME|ANSI)\s+)?B\d{1,2}(?:\.\d+)?|"
    r"API-?\d{3,4}|MSS\s+SP\s*\d+|SP\s*\d+|EN\s*\d{4,5}(?:-\d+)?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SectionDoc:
    volume_dir: Path
    section_dir: Path
    txt_path: Path
    pdf_path: Path | None
    json_path: Path | None
    standard_code: str
    standard_title: str


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def clean_name(value: str) -> str:
    value = value.replace("\ufb01", "fi").replace("\ufb02", "fl")
    return re.sub(r"\s+", " ", value).strip(" -\t\r\n")


def node_id(label: str, key: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    return f"{label.lower()}:{normalized}"


def slugify(value: str) -> str:
    value = clean_name(value).lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]


def table_slug(value: str) -> str:
    value = clean_name(value)
    value = re.sub(r"^(table|figure)\s*-\s*", "", value, flags=re.IGNORECASE)
    return slugify(value)


def edge_id(edge_type: str, source_id: str, target_id: str, extra: str = "") -> str:
    raw = f"{edge_type}|{source_id}|{target_id}|{extra}"
    normalized = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    return f"edge:{normalized}"


def normalize_standard_code(code: str) -> str:
    raw = clean_name(code).upper()
    dimensional = re.match(r"^(ASME|ANSI)\s+B\s*(\d{1,2}(?:\.\d+)?)$", raw)
    if dimensional:
        return f"{dimensional.group(1)} B{dimensional.group(2)}"
    text = raw.replace("SPECIFICATION ", "")
    text = re.sub(r"^(ASTM)\s+", "", text)
    text = re.sub(r"\s+", "", text)
    text = text.replace("_", "/")
    text = re.sub(r"^([AB])(\d)", r"\1-\2", text)
    text = re.sub(r"/([AB])(\d)", r"/\1-\2", text)
    text = re.sub(r"^S([AB])(\d)", r"S\1-\2", text)
    text = re.sub(r"/S([AB])(\d)", r"/S\1-\2", text)
    return text


def asme_equivalent(code: str) -> str:
    normalized = normalize_standard_code(code)
    if normalized.startswith(("ASME B", "ANSI B", "EN", "API", "MSS", "SP ")):
        return normalized
    if re.match(r"^A-\d{1,4}[A-Z]?(?:/A-\d{1,4}[A-Z]?M?)?$", normalized):
        return "/".join("S" + part if not part.startswith("S") else part for part in normalized.split("/"))
    if re.match(r"^B-\d{3,4}[A-Z]?(?:/B-\d{3,4}[A-Z]?M?)?$", normalized):
        return "/".join("S" + part if not part.startswith("S") else part for part in normalized.split("/"))
    if "/A-" in normalized or "/B-" in normalized:
        return re.sub(r"(^|/)([AB]-)", r"\1S\2", normalized)
    return normalized


def standard_aliases(asme_code: str) -> list[str]:
    code = normalize_standard_code(asme_code)
    aliases: set[str] = {code, code.replace("/", "_")}
    if code.startswith(("SA-", "SB-")):
        parts = code.split("/")
        astm_parts = [part[1:] if part.startswith(("SA-", "SB-")) else part for part in parts]
        astm = "/".join(astm_parts)
        aliases.update({astm, f"ASTM {astm}", astm.replace("-", ""), f"ASTM {astm.replace('-', '')}"})
        for part in parts:
            aliases.add(part)
            aliases.add(part.replace("-", ""))
            astm_part = part[1:] if part.startswith(("SA-", "SB-")) else part
            aliases.update({astm_part, astm_part.replace("-", ""), f"ASTM {astm_part}", f"ASTM {astm_part.replace('-', '')}"})
    return sorted(aliases)


def alias_lookup_keys(aliases: list[str]) -> set[str]:
    keys = set()
    for alias in aliases:
        keys.add(normalize_standard_code(alias))
        keys.add(asme_equivalent(alias))
        keys.add(re.sub(r"[^A-Z0-9]+", "", alias.upper()))
    return keys


def parse_section_dir(section_dir: Path) -> tuple[str, str]:
    name = section_dir.name
    match = TOP_STANDARD_RE.match(name)
    if match:
        return normalize_standard_code(match.group("code").replace("_", "/")), clean_name(match.group("title"))
    before_dash, _, after_dash = name.partition(" - ")
    code = normalize_standard_code(before_dash.replace("_", "/"))
    title = clean_name(after_dash)
    if title.endswith(before_dash):
        title = clean_name(title[: -len(before_dash)])
    return code, title


def find_section_docs(root: Path) -> list[SectionDoc]:
    docs: list[SectionDoc] = []
    for volume_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for section_dir in sorted(p for p in volume_dir.iterdir() if p.is_dir()):
            txt_files = sorted((section_dir / "txt").glob("*.txt"))
            if not txt_files:
                continue
            txt_path = txt_files[0]
            pdf_files = sorted((section_dir / "pdf").glob("*.pdf"))
            json_path = section_dir / "json" / "referenced_documents.json"
            code, title = parse_section_dir(section_dir)
            docs.append(
                SectionDoc(
                    volume_dir=volume_dir,
                    section_dir=section_dir,
                    txt_path=txt_path,
                    pdf_path=pdf_files[0] if pdf_files else None,
                    json_path=json_path if json_path.exists() else None,
                    standard_code=code,
                    standard_title=title,
                )
            )
    return docs


def make_node(label: str, key: str, name: str, create_at: str, create_by: str, **properties: Any) -> dict[str, Any]:
    return {
        "id": node_id(label, key),
        "label": label,
        "name": name,
        "create_at": create_at,
        "create_by": create_by,
        "properties": {k: v for k, v in properties.items() if v not in (None, "", [], {})},
    }


def make_edge(edge_type: str, source_id: str, target_id: str, create_at: str, create_by: str, **properties: Any) -> dict[str, Any]:
    return {
        "id": edge_id(edge_type, source_id, target_id, json.dumps(properties, ensure_ascii=False, sort_keys=True)),
        "type": edge_type,
        "source_id": source_id,
        "target_id": target_id,
        "create_at": create_at,
        "create_by": create_by,
        "properties": {k: v for k, v in properties.items() if v not in (None, "", [], {})},
    }


def extract_sections(txt_path: Path) -> list[str]:
    titles2: list[str] = []
    titles3: list[str] = []
    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = TAG_RE.match(line)
        if not match:
            continue
        tag = match.group("tag").strip()
        text = clean_name(match.group("text"))
        if tag == "标题 3":
            titles3.append(text)
        elif tag == "标题 2":
            titles2.append(text)
    selected = titles3 if titles3 else titles2
    seen: set[str] = set()
    result: list[str] = []
    for title in selected:
        key = title.casefold()
        if key not in seen:
            seen.add(key)
            result.append(title)
    return result


def extract_reference_codes(json_path: Path) -> list[str]:
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    text = clean_name(str(payload.get("text", "")))
    codes: list[str] = []
    seen: set[str] = set()
    for match in REFERENCE_CODE_RE.finditer(text):
        raw = clean_name(match.group("code"))
        normalized = normalize_standard_code(raw)
        if not normalized or normalized in {"A-", "B-"}:
            continue
        key = normalized.casefold()
        if key not in seen:
            seen.add(key)
            codes.append(normalized)
    return codes


PRODUCT_STANDARD_MAP: dict[str, list[dict[str, Any]]] = {
    "seamless_pipe": [
        {"standard": "ASTM A999", "context": "通用标准"},
        {"standard": "ASTM A312", "context": "材质标准 / 不锈钢"},
        {"standard": "ASTM A790", "context": "材质标准 / 双相钢"},
        {"standard": "ASTM B161", "context": "材质标准 / Nickel 200/201"},
        {"standard": "ASTM B165", "context": "材质标准 / Monel 400"},
        {"standard": "ASTM B167", "context": "材质标准 / Inconel 600/601/617"},
        {"standard": "ASTM B407", "context": "材质标准 / Incoloy 800/800H/800HT"},
        {"standard": "ASTM B423", "context": "材质标准 / Incoloy 825"},
        {"standard": "ASTM B444", "context": "材质标准 / Inconel 625"},
        {"standard": "ASTM B622", "context": "材质标准 / C276/C22/哈B"},
        {"standard": "ASME B36.19", "context": "尺寸标准"},
    ],
    "seamless_tube": [
        {"standard": "ASTM A1016", "context": "通用标准"},
        {"standard": "ASTM A213", "context": "材质标准 / 美标不锈钢"},
        {"standard": "ASTM A269", "context": "材质标准"},
        {"standard": "EN 10216-5", "context": "材质标准 / 欧标不锈钢"},
        {"standard": "ASTM A789", "context": "材质标准 / 双相钢"},
        {"standard": "ASTM B163", "context": "材质标准 / Monel 400"},
        {"standard": "ASTM B407", "context": "材质标准 / Incoloy 800/800H/800HT"},
        {"standard": "ASTM B423", "context": "材质标准 / Incoloy 825"},
        {"standard": "ASTM B444", "context": "材质标准 / Inconel 625"},
    ],
    "welded_pipe": [
        {"standard": "ASTM A312", "context": "材质标准 / 不锈钢"},
        {"standard": "ASTM A358", "context": "材质标准 / 仅焊管 pipe"},
        {"standard": "ASTM A790", "context": "材质标准 / 双相钢"},
        {"standard": "ASTM A928", "context": "材质标准 / 仅焊管 pipe"},
        {"standard": "ASTM B725", "context": "材质标准 / Nickel 200/201, Monel 400"},
        {"standard": "ASTM B517", "context": "材质标准 / Inconel 600"},
        {"standard": "ASTM B514", "context": "材质标准 / Incoloy 800/800H"},
        {"standard": "ASTM B423", "context": "材质标准 / Incoloy 825"},
        {"standard": "ASTM B705", "context": "材质标准 / Inconel 625"},
        {"standard": "ASTM B619", "context": "材质标准 / C276/C22/哈B"},
        {"standard": "ANSI B36.19", "context": "尺寸标准"},
    ],
    "welded_tube": [
        {"standard": "ASTM A249", "context": "材质标准 / 仅焊管 tube"},
        {"standard": "ASTM A269", "context": "材质标准"},
        {"standard": "ASTM A789", "context": "材质标准"},
        {"standard": "ASTM B163", "context": "材质标准"},
        {"standard": "EN 10216-5", "context": "材质标准"},
    ],
    "butt_weld_fitting_forged": [
        {"standard": "ASTM A403", "context": "材质标准 / 不锈钢"},
        {"standard": "ASTM A815", "context": "材质标准 / 双相钢"},
        {"standard": "ASTM B366", "context": "材质标准 / 镍合金"},
        {"standard": "ASME B16.9", "context": "尺寸标准"},
    ],
    "low_pressure_fitting_cast": [
        {"standard": "ASTM A351", "context": "材质标准"},
    ],
    "high_pressure_fitting_forged": [
        {"standard": "ASTM A182", "context": "材质标准"},
        {"standard": "ASME B16.11", "context": "尺寸标准"},
    ],
    "flange": [
        {"standard": "ASTM A182", "context": "材质标准 / 不锈钢/双相钢"},
        {"standard": "ASTM B564", "context": "材质标准 / 镍合金"},
        {"standard": "ASME B16.5", "context": "尺寸标准"},
    ],
    "plate": [
        {"standard": "ASTM A240", "context": "材质标准"},
        {"standard": "ASTM A480", "context": "通用标准"},
    ],
    "bar": [
        {"standard": "ASTM A182", "context": "材质标准"},
    ],
}


PRODUCT_NAMES = {
    "seamless_pipe": "Seamless Pipe",
    "seamless_tube": "Seamless Tube",
    "welded_pipe": "Welded Pipe",
    "welded_tube": "Welded Tube",
    "butt_weld_fitting_forged": "Butt Weld Fitting (Forged)",
    "low_pressure_fitting_cast": "Low Pressure Fitting (Cast)",
    "high_pressure_fitting_forged": "High Pressure Fitting (Forged)",
    "flange": "Flange",
    "plate": "Plate",
    "bar": "Bar",
}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_manifest(processing_root: Path, output_dir: Path) -> None:
    create_at = now_iso()
    docs = find_section_docs(processing_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}
    standard_alias_index: dict[str, str] = {}

    version = make_node("StandardVersion", STANDARD_VERSION, STANDARD_VERSION, create_at, CREATE_BY_CODE)
    nodes[version["id"]] = version

    def add_standard(code: str, title: str = "", source: str = "asme_section") -> str:
        asme_code = asme_equivalent(code)
        aliases = standard_aliases(asme_code)
        keys = alias_lookup_keys(aliases + [code])
        existing_id = next((standard_alias_index[k] for k in keys if k in standard_alias_index), None)
        if existing_id:
            existing = nodes[existing_id]
            existing_aliases = set(existing["properties"].get("aliases", []))
            existing_aliases.update(aliases)
            existing["properties"]["aliases"] = sorted(existing_aliases)
            if title and not existing["properties"].get("standard_title"):
                existing["properties"]["standard_title"] = title
            return existing_id
        std = make_node(
            "Standard",
            asme_code,
            asme_code,
            create_at,
            CREATE_BY_CODE,
            code=asme_code,
            standard_title=title,
            standard_version=STANDARD_VERSION if asme_code.startswith(("SA-", "SB-")) else None,
            aliases=aliases,
            source=source,
        )
        nodes[std["id"]] = std
        for key in keys:
            standard_alias_index[key] = std["id"]
        edge = make_edge("has_version", std["id"], version["id"], create_at, CREATE_BY_CODE)
        edges[edge["id"]] = edge
        return std["id"]

    volume_doc_ids: dict[Path, str] = {}
    for doc in docs:
        volume_doc = volume_doc_ids.get(doc.volume_dir)
        if not volume_doc:
            raw_pdf = Path("data/raw/产品标准") / f"{doc.volume_dir.name}.pdf"
            volume_node = make_node(
                "Document",
                doc.volume_dir.name,
                doc.volume_dir.name,
                create_at,
                CREATE_BY_CODE,
                file_path=str(raw_pdf if raw_pdf.exists() else doc.volume_dir),
                document_level="volume",
            )
            nodes[volume_node["id"]] = volume_node
            volume_doc = volume_node["id"]
            volume_doc_ids[doc.volume_dir] = volume_doc

        standard_id = add_standard(doc.standard_code, doc.standard_title)
        section_doc_node = make_node(
            "Document",
            str(doc.section_dir),
            doc.section_dir.name,
            create_at,
            CREATE_BY_CODE,
            file_path=str(doc.pdf_path or doc.txt_path),
            txt_path=str(doc.txt_path),
            document_level="sub_document",
            source_volume=doc.volume_dir.name,
        )
        nodes[section_doc_node["id"]] = section_doc_node
        for edge in [
            make_edge("has_sub_document", volume_doc, section_doc_node["id"], create_at, CREATE_BY_CODE),
            make_edge("is_about", section_doc_node["id"], standard_id, create_at, CREATE_BY_CODE),
        ]:
            edges[edge["id"]] = edge

        for section_title in extract_sections(doc.txt_path):
            section_name = slugify(section_title)
            section_node = make_node(
                "Section",
                f"{section_name}_{short_hash(str(doc.txt_path))}",
                section_name,
                create_at,
                CREATE_BY_CODE,
                title=section_title,
                file_path=str(doc.txt_path),
            )
            nodes[section_node["id"]] = section_node
            edge = make_edge("has_section", section_doc_node["id"], section_node["id"], create_at, CREATE_BY_CODE)
            edges[edge["id"]] = edge

        img_dir = doc.section_dir / "img"
        if img_dir.exists():
            for table_path in sorted(p for p in img_dir.iterdir() if p.is_file()):
                table_name = table_slug(table_path.stem)
                table_node = make_node(
                    "Table",
                    f"{table_name}_{short_hash(str(table_path))}",
                    table_name,
                    create_at,
                    CREATE_BY_CODE,
                    title=table_path.stem,
                    file_path=str(table_path),
                )
                nodes[table_node["id"]] = table_node
                edge = make_edge("has_table", section_doc_node["id"], table_node["id"], create_at, CREATE_BY_CODE)
                edges[edge["id"]] = edge

        if doc.json_path:
            for ref_code in extract_reference_codes(doc.json_path):
                ref_id = add_standard(ref_code, source="referenced_document")
                edge = make_edge(
                    "reference_to",
                    standard_id,
                    ref_id,
                    create_at,
                    CREATE_BY_CODE,
                    source_file=str(doc.json_path),
                    raw_reference_code=ref_code,
                )
                edges[edge["id"]] = edge

    product_source_path = Path("data/raw/产品知识框架讲解.docx")
    product_extract = {
        "source_file": str(product_source_path),
        "create_at": create_at,
        "create_by": CREATE_BY_MANUAL,
        "note": "Extracted from the single image embedded in 产品知识框架讲解.docx.",
        "products": PRODUCT_STANDARD_MAP,
    }
    (output_dir / "product_standard_extract.json").write_text(
        json.dumps(product_extract, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for product_key, refs in PRODUCT_STANDARD_MAP.items():
        product_node = make_node(
            "Product",
            product_key,
            PRODUCT_NAMES[product_key],
            create_at,
            CREATE_BY_MANUAL,
            source_file=str(product_source_path),
        )
        nodes[product_node["id"]] = product_node
        for ref in refs:
            standard_id = add_standard(ref["standard"], source="product_framework")
            edge = make_edge(
                "apply_to",
                product_node["id"],
                standard_id,
                create_at,
                CREATE_BY_MANUAL,
                source_file=str(product_source_path),
                source_context=ref["context"],
                raw_standard=ref["standard"],
            )
            edges[edge["id"]] = edge

    node_rows = sorted(nodes.values(), key=lambda row: (row["label"], row["id"]))
    edge_rows = sorted(edges.values(), key=lambda row: (row["type"], row["id"]))
    write_jsonl(output_dir / "nodes.jsonl", node_rows)
    write_jsonl(output_dir / "edges.jsonl", edge_rows)

    counts_by_label: dict[str, int] = {}
    counts_by_type: dict[str, int] = {}
    for row in node_rows:
        counts_by_label[row["label"]] = counts_by_label.get(row["label"], 0) + 1
    for row in edge_rows:
        counts_by_type[row["type"]] = counts_by_type.get(row["type"], 0) + 1

    summary = {
        "create_at": create_at,
        "graph_name": "MTSCO知识图谱",
        "subgraph_name": "standard_product_subgraph",
        "source_root": str(processing_root),
        "node_count": len(node_rows),
        "edge_count": len(edge_rows),
        "node_counts_by_label": counts_by_label,
        "edge_counts_by_type": counts_by_type,
        "section_documents": len(docs),
        "outputs": {
            "nodes": str(output_dir / "nodes.jsonl"),
            "edges": str(output_dir / "edges.jsonl"),
            "product_extract": str(output_dir / "product_standard_extract.json"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(render_readme(summary), encoding="utf-8")


def render_readme(summary: dict[str, Any]) -> str:
    node_lines = "\n".join(f"- {k}: {v}" for k, v in sorted(summary["node_counts_by_label"].items()))
    edge_lines = "\n".join(f"- {k}: {v}" for k, v in sorted(summary["edge_counts_by_type"].items()))
    return f"""# Standard Product Subgraph Manifest

Graph: MTSCO知识图谱

This directory contains the candidate node and edge list for the standard-reference-product subgraph. It is a review manifest only; nothing is written to Neo4j in this step.

## Outputs

- `nodes.jsonl`: candidate nodes.
- `edges.jsonl`: candidate edges.
- `product_standard_extract.json`: manual extraction from `data/raw/产品知识框架讲解.docx`.
- `summary.json`: machine-readable counts and paths.
- `summary.json` also declares `subgraph_name`; `scripts/kg/import_graph_manifest.py` reads it as the default `sub_graph_name` property during Neo4j import.

## Node Counts

{node_lines}

## Edge Counts

{edge_lines}

## Notes

- Standard nodes are merge-oriented. ASME `SA-*`/`SB-*` and ASTM `A*`/`B*` aliases are placed on the same `Standard` candidate where possible.
- `Section` and `Table` node names are short slugs, with an id hash and `file_path` metadata to avoid accidental merging across sub-documents.
- `create_by` is `code` for rule extraction and `manual` for the product framework image extraction.
- Neo4j import adds `graph_name` and `sub_graph_name` to every imported node and relationship. Override the latter with `--sub-graph-name` only when importing the same files into a different logical subgraph.
- `has_version` is included to connect `Standard` to `StandardVersion`, although the first Neo4j import script can omit it if you decide to keep version as metadata only.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build candidate graph manifest for ASME standard product subgraph.")
    parser.add_argument("--processing-root", default="data/processing/产品标准", type=Path)
    parser.add_argument("--output-dir", default="data/metadata/graph/standard_product_subgraph", type=Path)
    args = parser.parse_args()
    build_manifest(args.processing_root, args.output_dir)
    print(f"Wrote graph manifest to {args.output_dir}")


if __name__ == "__main__":
    main()

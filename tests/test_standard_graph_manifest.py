from __future__ import annotations

import json
from pathlib import Path

from app.db.minio import build_minio_uri, parse_raw_document_reference
from scripts.kg.build_standard_graph_manifest import build_manifest


def test_minio_reference_preserves_encoded_hash_in_object_name() -> None:
    uri = build_minio_uri(
        "knowledge-standard-assets",
        "产品标准/SA-1/table # 1.png",
    )

    reference = parse_raw_document_reference(uri)

    assert reference.bucket == "knowledge-standard-assets"
    assert reference.object_name == "产品标准/SA-1/table # 1.png"


def test_graph_manifest_uses_published_image_uri_for_table_node(
    tmp_path: Path,
) -> None:
    processing_root = tmp_path / "processing" / "产品标准"
    volume_dir = processing_root / "ASME-demo"
    section_dir = volume_dir / "SA-1 - Demo"
    txt_path = section_dir / "txt" / "SA-1 - Demo.txt"
    pdf_path = section_dir / "pdf" / "SA-1 - Demo.pdf"
    image_path = section_dir / "img" / "table - TABLE 1 Demo.png"
    txt_path.parent.mkdir(parents=True)
    pdf_path.parent.mkdir(parents=True)
    image_path.parent.mkdir(parents=True)
    txt_path.write_text("[paragraph] [标题 2] 1 Scope", encoding="utf-8")
    pdf_path.write_bytes(b"pdf")
    image_path.write_bytes(b"png")

    section_uri = build_minio_uri(
        "knowledge-raw-docs",
        "产品标准/ASME-demo(切分版)/SA-1 - Demo.pdf",
    )
    image_uri = build_minio_uri(
        "knowledge-standard-assets",
        "产品标准/ASME-demo(切分版)/SA-1 - Demo/table - TABLE 1 Demo.png",
    )
    (volume_dir / "manifest.json").write_text(
        json.dumps(
            {
                "sections": [
                    {
                        "index": 1,
                        "title": "Demo",
                        "standard_code": "SA-1",
                        "start_page": 1,
                        "end_page": 1,
                        "output_path": str(pdf_path),
                        "section_dir": str(section_dir),
                        "source_uri": section_uri,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (volume_dir / "assets_manifest.json").write_text(
        json.dumps(
            {
                "asset_bucket": "knowledge-standard-assets",
                "assets": [
                    {
                        "index": 1,
                        "asset_type": "table",
                        "caption": "TABLE 1 Demo",
                        "section_index": 1,
                        "standard_code": "SA-1",
                        "section_page": 1,
                        "source_page": 1,
                        "bbox": [0, 0, 10, 10],
                        "image_path": str(image_path),
                        "source_uri": image_uri,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "graph"
    build_manifest(
        processing_root,
        output_dir,
        publish_assets=False,
        verify_assets=False,
    )

    nodes = [json.loads(line) for line in (output_dir / "nodes.jsonl").read_text(encoding="utf-8").splitlines()]
    table = next(node for node in nodes if node["label"] == "Table")
    document = next(
        node
        for node in nodes
        if node["label"] == "Document" and node["properties"].get("document_level") == "sub_document"
    )
    assert table["properties"]["file_path"] == image_uri
    assert document["properties"]["file_path"] == section_uri
    products = {node["id"]: node for node in nodes if node["label"] == "Product"}
    assert len(products) == 11
    assert "无缝管" in products["product:seamless_pipe"]["properties"]["aliases"]
    assert "无缝换热管" in products["product:seamless_tube"]["properties"]["aliases"]
    assert products["product:plate"]["properties"]["category"] == "板材&棒材"
    assert products["product:wire"]["properties"]["chinese_name"] == "线材"

from pathlib import Path

from scripts.ingestion.sync_standard_text_assets import standard_text_object_name


def test_standard_text_object_name_matches_existing_asset_layout(tmp_path: Path) -> None:
    root = tmp_path / "产品标准"
    path = root / "ASME-Sec-II-A-Vol1-2023" / "SA-213 demo" / "txt" / "SA-213 demo.txt"
    assert standard_text_object_name(path, root) == (
        "产品标准/ASME-Sec-II-A-Vol1-2023(切分版)/SA-213 demo/SA-213 demo.txt"
    )

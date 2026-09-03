"""Validate Feishu knowledge sources, upload valid files to OSS, and register them.

No downloaded document is persisted locally: every validation/download is made
in a temporary directory that is removed at the end of the run.  A document is
inserted into lark_document_catalog only after both download and OSS upload
succeed.  Failures are intentionally recorded as names only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env.host", override=True)

from app.core.config import settings  # noqa: E402
from app.db.postgres import postgres_connection  # noqa: E402
from app.services.lark_client import download_node, get_access_token, sanitize_path_part  # noqa: E402
from app.services.lark_document_catalog import collect_records, ensure_catalog_table, upsert_records  # noqa: E402
from app.services.knowledge_quality import is_acceptable_knowledge_file  # noqa: E402


DEFAULT_SOURCE = PROJECT_ROOT / "data" / "src" / "vector_src.json"
DEFAULT_FAILURE_REPORT = PROJECT_ROOT / "data" / "metadata" / "lark_download_failures.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing_catalog_hashes() -> dict[str, dict[str, Any]]:
    """Return the digest recorded for the exact OSS object of each document."""
    with postgres_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT document_key, content_sha256, oss_object_key, ingested_at, lark_updated_at FROM lark_document_catalog")
        return {
            str(row["document_key"]): {
                "content_sha256": str(row["content_sha256"] or ""),
                "oss_object_key": str(row["oss_object_key"] or ""),
                "ingested_at": row["ingested_at"],
                "lark_updated_at": row["lark_updated_at"],
            }
            for row in cur.fetchall()
        }


def _existing_catalog_document_names() -> set[str]:
    """Return all registered export filenames, for safe failure-list pruning."""
    with postgres_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT document_name FROM lark_document_catalog")
        return {str(row["document_name"]) for row in cur.fetchall()}


def _previous_failure_names(path: Path) -> set[str]:
    """Read a best-effort, name-only failure list written by an earlier run."""
    if not path.is_file():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        values = data.get("failed", []) if isinstance(data, dict) else []
        return {str(value) for value in values if isinstance(value, str) and value}
    except (OSError, json.JSONDecodeError):
        return set()


def _same_lark_update(left: Any, right: Any) -> bool:
    return bool(left and right and left.astimezone(timezone.utc).replace(microsecond=0) == right.astimezone(timezone.utc).replace(microsecond=0))


def _oss_bucket():
    required = {
        "ALIYUN_OSS_ENDPOINT": settings.aliyun_oss_endpoint,
        "ALIYUN_ACCESS_KEY_ID": settings.aliyun_access_key_id,
        "ALIYUN_ACCESS_KEY_SECRET": settings.aliyun_access_key_secret,
        "ALIYUN_RAW_DATA_BUCKET": settings.aliyun_raw_data_bucket,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(f"Missing OSS configuration: {', '.join(missing)}")
    try:
        import oss2
    except ImportError as exc:
        raise RuntimeError(
            "Missing oss2 in this virtual environment. From the project root run: "
            ".venv/bin/python -m pip install 'oss2>=2.19.1,<3.0.0'"
        ) from exc
    return oss2.Bucket(
        oss2.Auth(settings.aliyun_access_key_id, settings.aliyun_access_key_secret),
        settings.aliyun_oss_endpoint,
        settings.aliyun_raw_data_bucket,
    )


def _oss_key(record: dict[str, Any], downloaded: Path, used: set[str]) -> str:
    path_parts = [sanitize_path_part(str(part)) for part in record.get("path_titles") or []]
    # A single_file source-map key declares its target file name explicitly;
    # Wiki leaves retain the export file name returned by Feishu.
    file_name = sanitize_path_part(str(record.get("storage_file_name") or downloaded.name))
    # Source-map keys may omit an extension for readability.  Keep the actual
    # export suffix so the OSS object remains a usable, typed file.
    if not PurePosixPath(file_name).suffix and downloaded.suffix:
        file_name = f"{file_name}{downloaded.suffix}"
    candidate = "/".join(["knowledge", *path_parts, file_name])
    if candidate not in used:
        used.add(candidate)
        return candidate
    token = sanitize_path_part(str(record.get("node_token") or record["document_key"]))[:12]
    path = PurePosixPath(candidate)
    candidate = str(path.with_name(f"{path.stem}_{token}{path.suffix}"))
    used.add(candidate)
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--check-only", action="store_true", help="Download-test only; do not upload or update PostgreSQL.")
    parser.add_argument("--failure-report", type=Path, default=DEFAULT_FAILURE_REPORT)
    args = parser.parse_args()

    mode = "download check" if args.check_only else "download + OSS upload + catalog registration"
    print(f"[start] mode={mode}", flush=True)
    print(f"[scan] reading Feishu sources from {args.source}", flush=True)
    records, scan_failures = collect_records(
        args.source,
        progress=lambda state, source_type, source_name: print(
            f"[scan] {state} {source_type} {source_name}", flush=True
        ),
    )
    prior_failure_names = _previous_failure_names(args.failure_report)
    failed = set(prior_failure_names)
    failed.update(str(item.get("title") or item.get("source_name") or "") for item in scan_failures)
    print(
        f"[scan] candidates={len(records)} scan_failures={len(scan_failures)}",
        flush=True,
    )
    bucket = None if args.check_only else _oss_bucket()
    if bucket is not None:
        # The catalog is the hash cache.  A fresh deployment must create it
        # before reading existing hashes for the unchanged-event fast path.
        ensure_catalog_table()
        print(f"[oss] target_bucket={settings.aliyun_raw_data_bucket}", flush=True)
    uploaded: list[dict[str, Any]] = []
    known = _existing_catalog_hashes() if bucket is not None else {}
    catalog_names = _existing_catalog_document_names() if bucket is not None else set()
    downloadable = 0
    used_keys: set[str] = set()
    access_token = get_access_token()
    print("[download] Feishu access token acquired", flush=True)

    with TemporaryDirectory(prefix="lark-kb-") as temporary:
        download_root = Path(temporary)
        for index, record in enumerate(records, start=1):
            name = str(record["document_name"])
            prior = known.get(str(record["document_key"]))
            # Only use the old name-only report when the catalog has no record
            # with that filename at all.  A same-name catalog entry can be a
            # different Lark document, so it must always be retried.
            if name in prior_failure_names and name not in catalog_names:
                print(f"[{index}/{len(records)}] pruned prior-failure {name}", flush=True)
                continue
            if prior and prior["content_sha256"] and _same_lark_update(record.get("lark_updated_at"), prior["lark_updated_at"]):
                print(f"[{index}/{len(records)}] pruned unchanged-event {name}", flush=True)
                continue
            print(f"[{index}/{len(records)}] download {name}", flush=True)
            try:
                ok, value = download_node(
                    access_token,
                    dict(record["raw_node"]),
                    download_root,
                    used_paths=set(),
                    overwrite=True,
                )
                downloaded = Path(value)
                if not ok or not downloaded.is_file() or downloaded.stat().st_size <= 0:
                    raise RuntimeError("download unavailable")
                accepted, rule = is_acceptable_knowledge_file(downloaded)
                if not accepted:
                    raise RuntimeError(f"quality rejected: {rule}")
                downloadable += 1
                if bucket is None:
                    print(f"[{index}/{len(records)}] ok {name}", flush=True)
                    continue
                key = _oss_key(record, downloaded, used_keys)
                digest = _sha256(downloaded)
                if prior and prior["oss_object_key"] and not prior["content_sha256"]:
                    key = prior["oss_object_key"]
                needs_upload = prior is None or not prior["oss_object_key"] or (prior["content_sha256"] and prior["content_sha256"] != digest)
                if needs_upload:
                    print(f"[{index}/{len(records)}] upload {name}", flush=True)
                    bucket.put_object_from_file(key, str(downloaded))
                uploaded.append(
                    {
                        **record,
                        "oss_object_key": key,
                        "oss_uri": f"oss://{settings.aliyun_raw_data_bucket}/{key}",
                        "content_sha256": digest,
                        "content_size": downloaded.stat().st_size,
                        "ingested_at": datetime.now(timezone.utc) if needs_upload else prior["ingested_at"],
                    }
                )
                print(f"[{index}/{len(records)}] ok {name}", flush=True)
            except Exception:
                failed.add(name)
                print(f"[{index}/{len(records)}] failed {name}", flush=True)

    args.failure_report.parent.mkdir(parents=True, exist_ok=True)
    args.failure_report.write_text(
        json.dumps({"failed": sorted(item for item in failed if item)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[report] failures={len(failed)} file={args.failure_report}", flush=True)
    if args.check_only:
        print(json.dumps({"checked": len(records), "downloadable": downloadable, "failure_report": str(args.failure_report)}, ensure_ascii=False), flush=True)
        return 0

    ensure_catalog_table()
    print(f"[catalog] registering oss_uploaded={len(uploaded)}", flush=True)
    upserted = upsert_records(uploaded)
    print(json.dumps({"checked": len(records), "downloadable": downloadable, "oss_uploaded": upserted, "failure_report": str(args.failure_report)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

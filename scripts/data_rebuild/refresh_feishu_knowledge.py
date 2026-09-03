"""Brute-force scan Feishu knowledge and incrementally refresh OSS and Milvus.

Every candidate is downloaded and hashed.  Only new, changed, or relocated
documents are uploaded to OSS and re-ingested.  The ingestion script uses a
stable Lark document ID, so a successful Milvus upsert replaces old chunks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
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
from app.services.knowledge_quality import is_acceptable_knowledge_file  # noqa: E402
from app.services.lark_client import download_node, get_access_token  # noqa: E402
from app.services.lark_document_catalog import collect_records, ensure_catalog_table, upsert_records  # noqa: E402
from scripts.data_rebuild.sync_feishu_knowledge_to_oss import (  # noqa: E402
    _oss_bucket,
    _oss_key,
    _previous_failure_names,
)

DEFAULT_SOURCE = PROJECT_ROOT / "data" / "src" / "vector_src.json"
DEFAULT_FAILURE_REPORT = PROJECT_ROOT / "data" / "metadata" / "lark_incremental_failures.json"
DEFAULT_CHANGED_KEYS = PROJECT_ROOT / "data" / "metadata" / "lark_incremental_changed_document_keys.txt"
INGEST_SCRIPT = PROJECT_ROOT / "scripts" / "data_rebuild" / "ingest_oss_knowledge.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def existing_catalog() -> dict[str, dict[str, Any]]:
    with postgres_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT document_key, content_sha256, oss_object_key, content_size, ingested_at, lark_updated_at FROM lark_document_catalog")
        return {
            str(row["document_key"]): {
                "content_sha256": str(row["content_sha256"] or ""),
                "oss_object_key": str(row["oss_object_key"] or ""),
                "content_size": row["content_size"],
                "ingested_at": row["ingested_at"],
                "lark_updated_at": row["lark_updated_at"],
            }
            for row in cur.fetchall()
        }


def catalog_document_names() -> set[str]:
    with postgres_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT document_name FROM lark_document_catalog")
        return {str(row["document_name"]) for row in cur.fetchall()}


def same_lark_update(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    return left.astimezone(timezone.utc).replace(microsecond=0) == right.astimezone(timezone.utc).replace(microsecond=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--failure-report", type=Path, default=DEFAULT_FAILURE_REPORT)
    parser.add_argument("--changed-keys-file", type=Path, default=DEFAULT_CHANGED_KEYS)
    parser.add_argument("--image-workers", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="Download, quality-check, and hash only; do not modify OSS, catalog, or Milvus.")
    parser.add_argument("--skip-ingest", action="store_true", help="Update OSS/catalog only; do not invoke incremental vector ingestion.")
    args = parser.parse_args()

    print(f"[scan] source={args.source}", flush=True)
    records, scan_failures = collect_records(
        args.source,
        progress=lambda state, kind, name: print(f"[scan] {state} {kind} {name}", flush=True),
    )
    known = existing_catalog()
    known_names = catalog_document_names()
    bucket = None if args.dry_run else _oss_bucket()
    token = get_access_token()
    catalog_updates: list[dict[str, Any]] = []
    changed: list[dict[str, Any]] = []
    prior_failure_names = _previous_failure_names(PROJECT_ROOT / "data" / "metadata" / "lark_download_failures.json")
    failures = set(prior_failure_names)
    failures.update(str(item.get("title") or item.get("source_name") or "") for item in scan_failures)
    used_keys: set[str] = set()

    with TemporaryDirectory(prefix="lark-refresh-") as temporary:
        temporary_root = Path(temporary)
        for index, record in enumerate(records, 1):
            name = str(record["document_name"])
            prior = known.get(str(record["document_key"]))
            # The old failure report is keyed only by filename.  Prune only
            # when no catalog row has that name, otherwise a different Lark
            # document with the same export name could be skipped incorrectly.
            if name in prior_failure_names and name not in known_names:
                print(f"[{index}/{len(records)}] pruned prior-failure {name}", flush=True)
                continue
            if prior and prior["content_sha256"] and same_lark_update(record.get("lark_updated_at"), prior["lark_updated_at"]):
                print(f"[{index}/{len(records)}] pruned unchanged-event {name}", flush=True)
                continue
            print(f"[{index}/{len(records)}] download+hash {name}", flush=True)
            try:
                ok, value = download_node(token, dict(record["raw_node"]), temporary_root, used_paths=set(), overwrite=True)
                downloaded = Path(value)
                if not ok or not downloaded.is_file() or downloaded.stat().st_size <= 0:
                    raise RuntimeError("download unavailable")
                accepted, rule = is_acceptable_knowledge_file(downloaded)
                if not accepted:
                    raise RuntimeError(f"quality rejected: {rule}")
                digest = sha256(downloaded)
                key = _oss_key(record, downloaded, used_keys)
                # A missing historic hash is a catalog backfill, not evidence
                # of new content.  Do not re-embed solely for that migration.
                content_changed = prior is None or (bool(prior["content_sha256"]) and prior["content_sha256"] != digest)
                path_changed = prior is None or prior["oss_object_key"] != key
                needs_upload = content_changed or path_changed or not prior or not prior["oss_object_key"]
                if bucket is not None and needs_upload:
                    print(f"[{index}/{len(records)}] upload {name}", flush=True)
                    bucket.put_object_from_file(key, str(downloaded))
                catalog_row = {
                    **record,
                    "oss_object_key": key,
                    "oss_uri": f"oss://{settings.aliyun_raw_data_bucket}/{key}",
                    "content_sha256": digest,
                    "content_size": downloaded.stat().st_size,
                    "ingested_at": datetime.now(timezone.utc) if needs_upload else prior["ingested_at"],
                }
                catalog_updates.append(catalog_row)
                if content_changed or path_changed:
                    changed.append(catalog_row)
                    print(f"[{index}/{len(records)}] changed {name}", flush=True)
                else:
                    print(f"[{index}/{len(records)}] hash backfilled {name}", flush=True)
            except Exception:
                failures.add(name)
                print(f"[{index}/{len(records)}] rejected-or-failed {name}", flush=True)

    args.failure_report.parent.mkdir(parents=True, exist_ok=True)
    args.failure_report.write_text(json.dumps({"failed": sorted(item for item in failures if item)}, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"scanned": len(records), "catalog_updates": len(catalog_updates), "changed": len(changed), "failures": len(failures), "dry_run": True}, ensure_ascii=False), flush=True)
        return 0

    ensure_catalog_table()
    upsert_records(catalog_updates)
    args.changed_keys_file.parent.mkdir(parents=True, exist_ok=True)
    args.changed_keys_file.write_text("\n".join(str(row["document_key"]) for row in changed) + ("\n" if changed else ""), encoding="utf-8")
    print(f"[catalog] changed={len(changed)} keys_file={args.changed_keys_file}", flush=True)
    if changed and not args.skip_ingest:
        command = [sys.executable, str(INGEST_SCRIPT), "--document-key-file", str(args.changed_keys_file), "--continue-on-error"]
        if (PROJECT_ROOT / "data" / "processing" / "global.bm25.json").is_file():
            command.append("--use-existing-bm25")
        print("[ingest] starting changed documents", flush=True)
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if completed.returncode:
            return completed.returncode
    print(json.dumps({"scanned": len(records), "changed": len(changed), "failures": len(failures)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

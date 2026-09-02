"""Compatibility entry point for the material path/link catalogue rebuild."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env.host", override=True)

from scripts.storage.build_lark_material_path_mapping import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

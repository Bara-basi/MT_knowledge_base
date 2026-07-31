from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_lark_credentials() -> tuple[str, str]:
    """Load the Lark application credentials from the project environment."""

    load_dotenv(PROJECT_ROOT / ".env")
    app_id = os.getenv("LARK_APP_ID") or os.getenv("FEISHU_APP_ID")
    app_secret = os.getenv("LARK_APP_SECRET") or os.getenv("FEISHU_APP_SECRET")
    missing = [
        name
        for name, value in (
            ("LARK_APP_ID", app_id),
            ("LARK_APP_SECRET", app_secret),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing required Lark credentials in .env: " + ", ".join(missing)
        )
    return app_id, app_secret


def use_local_lark_credentials() -> None:
    """Expose project-local Lark credentials under both supported variable names."""

    app_id, app_secret = get_lark_credentials()

    os.environ["FEISHU_APP_ID"] = app_id
    os.environ["FEISHU_APP_SECRET"] = app_secret
    os.environ["LARK_APP_ID"] = app_id
    os.environ["LARK_APP_SECRET"] = app_secret

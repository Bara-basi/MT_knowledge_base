from __future__ import annotations

import os


DEFAULT_LOCAL_LARK_APP_ID = "cli_aa895b208df9dcdd"
DEFAULT_LOCAL_LARK_APP_SECRET = "AfESVeY5n9m7By2plKh97g05C7TtbCAZ"


def use_local_lark_credentials() -> None:
    """Force selected storage scripts to use the local Lark robot credentials."""

    app_id = os.getenv("LOCAL_LARK_APP_ID", DEFAULT_LOCAL_LARK_APP_ID)
    app_secret = os.getenv("LOCAL_LARK_APP_SECRET", DEFAULT_LOCAL_LARK_APP_SECRET)

    os.environ["FEISHU_APP_ID"] = app_id
    os.environ["FEISHU_APP_SECRET"] = app_secret
    os.environ["LARK_APP_ID"] = app_id
    os.environ["LARK_APP_SECRET"] = app_secret

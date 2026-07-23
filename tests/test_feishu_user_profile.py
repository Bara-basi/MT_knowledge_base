from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.api.v1 import feishu


def test_fetch_feishu_user_info_uses_detail_endpoint_and_union_id(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        status_code = 200
        text = '{"code":0}'

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "code": 0,
                "data": {
                    "user": {
                        "union_id": "on_union",
                        "user_id": "tenant-user-id",
                        "department_ids": ["od_academy"],
                    }
                },
            }

    class FakeClient:
        def __init__(self, timeout):
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url, params, headers):
            captured.update(url=url, params=params, headers=headers)
            return FakeResponse()

    monkeypatch.setattr(feishu.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(
        feishu,
        "settings",
        SimpleNamespace(
            feishu_timeout=30,
            feishu_base_url="https://open.feishu.cn",
        ),
    )

    info = asyncio.run(
        feishu._fetch_feishu_user_info(
            token="tenant-token",
            user_id="on_union/with-special",
            user_id_type="union_id",
        )
    )

    assert captured["url"].endswith("/contact/v3/users/on_union%2Fwith-special")
    assert captured["params"] == {
        "user_id_type": "union_id",
        "department_id_type": "open_department_id",
    }
    assert info["user_id"] == "tenant-user-id"
    assert info["department_ids"] == ["od_academy"]


def test_fetch_feishu_user_profile_resolves_department_and_job(monkeypatch) -> None:
    async def fake_token():
        return "tenant-token"

    async def fake_user_info(**kwargs):
        assert kwargs == {
            "token": "tenant-token",
            "user_id": "on_union",
            "user_id_type": "union_id",
        }
        return {
            "union_id": "on_union",
            "user_id": "tenant-user-id",
            "open_id": "ou_open",
            "name": "张三",
            "department_ids": ["od_academy"],
            "job_title": "培训生",
            "employee_type": 1,
        }

    async def fake_department_info(**kwargs):
        assert kwargs == {
            "token": "tenant-token",
            "department_id": "od_academy",
        }
        return {"name": "迈拓思学园"}

    monkeypatch.setattr(feishu, "_get_tenant_access_token", fake_token)
    monkeypatch.setattr(feishu, "_fetch_feishu_user_info", fake_user_info)
    monkeypatch.setattr(feishu, "_fetch_feishu_department_info", fake_department_info)
    monkeypatch.setattr(
        feishu,
        "settings",
        SimpleNamespace(feishu_app_id="app-id", feishu_app_secret="app-secret"),
    )

    profile = asyncio.run(feishu.fetch_feishu_user_profile_by_union_id("on_union"))

    assert profile.union_id == "on_union"
    assert profile.user_id == "tenant-user-id"
    assert profile.department_ids == ("od_academy",)
    assert profile.department_names == ("迈拓思学园",)
    assert profile.job_title == "培训生"
    assert profile.employee_type == 1
    assert profile.department_field_available is True
    assert profile.job_title_field_available is True


def test_fetch_feishu_user_profile_reports_trimmed_fields(monkeypatch) -> None:
    async def fake_token():
        return "tenant-token"

    async def fake_user_info(**_kwargs):
        return {
            "union_id": "on_union",
            "user_id": "tenant-user-id",
            "name": "张三",
        }

    monkeypatch.setattr(feishu, "_get_tenant_access_token", fake_token)
    monkeypatch.setattr(feishu, "_fetch_feishu_user_info", fake_user_info)
    monkeypatch.setattr(
        feishu,
        "settings",
        SimpleNamespace(feishu_app_id="app-id", feishu_app_secret="app-secret"),
    )

    profile = asyncio.run(feishu.fetch_feishu_user_profile_by_union_id("on_union"))

    assert profile.department_ids == ()
    assert profile.department_names == ()
    assert profile.job_title is None
    assert profile.department_field_available is False
    assert profile.job_title_field_available is False
    assert profile.returned_fields == ("name", "union_id", "user_id")


def test_fetch_feishu_user_profile_keeps_basic_info_when_department_is_forbidden(monkeypatch) -> None:
    async def fake_token():
        return "tenant-token"

    async def fake_user_info(**_kwargs):
        return {
            "union_id": "on_union",
            "user_id": "tenant-user-id",
            "name": "测试用户",
            "department_ids": ["od_hidden"],
            "job_title": "培训专员",
        }

    async def fake_department_info(**_kwargs):
        raise RuntimeError("no dept authority")

    monkeypatch.setattr(feishu, "_get_tenant_access_token", fake_token)
    monkeypatch.setattr(feishu, "_fetch_feishu_user_info", fake_user_info)
    monkeypatch.setattr(feishu, "_fetch_feishu_department_info", fake_department_info)
    monkeypatch.setattr(
        feishu,
        "settings",
        SimpleNamespace(feishu_app_id="app-id", feishu_app_secret="app-secret"),
    )

    profile = asyncio.run(feishu.fetch_feishu_user_profile_by_union_id("on_union"))

    assert profile.name == "测试用户"
    assert profile.department_ids == ("od_hidden",)
    assert profile.department_names == ()
    assert profile.job_title == "培训专员"

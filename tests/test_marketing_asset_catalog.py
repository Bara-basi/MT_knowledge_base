from __future__ import annotations

from contextlib import contextmanager

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1 import documents
from app.services import marketing_asset_catalog as catalog


class FakeCursor:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.executed: list[tuple[str, object]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def execute(self, query, params=None) -> None:
        self.executed.append((str(query), params))

    def fetchall(self) -> list[dict]:
        return self.rows


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self.cursor_instance = cursor

    def cursor(self) -> FakeCursor:
        return self.cursor_instance


@contextmanager
def fake_postgres(cursor: FakeCursor):
    yield FakeConnection(cursor)


def test_search_prioritizes_exact_file_name(monkeypatch) -> None:
    cursor = FakeCursor(
        [
            {
                "library_name": "MTSCO营销资料库",
                "full_path": "MTSCO营销资料库/产品/样册/管材样册.pdf",
                "document_name": "管材样册.pdf",
                "feishu_link": "https://example.feishu.cn/wiki/a",
                "normalized_path": "mtsco营销资料库/产品/样册/管材样册.pdf",
            },
            {
                "library_name": "MTSCO营销资料库",
                "full_path": "MTSCO营销资料库/历史资料/管材样册旧版.pdf",
                "document_name": "管材样册旧版.pdf",
                "feishu_link": "https://example.feishu.cn/wiki/b",
                "normalized_path": "mtsco营销资料库/历史资料/管材样册旧版.pdf",
            },
        ]
    )
    monkeypatch.setattr(catalog, "postgres_connection", lambda: fake_postgres(cursor))

    matches = catalog.search_marketing_assets("管材样册.pdf")

    assert [match.feishu_link for match in matches] == [
        "https://example.feishu.cn/wiki/a",
        "https://example.feishu.cn/wiki/b",
    ]
    assert matches[0].match_mode == "exact_name"
    assert "normalized_path LIKE" in cursor.executed[0][0]


def test_harness_search_response_is_compact_text(monkeypatch) -> None:
    app = FastAPI()
    app.include_router(documents.router)
    monkeypatch.setattr(
        documents,
        "search_marketing_assets",
        lambda _query, limit: [
            catalog.MarketingAssetMatch(
                library_name="MTSCO营销资料库",
                full_path="MTSCO营销资料库/样册/管材样册.pdf",
                document_name="管材样册.pdf",
                feishu_link="https://example.feishu.cn/wiki/a",
                score=100,
                match_mode="exact_name",
            )
        ][:limit],
    )

    response = TestClient(app).post(
        "/documents/marketing-assets/search",
        json={"query": "管材样册", "source": "harness"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "query": "管材样册",
        "source": "harness",
        "count": 1,
        "text": "找到 1 条与“管材样册”匹配的营销资料：\n1. 路径：MTSCO营销资料库/样册/管材样册.pdf\n   飞书链接：https://example.feishu.cn/wiki/a",
    }

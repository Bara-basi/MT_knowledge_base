from __future__ import annotations

from contextlib import contextmanager

from app.services import document_direct_retrieval as service


class FakeCursor:
    def __init__(self, exact_rows, fuzzy_rows) -> None:
        self.exact_rows = exact_rows
        self.fuzzy_rows = fuzzy_rows
        self.rows = []
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def execute(self, query, params=None):
        text = str(query)
        self.executed.append((text, params))
        self.rows = self.exact_rows if "document_name = %s" in text else self.fuzzy_rows

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, cursor) -> None:
        self.cursor_instance = cursor

    def cursor(self):
        return self.cursor_instance


class FakeObject:
    def __init__(self, object_name: str) -> None:
        self.object_name = object_name


class FakeResponse:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def read(self):
        return self.data

    def close(self):
        pass

    def release_conn(self):
        pass


class FakeMinioClient:
    def __init__(self) -> None:
        self.objects = [
            FakeObject("dept/manual.docx/txt/manual.txt"),
            FakeObject("dept/manual.docx/img/page1.png"),
            FakeObject("dept/manual.docx/chunk/manual.chunks.json"),
        ]

    def list_objects(self, bucket, prefix, recursive):
        assert bucket == "knowledge-processed-docs"
        assert prefix == "dept/manual.docx/"
        assert recursive is True
        return self.objects

    def get_object(self, bucket, object_name):
        assert object_name == "dept/manual.docx/txt/manual.txt"
        return FakeResponse("正文内容".encode("utf-8"))


@contextmanager
def fake_postgres(cursor):
    yield FakeConnection(cursor)


def test_document_search_prefers_exact_match(monkeypatch) -> None:
    exact_rows = [
        {
            "document_name": "manual.docx",
            "document_original_path": "minio://knowledge-raw-docs/dept/manual.docx",
            "processed_document_path": "minio://knowledge-processed-docs/dept/manual.docx",
        }
    ]
    fuzzy_rows = [
        {
            "document_name": "manual old.docx",
            "document_original_path": "minio://knowledge-raw-docs/dept/manual old.docx",
            "processed_document_path": "minio://knowledge-processed-docs/dept/manual old.docx",
        }
    ]
    cursor = FakeCursor(exact_rows, fuzzy_rows)
    monkeypatch.setattr(service, "postgres_connection", lambda: fake_postgres(cursor))

    matches = service.search_processed_documents("manual.docx")

    assert [item.document_name for item in matches] == ["manual.docx"]
    assert matches[0].match_mode == "exact"
    assert len(cursor.executed) == 1


def test_document_search_falls_back_to_fuzzy_match(monkeypatch) -> None:
    fuzzy_rows = [
        {
            "document_name": "manual old.docx",
            "document_original_path": "minio://knowledge-raw-docs/dept/manual old.docx",
            "processed_document_path": "minio://knowledge-processed-docs/dept/manual old.docx",
        }
    ]
    cursor = FakeCursor([], fuzzy_rows)
    monkeypatch.setattr(service, "postgres_connection", lambda: fake_postgres(cursor))

    matches = service.search_processed_documents("manual")

    assert [item.document_name for item in matches] == ["manual old.docx"]
    assert matches[0].match_mode == "fuzzy"
    assert len(cursor.executed) == 2


def test_direct_retrieval_reads_txt_as_one_chunk(monkeypatch) -> None:
    row = {
        "document_name": "manual.docx",
        "document_original_path": "minio://knowledge-raw-docs/dept/manual.docx",
        "processed_document_path": "minio://knowledge-processed-docs/dept/manual.docx",
    }
    cursor = FakeCursor([row], [])
    monkeypatch.setattr(service, "postgres_connection", lambda: fake_postgres(cursor))
    monkeypatch.setattr(service, "get_minio_client", lambda: FakeMinioClient())

    chunks, matches = service.build_direct_chunks("manual.docx")

    assert [item.document_name for item in matches] == ["manual.docx"]
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "document_direct"
    assert "正文内容" in chunks[0].content
    assert chunks[0].imgs == [
        {
            "index": 0,
            "img_name": "page1.png",
            "img_path": "minio://knowledge-processed-docs/dept/manual.docx/img/page1.png",
        }
    ]

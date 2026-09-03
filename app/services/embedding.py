from __future__ import annotations

import json
import os
import re
import tempfile
import csv
from functools import cached_property
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import httpx

from app.core.config import settings
from app.models.chunk import Chunk


REQUIRED_MODEL_FILES = {"config.json", "pytorch_model.bin"}
REQUIRED_TOKENIZER_FILES = {"tokenizer.json", "tokenizer_config.json", "vocab.txt"}
BM25_LANGUAGE = "zh"
REMOTE_EMBEDDING_MAX_CHARS = 500
GLOBAL_BM25_MODEL_FILE = Path("data") / "processing" / "global.bm25.json"
IMG_TAG_PATTERN = re.compile(r'(?s)<img\s+index="\d+">(?P<body>.*?)</img>')
LINK_TAG_PATTERN = re.compile(r'(?s)<a\s+index="\d+">(?P<body>.*?)</a>')


class EmbeddingDependencyError(RuntimeError):
    """Raised when optional embedding dependencies are not installed."""


class EmbeddingAPIError(RuntimeError):
    """Raised when the remote embedding API returns an invalid or failed response."""


class EmbeddingLocalError(RuntimeError):
    """Raised when the local embedding model cannot produce vectors."""


class EmbeddingService:
    """Text embedding service backed by BAAI/bge-large-zh-v1.5."""

    def __init__(
        self,
        model_name: str = settings.embedding_model_name,
        cache_dir: str = settings.embedding_cache_dir,
        device: str | None = settings.embedding_device,
        batch_size: int = settings.embedding_batch_size,
        normalize_embeddings: bool = settings.embedding_normalize,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

    @property
    def use_local_model(self) -> bool:
        return self._has_local_model() or settings.use_local_embedding_model

    @cached_property
    def tokenizer(self):
        self._configure_model_cache()

        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise EmbeddingDependencyError(
                "Embedding dependencies are missing. Install transformers and torch "
                "before using EmbeddingService."
            ) from exc

        model_source, local_files_only = self._resolve_model_source()
        return AutoTokenizer.from_pretrained(
            model_source,
            cache_dir=str(self.cache_dir / "huggingface" / "hub"),
            local_files_only=local_files_only,
        )

    @cached_property
    def model(self):
        self._configure_model_cache()

        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:
            raise EmbeddingDependencyError(
                "Embedding dependencies are missing. Install transformers and torch "
                "before using EmbeddingService."
            ) from exc

        model_source, local_files_only = self._resolve_model_source()
        model = AutoModel.from_pretrained(
            model_source,
            cache_dir=str(self.cache_dir / "huggingface" / "hub"),
            local_files_only=local_files_only,
        )
        model.to(self._resolve_device(torch))
        model.eval()
        return model

    def _resolve_model_source(self) -> tuple[str, bool]:
        local_path = Path(self.model_name)
        if local_path.exists():
            return str(local_path), True

        snapshot_path = self._find_local_huggingface_snapshot()
        if snapshot_path is not None:
            return str(snapshot_path), True

        return self.model_name, False

    def _find_local_huggingface_snapshot(self) -> Path | None:
        repo_dir = (
            self.cache_dir
            / "huggingface"
            / "hub"
            / f"models--{self.model_name.replace('/', '--')}"
        )
        snapshots_dir = repo_dir / "snapshots"
        if not snapshots_dir.exists():
            return None

        refs_main = repo_dir / "refs" / "main"
        candidates: list[Path] = []
        if refs_main.exists():
            ref = refs_main.read_text(encoding="utf-8").strip()
            if ref:
                candidates.append(snapshots_dir / ref)
        candidates.extend(path for path in snapshots_dir.iterdir() if path.is_dir())

        for candidate in candidates:
            if _is_complete_model_snapshot(candidate):
                return candidate
        return None

    def _configure_model_cache(self) -> None:
        configure_embedding_runtime_cache(self.cache_dir)

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        text_list = [text.strip() for text in texts if text and text.strip()]
        if not text_list:
            return []

        if self.use_local_model:
            try:
                return self._embed_texts_local(text_list)
            except EmbeddingLocalError as exc:
                if not settings.siliconflow_api_key:
                    raise
                print(
                    "[embedding] local embedding failed; falling back to remote API: "
                    f"{exc}",
                    flush=True,
                )

        return self._embed_texts_remote(text_list)

    def _embed_texts_local(self, texts: list[str]) -> list[list[float]]:
        if not self._has_local_model() and not settings.use_local_embedding_model:
            raise EmbeddingLocalError("Local embedding model is not available.")

        try:
            import torch
            import torch.nn.functional as functional
        except ImportError as exc:
            raise EmbeddingLocalError(
                "Embedding dependencies are missing. Install torch before using "
                "EmbeddingService."
            ) from exc

        try:
            device = self._resolve_device(torch)
            vectors: list[list[float]] = []
            with torch.no_grad():
                for start in range(0, len(texts), self.batch_size):
                    batch = texts[start : start + self.batch_size]
                    encoded = self.tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    )
                    encoded = {key: value.to(device) for key, value in encoded.items()}
                    output = self.model(**encoded)
                    batch_vectors = output.last_hidden_state[:, 0]
                    if self.normalize_embeddings:
                        batch_vectors = functional.normalize(batch_vectors, p=2, dim=1)
                    vectors.extend(batch_vectors.cpu().float().tolist())
        except Exception as exc:
            raise EmbeddingLocalError(str(exc)) from exc

        return vectors

    def _has_local_model(self) -> bool:
        _, local_files_only = self._resolve_model_source()
        return local_files_only

    def _embed_texts_remote(self, texts: list[str]) -> list[list[float]]:
        self._ensure_remote_configured()

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [_trim_remote_embedding_input(text) for text in texts[start : start + self.batch_size]]
            payload = {
                "model": self.model_name,
                "input": batch,
                "encoding_format": "float",
            }
            data = self._post_siliconflow("/embeddings", payload)
            items = data.get("data")
            if not isinstance(items, list):
                raise EmbeddingAPIError(f"Embedding API response missing data list: {data}")

            ordered_items = sorted(
                items,
                key=lambda item: item.get("index", 0) if isinstance(item, dict) else 0,
            )
            for item in ordered_items:
                if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                    raise EmbeddingAPIError(f"Invalid embedding item returned: {item}")
                vectors.append([float(value) for value in item["embedding"]])

            if len(ordered_items) != len(batch):
                raise EmbeddingAPIError(
                    f"Embedding API returned {len(ordered_items)} vectors for {len(batch)} inputs."
                )

        return vectors

    def _post_siliconflow(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{settings.siliconflow_base_url}{path}"
        headers = {
            "Authorization": f"Bearer {settings.siliconflow_api_key}",
            "Content-Type": "application/json",
        }
        try:
            timeout = httpx.Timeout(
                timeout=settings.siliconflow_timeout,
                connect=settings.siliconflow_connect_timeout,
                read=settings.siliconflow_read_timeout,
                write=settings.siliconflow_write_timeout,
                pool=settings.siliconflow_pool_timeout,
            )
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise EmbeddingAPIError(f"Embedding API request timed out: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise EmbeddingAPIError(
                f"Embedding API returned {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise EmbeddingAPIError(f"Failed to call Embedding API: {exc}") from exc

        return response.json()

    def _ensure_remote_configured(self) -> None:
        if not settings.siliconflow_api_key:
            raise EmbeddingAPIError(
                "Missing SiliconFlow API key. Set SILICONFLOW_API_KEY before "
                "USE_LOCAL_EMBEDDING_MODEL=false."
            )

    def warmup(self) -> None:
        if not self.use_local_model:
            return
        _ = self.tokenizer
        _ = self.model

    def _resolve_device(self, torch_module) -> str:
        if self.device:
            return self.device
        return "cuda" if torch_module.cuda.is_available() else "cpu"

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_texts([text])
        if not vectors:
            raise ValueError("Cannot embed an empty query.")
        return vectors[0]

    def embed_bm25_query(self, text: str, bm25_model_file: str | Path) -> dict[int, float]:
        bm25_model = load_bm25_embedding_function(bm25_model_file)
        sparse_matrix = bm25_model.encode_queries([text])
        vectors = sparse_matrix_to_json_vectors(sparse_matrix)
        if not vectors:
            return {}
        return {int(key): value for key, value in vectors[0].items()}

    def build_embedding_text(self, chunk: Chunk) -> str:
        return _join_embedding_parts(
            _file_name_stem(chunk.metadata.get("file_name")),
            chunk.metadata.get("path"),
            _format_content_for_embedding(chunk.content),
        )

    def embed_chunks(
        self,
        chunks: Iterable[Chunk],
        *,
        bm25_model: Any | None = None,
        bm25_model_file: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        chunk_list = list(chunks)
        if not chunk_list:
            return []

        embedding_texts = [self.build_embedding_text(chunk) for chunk in chunk_list]
        vectors = self.embed_texts(embedding_texts)
        if bm25_model is None:
            if bm25_model_file is None:
                bm25_model_file = default_bm25_model_file()
            bm25_model = load_bm25_embedding_function(bm25_model_file)
        sparse_matrix = bm25_model.encode_documents(embedding_texts)
        bm25_vectors = sparse_matrix_to_json_vectors(sparse_matrix)

        results: list[dict[str, Any]] = []
        for chunk, embedding_text, vector, bm25_vector in zip(
            chunk_list,
            embedding_texts,
            vectors,
            bm25_vectors,
        ):
            data = chunk.to_dict()
            data["embedding_text"] = embedding_text
            data["embedding"] = vector
            data["embedding_model"] = self.model_name
            data["embedding_dimension"] = len(vector)
            data["bm25_embedding"] = bm25_vector
            data["bm25_model"] = "pymilvus.model.sparse.BM25EmbeddingFunction"
            data["bm25_language"] = BM25_LANGUAGE
            data["bm25_dimension"] = getattr(bm25_model, "dim", None)
            data["bm25_model_file"] = str(bm25_model_file or default_bm25_model_file())
            results.append(data)
        return results

    def embed_chunk_file(
        self,
        chunk_file: str | Path,
        output_file: str | Path | None = None,
        *,
        bm25_model: Any | None = None,
        bm25_model_file: str | Path | None = None,
    ) -> Path:
        chunk_path = Path(chunk_file)
        chunks = load_chunks(chunk_path)
        embeddings = self.embed_chunks(
            chunks,
            bm25_model=bm25_model,
            bm25_model_file=bm25_model_file,
        )

        if output_file is None:
            output_file = (
                chunk_path.parents[1]
                / "embedding"
                / chunk_path.name.replace(".chunks.json", ".embeddings.json")
            )

        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(embeddings, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def save_global_bm25_model_file(
        self,
        chunks: Iterable[Chunk],
        output_file: str | Path | None = None,
    ) -> tuple[Path, Any]:
        output_path = Path(output_file) if output_file is not None else default_bm25_model_file()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_list = list(chunks)
        if not chunk_list:
            raise ValueError("Cannot train a global BM25 model without chunks.")
        embedding_texts = [self.build_embedding_text(chunk) for chunk in chunk_list]
        bm25_model = build_bm25_embedding_function()
        bm25_model.fit(embedding_texts)
        try:
            bm25_model.save(str(output_path))
        except PermissionError as exc:
            raise PermissionError(
                f"Cannot write the global BM25 model at {output_path}. "
                "Run the ingestion command as the owner of data/processing "
                "(the production service user is mtsco), or repair that directory's ownership."
            ) from exc
        return output_path, bm25_model

    def save_bm25_model_file(self, chunks: list[Chunk], embedding_file: Path) -> Path:
        embedding_texts = [self.build_embedding_text(chunk) for chunk in chunks]
        bm25_model = build_bm25_embedding_function()
        bm25_model.fit(embedding_texts)
        output_file = default_bm25_model_file()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        bm25_model.save(str(output_file))
        return output_file

    def embed_processing_chunks(self, document_name: str) -> Path:
        chunk_file = (
            Path("data")
            / "processing"
            / document_name
            / "chunk"
            / f"{document_name}.chunks.json"
        )
        output_file = (
            Path("data")
            / "processing"
            / document_name
            / "embedding"
            / f"{document_name}.embeddings.json"
        )
        return self.embed_chunk_file(chunk_file, output_file)

    def smoke_test(self) -> dict[str, int | str]:
        vector = self.embed_query("MTSCO embedding smoke test")
        model_source, local_files_only = self._resolve_model_source()
        return {
            "model_name": self.model_name,
            "model_source": model_source,
            "local_files_only": str(local_files_only),
            "cache_dir": str(self.cache_dir),
            "dimension": len(vector),
            "batch_size": self.batch_size,
        }


embedding_service = EmbeddingService()


def default_bm25_model_file() -> Path:
    return GLOBAL_BM25_MODEL_FILE


def build_bm25_embedding_function():
    configure_embedding_runtime_cache(Path(settings.embedding_cache_dir))
    try:
        from pymilvus.model.sparse import BM25EmbeddingFunction
        from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
    except ImportError as exc:
        raise EmbeddingDependencyError(
            "BM25 sparse embedding dependencies are missing. Install them with: "
            'pip install "pymilvus[model]"'
        ) from exc

    load_jieba_expanded_vocab(
        settings.jieba_expanded_vocab_file,
        freq=settings.jieba_expanded_vocab_freq,
    )
    analyzer = build_default_analyzer(language=BM25_LANGUAGE)
    return BM25EmbeddingFunction(analyzer)


def load_jieba_expanded_vocab(
    vocab_file: str | Path | None = None,
    *,
    freq: int | None = None,
) -> int:
    path = Path(vocab_file or settings.jieba_expanded_vocab_file)
    if not path.exists():
        return 0

    try:
        import jieba
    except ImportError as exc:
        raise EmbeddingDependencyError(
            "jieba is required to load the expanded BM25 vocabulary."
        ) from exc

    word_freq = freq if freq is not None else settings.jieba_expanded_vocab_freq
    loaded = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            word = str(row.get("word") or row.get("词") or "").strip()
            tag = str(row.get("pos") or row.get("词性") or "nz").strip() or "nz"
            if not word:
                continue
            jieba.add_word(word, freq=word_freq, tag=tag)
            loaded += 1
    return loaded


def load_jieba_expand_vocab(
    vocab_file: str | Path | None = None,
    *,
    freq: int | None = None,
) -> int:
    return load_jieba_expanded_vocab(vocab_file, freq=freq)


def load_bm25_embedding_function(model_file: str | Path):
    bm25_model = build_bm25_embedding_function()
    safe_model_file = _ensure_bm25_model_loadable_on_windows(model_file)
    try:
        bm25_model.load(str(safe_model_file))
    finally:
        if safe_model_file != Path(model_file):
            safe_model_file.unlink(missing_ok=True)
    return bm25_model


def build_bm25_document_embeddings(texts: list[str]):
    bm25_model = build_bm25_embedding_function()
    bm25_model.fit(texts)
    sparse_matrix = bm25_model.encode_documents(texts)
    return bm25_model, sparse_matrix_to_json_vectors(sparse_matrix)


def sparse_matrix_to_json_vectors(sparse_matrix) -> list[dict[str, float]]:
    csr_matrix = sparse_matrix.tocsr()
    vectors: list[dict[str, float]] = []
    for row_index in range(csr_matrix.shape[0]):
        start = csr_matrix.indptr[row_index]
        end = csr_matrix.indptr[row_index + 1]
        indices = csr_matrix.indices[start:end]
        data = csr_matrix.data[start:end]
        vectors.append(
            {
                str(index): float(value)
                for index, value in zip(indices.tolist(), data.tolist())
            }
        )
    return vectors


def configure_embedding_runtime_cache(cache_dir: str | Path) -> None:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    huggingface_home = cache_path / "huggingface"
    env_paths = {
        "HF_HOME": huggingface_home,
        "HF_HUB_CACHE": huggingface_home / "hub",
        "HUGGINGFACE_HUB_CACHE": huggingface_home / "hub",
        "SENTENCE_TRANSFORMERS_HOME": cache_path / "sentence-transformers",
        "NLTK_DATA": cache_path / "nltk_data",
        "JIEBA_CACHE_DIR": cache_path / "jieba",
        "TEMP": cache_path / "tmp",
        "TMP": cache_path / "tmp",
    }

    for name, path in env_paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)

    tempfile.tempdir = str(cache_path / "tmp")

    try:
        import jieba
    except ImportError:
        return

    jieba.dt.tmp_dir = str(cache_path / "tmp")
    jieba.dt.cache_file = "jieba.cache"


def load_chunks(chunk_file: str | Path) -> list[Chunk]:
    path = Path(chunk_file)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Chunk file must contain a JSON list: {path}")

    chunks: list[Chunk] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError(f"Chunk item must be a JSON object: {path}")
        chunks.append(
            Chunk(
                content=item["content"],
                metadata=item.get("metadata", {}),
                chunk_index=item.get("chunk_index"),
                file_id=item.get("file_id"),
                vector_id=item.get("vector_id"),
            )
        )
    return chunks


def _rewrite_json_without_ascii_escape(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_bm25_model_loadable_on_windows(model_file: str | Path) -> Path:
    path = Path(model_file)
    data = path.read_text(encoding="utf-8")
    if data.isascii():
        return path

    payload = json.loads(data)
    safe_path = path.with_name(f"{path.stem}.{uuid4().hex}.ascii{path.suffix}")
    safe_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="ascii")
    return safe_path


def _format_content_for_embedding(content: str) -> str:
    text = content.strip()
    if not text:
        return text

    text = _strip_asset_tags_for_embedding(text)
    table_data = _parse_json_table_content(text)
    if table_data is None:
        return text
    return _format_table_data_for_embedding(table_data)


def _join_embedding_parts(*values: Any) -> str:
    return " ".join(str(value).strip() for value in values if _has_value(value))


def _file_name_stem(file_name: Any) -> str:
    value = str(file_name or "").strip()
    if not value:
        return ""
    return Path(value).stem


def _strip_asset_tags_for_embedding(text: str) -> str:
    def replace_image(match: re.Match[str]) -> str:
        body = re.sub(r"\s+", " ", match.group("body")).strip()
        body = re.sub(r"^图片\s*[:：]\s*", "", body)
        return body

    def replace_link(match: re.Match[str]) -> str:
        body = re.sub(r"\s+", " ", match.group("body")).strip()
        body = re.sub(r"^链接\s*[:：]\s*", "", body)
        return body

    text = IMG_TAG_PATTERN.sub(replace_image, text)
    text = LINK_TAG_PATTERN.sub(replace_link, text)
    return text.strip()


def _parse_json_table_content(text: str) -> Any | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None

    if _looks_like_table_data(parsed):
        return parsed
    if isinstance(parsed, dict):
        for key in ("table_data", "data", "rows"):
            value = parsed.get(key)
            if _looks_like_table_data(value):
                return value
    return None


def _looks_like_table_data(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    return all(isinstance(row, dict) for row in value) or all(
        isinstance(row, list) for row in value
    )


def _format_table_data_for_embedding(table_data: Any) -> str:
    if all(isinstance(row, dict) for row in table_data):
        headers = _ordered_table_headers(table_data)
        table_data = [
            row
            for row in table_data
            if not _is_repeated_dict_header_row(row, headers)
        ]
        rows = [
            [_table_cell_for_embedding(row.get(header, "")) for header in headers]
            for row in table_data
        ]
        return json.dumps([headers, *rows], ensure_ascii=False, separators=(",", ":"))

    rows = list(table_data)
    normalized_rows = [
        [_table_cell_for_embedding(cell) for cell in row]
        for row in rows
    ]
    if normalized_rows:
        header_signature = [_normalized_table_token(cell) for cell in normalized_rows[0]]
        normalized_rows = [
            normalized_rows[0],
            *[
                row
                for row in normalized_rows[1:]
                if [_normalized_table_token(cell) for cell in row] != header_signature
            ],
        ]
    return json.dumps(normalized_rows, ensure_ascii=False, separators=(",", ":"))


def _is_repeated_dict_header_row(row: dict[str, Any], headers: list[str]) -> bool:
    populated = 0
    matched = 0
    for header in headers:
        value = row.get(header, "")
        if not _has_value(value):
            continue
        populated += 1
        if _normalized_table_token(value) == _normalized_table_token(header):
            matched += 1
    return populated > 0 and matched / populated >= 0.8


def _normalized_table_token(value: Any) -> str:
    return re.sub(r"[\s:：_\-—]+", "", str(value or "")).casefold()


def _ordered_table_headers(rows: list[dict[str, Any]]) -> list[str]:
    headers: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            header = str(key)
            if header not in seen:
                seen.add(header)
                headers.append(header)
    return headers


def _table_cell_for_embedding(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if value is None:
        return ""
    return value


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _trim_remote_embedding_input(text: str) -> str:
    text = text.strip()
    if len(text) <= REMOTE_EMBEDDING_MAX_CHARS:
        return text
    return text[:REMOTE_EMBEDDING_MAX_CHARS]


def _is_complete_model_snapshot(path: Path) -> bool:
    if not path.is_dir():
        return False
    files = {item.name for item in path.iterdir() if item.is_file()}
    has_model = REQUIRED_MODEL_FILES.issubset(files)
    has_tokenizer = bool(REQUIRED_TOKENIZER_FILES & files)
    return has_model and has_tokenizer

from __future__ import annotations

import json
import os
import tempfile
from functools import cached_property
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from app.core.config import settings
from app.models.chunk import Chunk


EMBEDDING_METADATA_EXCLUDE_KEYS = {"link", "img"}
REQUIRED_MODEL_FILES = {"config.json", "pytorch_model.bin"}
REQUIRED_TOKENIZER_FILES = {"tokenizer.json", "tokenizer_config.json", "vocab.txt"}
BM25_LANGUAGE = "zh"


class EmbeddingDependencyError(RuntimeError):
    """Raised when optional embedding dependencies are not installed."""


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

        try:
            import torch
            import torch.nn.functional as functional
        except ImportError as exc:
            raise EmbeddingDependencyError(
                "Embedding dependencies are missing. Install torch before using "
                "EmbeddingService."
            ) from exc

        device = self._resolve_device(torch)
        vectors: list[list[float]] = []
        with torch.no_grad():
            for start in range(0, len(text_list), self.batch_size):
                batch = text_list[start : start + self.batch_size]
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

        return vectors

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
        metadata_lines = [
            f"{key}: {_format_metadata_value(value)}"
            for key, value in chunk.metadata.items()
            if key not in EMBEDDING_METADATA_EXCLUDE_KEYS and _has_value(value)
        ]
        parts = []
        if metadata_lines:
            parts.append("metadata:\n" + "\n".join(metadata_lines))
        parts.append("content:\n" + chunk.content.strip())
        return "\n\n".join(parts)

    def embed_chunks(self, chunks: Iterable[Chunk]) -> list[dict[str, Any]]:
        chunk_list = list(chunks)
        embedding_texts = [self.build_embedding_text(chunk) for chunk in chunk_list]
        vectors = self.embed_texts(embedding_texts)
        bm25_model, bm25_vectors = build_bm25_document_embeddings(embedding_texts)

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
            results.append(data)
        return results

    def embed_chunk_file(
        self,
        chunk_file: str | Path,
        output_file: str | Path | None = None,
    ) -> Path:
        chunk_path = Path(chunk_file)
        chunks = load_chunks(chunk_path)
        embeddings = self.embed_chunks(chunks)

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
        self.save_bm25_model_file(chunks, path)
        return path

    def save_bm25_model_file(self, chunks: list[Chunk], embedding_file: Path) -> Path:
        embedding_texts = [self.build_embedding_text(chunk) for chunk in chunks]
        bm25_model = build_bm25_embedding_function()
        bm25_model.fit(embedding_texts)
        output_file = embedding_file.with_suffix("").with_suffix(".bm25.json")
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

    analyzer = build_default_analyzer(language=BM25_LANGUAGE)
    return BM25EmbeddingFunction(analyzer)


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
                document_id=item.get("document_id"),
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


def _format_metadata_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _is_complete_model_snapshot(path: Path) -> bool:
    if not path.is_dir():
        return False
    files = {item.name for item in path.iterdir() if item.is_file()}
    has_model = REQUIRED_MODEL_FILES.issubset(files)
    has_tokenizer = bool(REQUIRED_TOKENIZER_FILES & files)
    return has_model and has_tokenizer

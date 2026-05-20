from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import httpx

from app.core.config import settings
from app.services.embedding import configure_embedding_runtime_cache


REQUIRED_CONFIG_FILES = {"config.json"}
REQUIRED_MODEL_WEIGHT_FILES = {
    "model.safetensors",
    "pytorch_model.bin",
}
REQUIRED_TOKENIZER_FILES = {
    "sentencepiece.bpe.model",
    "tokenizer.json",
    "tokenizer.model",
    "vocab.txt",
}


class RerankDependencyError(RuntimeError):
    """Raised when reranker runtime dependencies are missing."""


class RerankAPIError(RuntimeError):
    """Raised when the remote rerank API returns an invalid or failed response."""


@dataclass(frozen=True)
class RerankCandidate:
    id: str
    content: str


@dataclass(frozen=True)
class RerankScore:
    id: str
    score: float


class RerankService:
    """Cross-encoder reranker backed by BAAI/bge-reranker-v2-m3."""

    def __init__(
        self,
        model_name: str = settings.reranker_model_name,
        cache_dir: str = settings.reranker_cache_dir,
        device: str | None = settings.reranker_device,
        batch_size: int = settings.reranker_batch_size,
        max_length: int = settings.reranker_max_length,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

    @property
    def use_local_model(self) -> bool:
        return settings.use_local_rerank_model

    @cached_property
    def tokenizer(self):
        self._configure_model_cache()

        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RerankDependencyError(
                "Rerank dependencies are missing. Install transformers before using "
                "RerankService."
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
            from transformers import AutoModelForSequenceClassification
        except ImportError as exc:
            raise RerankDependencyError(
                "Rerank dependencies are missing. Install transformers and torch "
                "before using RerankService."
            ) from exc

        model_source, local_files_only = self._resolve_model_source()
        model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            cache_dir=str(self.cache_dir / "huggingface" / "hub"),
            local_files_only=local_files_only,
        )
        model.to(self._resolve_device(torch))
        model.eval()
        return model

    def rerank(self, query: str, candidates: list[RerankCandidate]) -> list[RerankScore]:
        candidate_list = [
            candidate
            for candidate in candidates
            if candidate.id and candidate.content and candidate.content.strip()
        ]
        if not query.strip() or not candidate_list:
            return []
        if not self.use_local_model:
            return self._rerank_remote(query, candidate_list)

        try:
            import torch
        except ImportError as exc:
            raise RerankDependencyError(
                "Rerank dependencies are missing. Install torch before using "
                "RerankService."
            ) from exc

        device = self._resolve_device(torch)
        scores: list[RerankScore] = []
        with torch.no_grad():
            for start in range(0, len(candidate_list), self.batch_size):
                batch = candidate_list[start : start + self.batch_size]
                pairs = [(query, candidate.content) for candidate in batch]
                encoded = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                logits = self.model(**encoded).logits.view(-1)
                for candidate, score in zip(batch, logits.cpu().float().tolist()):
                    scores.append(RerankScore(id=candidate.id, score=float(score)))

        return sorted(scores, key=lambda item: item.score, reverse=True)

    def _rerank_remote(
        self,
        query: str,
        candidates: list[RerankCandidate],
    ) -> list[RerankScore]:
        self._ensure_remote_configured()
        scores: list[RerankScore] = []

        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start : start + self.batch_size]
            payload = {
                "model": self.model_name,
                "query": query,
                "documents": [candidate.content for candidate in batch],
                "top_n": len(batch),
                "return_documents": False,
            }
            data = self._post_siliconflow("/rerank", payload)
            results = data.get("results")
            if not isinstance(results, list):
                raise RerankAPIError(f"Rerank API response missing results list: {data}")

            for item in results:
                if not isinstance(item, dict):
                    raise RerankAPIError(f"Invalid rerank result returned: {item}")
                index = item.get("index")
                score = item.get("relevance_score")
                if not isinstance(index, int) or not isinstance(score, (int, float)):
                    raise RerankAPIError(f"Invalid rerank score item returned: {item}")
                if index < 0 or index >= len(batch):
                    raise RerankAPIError(f"Rerank result index out of range: {item}")
                scores.append(RerankScore(id=batch[index].id, score=float(score)))

        return sorted(scores, key=lambda item: item.score, reverse=True)

    def _post_siliconflow(self, path: str, payload: dict) -> dict:
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
            raise RerankAPIError(f"Rerank API request timed out: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise RerankAPIError(
                f"Rerank API returned {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RerankAPIError(f"Failed to call Rerank API: {exc}") from exc

        return response.json()

    def _ensure_remote_configured(self) -> None:
        if not settings.siliconflow_api_key:
            raise RerankAPIError(
                "Missing SiliconFlow API key. Set SILICONFLOW_API_KEY before "
                "USE_LOCAL_RERANK_MODEL=false."
            )

    def warmup(self) -> None:
        if not self.use_local_model:
            return
        _ = self.tokenizer
        _ = self.model

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

    def _resolve_device(self, torch_module) -> str:
        if self.device:
            return self.device
        return "cuda" if torch_module.cuda.is_available() else "cpu"


rerank_service = RerankService()


def _is_complete_model_snapshot(path: Path) -> bool:
    if not path.is_dir():
        return False
    files = {item.name for item in path.iterdir() if item.is_file()}
    has_config = REQUIRED_CONFIG_FILES.issubset(files)
    has_model = bool(REQUIRED_MODEL_WEIGHT_FILES & files)
    has_tokenizer = bool(REQUIRED_TOKENIZER_FILES & files)
    return has_config and has_model and has_tokenizer

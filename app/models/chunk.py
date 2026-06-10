from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A semantic chunk ready for embedding, retrieval, and later persistence."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_index: int | None = None
    file_id: str | None = None
    vector_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "content": self.content,
            "metadata": self.metadata,
        }

        if self.chunk_index is not None:
            data["chunk_index"] = self.chunk_index
        if self.file_id is not None:
            data["file_id"] = self.file_id
        if self.vector_id is not None:
            data["vector_id"] = self.vector_id

        return data

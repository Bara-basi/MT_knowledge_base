from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pymilvus import DataType, MilvusClient

from app.core.config import settings


DEFAULT_COLLECTION_DESCRIPTION = "MTSCO internal knowledge base text chunks"


@dataclass(frozen=True)
class MilvusCollectionConfig:
    name: str = settings.milvus_collection_name
    vector_dim: int = settings.milvus_vector_dim
    metric_type: str = settings.milvus_metric_type
    index_type: str = settings.milvus_index_type
    index_m: int = settings.milvus_index_m
    index_ef_construction: int = settings.milvus_index_ef_construction


def get_milvus_client(timeout: float | None = 10.0) -> MilvusClient:
    """Create a Milvus client from environment-backed settings."""

    return MilvusClient(
        uri=settings.milvus_uri,
        user=settings.milvus_user,
        password=settings.milvus_password,
        db_name=settings.milvus_db_name,
        timeout=timeout,
    )


def build_chunk_schema(config: MilvusCollectionConfig | None = None) -> Any:
    """Build the collection schema used for chunk vectors."""

    config = config or MilvusCollectionConfig()
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
        description=DEFAULT_COLLECTION_DESCRIPTION,
    )
    schema.add_field(
        field_name="id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        max_length=128,
    )
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=config.vector_dim)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=8192)
    schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
    schema.add_field(field_name="metadata", datatype=DataType.JSON)
    return schema


def build_index_params(config: MilvusCollectionConfig | None = None) -> Any:
    """Build vector index parameters for local/development Milvus setup."""

    config = config or MilvusCollectionConfig()
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type=config.index_type,
        metric_type=config.metric_type,
        params={
            "M": config.index_m,
            "efConstruction": config.index_ef_construction,
        },
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"drop_ratio_build": 0.2},
    )
    return index_params


def ensure_chunk_collection(
    client: MilvusClient | None = None,
    config: MilvusCollectionConfig | None = None,
) -> dict[str, Any]:
    """Create and load the chunk collection if it does not already exist."""

    client = client or get_milvus_client()
    config = config or MilvusCollectionConfig()

    created = False
    if not client.has_collection(config.name):
        client.create_collection(
            collection_name=config.name,
            schema=build_chunk_schema(config),
            index_params=build_index_params(config),
        )
        created = True

    client.load_collection(config.name)
    return {
        "collection_name": config.name,
        "created": created,
        "description": client.describe_collection(config.name),
        "load_state": client.get_load_state(config.name),
    }


def check_milvus_health(client: MilvusClient | None = None) -> dict[str, Any]:
    """Return basic Milvus connectivity details without mutating data."""

    client = client or get_milvus_client()
    return {
        "uri": settings.milvus_uri,
        "collections": client.list_collections(),
    }


def drop_chunk_collection(
    client: MilvusClient | None = None,
    config: MilvusCollectionConfig | None = None,
) -> dict[str, Any]:
    client = client or get_milvus_client()
    config = config or MilvusCollectionConfig()

    dropped = False
    if client.has_collection(config.name):
        client.drop_collection(config.name)
        dropped = True

    return {
        "collection_name": config.name,
        "dropped": dropped,
    }

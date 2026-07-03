from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from app.core.config import settings


DEFAULT_NODE_LABEL = "GraphNode"
KNOWN_NODE_LABELS = {
    "Document",
    "Product",
    "Section",
    "Standard",
    "StandardVersion",
    "Table",
}
KNOWN_RELATIONSHIP_TYPES = {
    "apply_to",
    "has_section",
    "has_sub_document",
    "has_table",
    "has_version",
    "is_about",
    "reference_to",
}


class Neo4jDependencyError(RuntimeError):
    """Raised when the Neo4j Python driver is not installed."""


def _load_neo4j() -> Any:
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise Neo4jDependencyError(
            "Neo4j driver is missing. Install project dependencies or run: pip install neo4j"
        ) from exc
    return GraphDatabase


def get_neo4j_driver() -> Any:
    """Create a Neo4j driver from environment-backed settings."""

    graph_database = _load_neo4j()
    auth = (settings.neo4j_user, settings.neo4j_password)
    return graph_database.driver(settings.neo4j_uri, auth=auth)


@contextmanager
def neo4j_session(**kwargs: Any) -> Iterator[Any]:
    driver = get_neo4j_driver()
    try:
        with driver.session(database=settings.neo4j_database, **kwargs) as session:
            yield session
    finally:
        driver.close()


def check_neo4j_health() -> dict[str, Any]:
    """Return basic Neo4j connectivity details without mutating data."""

    driver = get_neo4j_driver()
    try:
        driver.verify_connectivity()
        with driver.session(database=settings.neo4j_database) as session:
            row = session.run(
                "RETURN $uri AS uri, $database AS database, datetime() AS server_time",
                uri=settings.neo4j_uri,
                database=settings.neo4j_database,
            ).single()
            return dict(row or {})
    finally:
        driver.close()


def ensure_neo4j_schema(
    *,
    node_labels: set[str] | None = None,
    relationship_types: set[str] | None = None,
) -> dict[str, Any]:
    """Create constraints used by the knowledge graph import code."""

    node_labels = node_labels or KNOWN_NODE_LABELS
    relationship_types = relationship_types or KNOWN_RELATIONSHIP_TYPES

    statements = [
        (
            "graph_node_id_unique",
            f"CREATE CONSTRAINT graph_node_id_unique IF NOT EXISTS "
            f"FOR (n:{DEFAULT_NODE_LABEL}) REQUIRE n.id IS UNIQUE",
        )
    ]
    for label in sorted(node_labels):
        safe_label = validate_neo4j_identifier(label)
        statements.append(
            (
                f"{safe_label.lower()}_id_unique",
                f"CREATE CONSTRAINT {safe_label.lower()}_id_unique IF NOT EXISTS "
                f"FOR (n:{safe_label}) REQUIRE n.id IS UNIQUE",
            )
        )

    for rel_type in sorted(relationship_types):
        safe_rel_type = validate_neo4j_identifier(rel_type)
        statements.append(
            (
                f"{safe_rel_type.lower()}_id_unique",
                f"CREATE CONSTRAINT {safe_rel_type.lower()}_id_unique IF NOT EXISTS "
                f"FOR ()-[r:{safe_rel_type}]-() REQUIRE r.id IS UNIQUE",
            )
        )

    driver = get_neo4j_driver()
    try:
        with driver.session(database=settings.neo4j_database) as session:
            for _, statement in statements:
                session.run(statement).consume()
        return {
            "database": settings.neo4j_database,
            "graph_name": settings.neo4j_graph_name,
            "constraints": [name for name, _ in statements],
        }
    finally:
        driver.close()


def delete_graph(graph_name: str | None = None) -> dict[str, Any]:
    """Delete all imported graph nodes and relationships for one logical graph."""

    graph_name = graph_name or settings.neo4j_graph_name
    driver = get_neo4j_driver()
    try:
        with driver.session(database=settings.neo4j_database) as session:
            result = session.run(
                f"""
                MATCH (n:{DEFAULT_NODE_LABEL} {{graph_name: $graph_name}})
                DETACH DELETE n
                """,
                graph_name=graph_name,
            ).consume()
        counters = result.counters
        return {
            "database": settings.neo4j_database,
            "graph_name": graph_name,
            "nodes_deleted": counters.nodes_deleted,
            "relationships_deleted": counters.relationships_deleted,
        }
    finally:
        driver.close()


def validate_neo4j_identifier(value: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise ValueError("Neo4j identifier cannot be empty.")
    if not all(ch.isalnum() or ch == "_" for ch in cleaned):
        raise ValueError(f"Unsafe Neo4j identifier: {value!r}")
    if cleaned[0].isdigit():
        raise ValueError(f"Neo4j identifier cannot start with a digit: {value!r}")
    return cleaned

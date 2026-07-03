from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from app.core.config import settings
from app.db.neo4j import (
    DEFAULT_NODE_LABEL,
    get_neo4j_driver,
    validate_neo4j_identifier,
)


class GraphStoreError(RuntimeError):
    """Raised when graph records cannot be written to Neo4j."""


class GraphStoreService:
    """Neo4j-backed store for logical knowledge graph nodes and relationships."""

    def __init__(
        self,
        driver: Any | None = None,
        *,
        database: str | None = None,
        graph_name: str | None = None,
        sub_graph_name: str | None = None,
    ) -> None:
        self.driver = driver or get_neo4j_driver()
        self.database = database or settings.neo4j_database
        self.graph_name = graph_name or settings.neo4j_graph_name
        self.sub_graph_name = sub_graph_name
        self._owns_driver = driver is None

    def close(self) -> None:
        if self._owns_driver:
            self.driver.close()

    def upsert_node(self, node: dict[str, Any]) -> dict[str, Any]:
        label = validate_neo4j_identifier(str(node.get("label") or ""))
        props = normalize_node_properties(
            node,
            graph_name=self.graph_name,
            sub_graph_name=self.sub_graph_name,
        )
        query = f"""
        MERGE (n:{DEFAULT_NODE_LABEL} {{id: $id}})
        ON CREATE SET n.create_at = $create_at
        SET n:{label}
        SET n += $props
        RETURN n.id AS id, labels(n) AS labels, n.name AS name
        """
        with self.driver.session(database=self.database) as session:
            row = session.execute_write(
                lambda tx: tx.run(
                    query,
                    id=props["id"],
                    create_at=props.get("create_at"),
                    props=props,
                ).single()
            )
        return dict(row or {})

    def upsert_nodes(self, nodes: Iterable[dict[str, Any]], *, batch_size: int = 500) -> dict[str, Any]:
        count = 0
        by_label: dict[str, list[dict[str, Any]]] = {}
        for node in nodes:
            label = validate_neo4j_identifier(str(node.get("label") or ""))
            by_label.setdefault(label, []).append(
                normalize_node_properties(
                    node,
                    graph_name=self.graph_name,
                    sub_graph_name=self.sub_graph_name,
                )
            )

        with self.driver.session(database=self.database) as session:
            for label, rows in by_label.items():
                query = f"""
                UNWIND $rows AS row
                MERGE (n:{DEFAULT_NODE_LABEL} {{id: row.id}})
                ON CREATE SET n.create_at = row.create_at
                SET n:{label}
                SET n += row
                """
                for batch in batched(rows, batch_size):
                    session.execute_write(lambda tx, batch=batch: tx.run(query, rows=batch).consume())
                    count += len(batch)
        return {
            "graph_name": self.graph_name,
            "sub_graph_name": self.sub_graph_name,
            "nodes_upserted": count,
        }

    def upsert_relationship(self, relationship: dict[str, Any]) -> dict[str, Any]:
        rel_type = validate_neo4j_identifier(str(relationship.get("type") or ""))
        props = normalize_relationship_properties(
            relationship,
            graph_name=self.graph_name,
            sub_graph_name=self.sub_graph_name,
        )
        query = f"""
        MATCH (source:{DEFAULT_NODE_LABEL} {{id: $source_id}})
        MATCH (target:{DEFAULT_NODE_LABEL} {{id: $target_id}})
        MERGE (source)-[r:{rel_type} {{id: $id}}]->(target)
        ON CREATE SET r.create_at = $create_at
        SET r += $props
        RETURN r.id AS id, type(r) AS type, source.id AS source_id, target.id AS target_id
        """
        with self.driver.session(database=self.database) as session:
            row = session.execute_write(
                lambda tx: tx.run(
                    query,
                    id=props["id"],
                    source_id=props["source_id"],
                    target_id=props["target_id"],
                    create_at=props.get("create_at"),
                    props=props,
                ).single()
            )
        return dict(row or {})

    def upsert_relationships(self, relationships: Iterable[dict[str, Any]], *, batch_size: int = 500) -> dict[str, Any]:
        count = 0
        skipped = 0
        by_type: dict[str, list[dict[str, Any]]] = {}
        for relationship in relationships:
            rel_type = validate_neo4j_identifier(str(relationship.get("type") or ""))
            by_type.setdefault(rel_type, []).append(
                normalize_relationship_properties(
                    relationship,
                    graph_name=self.graph_name,
                    sub_graph_name=self.sub_graph_name,
                )
            )

        with self.driver.session(database=self.database) as session:
            for rel_type, rows in by_type.items():
                query = f"""
                UNWIND $rows AS row
                MATCH (source:{DEFAULT_NODE_LABEL} {{id: row.source_id}})
                MATCH (target:{DEFAULT_NODE_LABEL} {{id: row.target_id}})
                MERGE (source)-[r:{rel_type} {{id: row.id}}]->(target)
                ON CREATE SET r.create_at = row.create_at
                SET r += row
                RETURN count(r) AS written
                """
                for batch in batched(rows, batch_size):
                    written = session.execute_write(
                        lambda tx, batch=batch: tx.run(query, rows=batch).single()
                    )
                    written_count = int((written or {}).get("written", 0))
                    count += written_count
                    skipped += len(batch) - written_count
        return {
            "graph_name": self.graph_name,
            "sub_graph_name": self.sub_graph_name,
            "relationships_upserted": count,
            "relationships_skipped": skipped,
        }

    def import_manifest(
        self,
        nodes_file: str | Path,
        edges_file: str | Path,
        *,
        batch_size: int = 500,
    ) -> dict[str, Any]:
        nodes = load_jsonl(nodes_file)
        edges = load_jsonl(edges_file)
        node_result = self.upsert_nodes(nodes, batch_size=batch_size)
        edge_result = self.upsert_relationships(edges, batch_size=batch_size)
        return {
            "graph_name": self.graph_name,
            "sub_graph_name": self.sub_graph_name,
            "nodes_file": str(nodes_file),
            "edges_file": str(edges_file),
            **node_result,
            **edge_result,
        }

    def update_node_properties(self, node_id: str, properties: dict[str, Any]) -> dict[str, Any]:
        props = flatten_properties(properties)
        query = f"""
        MATCH (n:{DEFAULT_NODE_LABEL} {{id: $id, graph_name: $graph_name}})
        SET n += $props
        RETURN n.id AS id, labels(n) AS labels, properties(n) AS properties
        """
        with self.driver.session(database=self.database) as session:
            row = session.execute_write(
                lambda tx: tx.run(query, id=node_id, graph_name=self.graph_name, props=props).single()
            )
        return dict(row or {})

    def update_relationship_properties(self, relationship_id: str, properties: dict[str, Any]) -> dict[str, Any]:
        props = flatten_properties(properties)
        query = """
        MATCH ()-[r {id: $id, graph_name: $graph_name}]->()
        SET r += $props
        RETURN r.id AS id, type(r) AS type, properties(r) AS properties
        """
        with self.driver.session(database=self.database) as session:
            row = session.execute_write(
                lambda tx: tx.run(query, id=relationship_id, graph_name=self.graph_name, props=props).single()
            )
        return dict(row or {})

    def delete_node(self, node_id: str, *, detach: bool = True) -> dict[str, Any]:
        delete_clause = "DETACH DELETE n" if detach else "DELETE n"
        query = f"""
        MATCH (n:{DEFAULT_NODE_LABEL} {{id: $id, graph_name: $graph_name}})
        {delete_clause}
        """
        with self.driver.session(database=self.database) as session:
            result = session.execute_write(
                lambda tx: tx.run(query, id=node_id, graph_name=self.graph_name).consume()
            )
        return {
            "node_id": node_id,
            "nodes_deleted": result.counters.nodes_deleted,
            "relationships_deleted": result.counters.relationships_deleted,
        }

    def delete_relationship(self, relationship_id: str) -> dict[str, Any]:
        query = """
        MATCH ()-[r {id: $id, graph_name: $graph_name}]->()
        DELETE r
        """
        with self.driver.session(database=self.database) as session:
            result = session.execute_write(
                lambda tx: tx.run(query, id=relationship_id, graph_name=self.graph_name).consume()
            )
        return {
            "relationship_id": relationship_id,
            "relationships_deleted": result.counters.relationships_deleted,
        }

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        query = f"""
        MATCH (n:{DEFAULT_NODE_LABEL} {{id: $id, graph_name: $graph_name}})
        RETURN n.id AS id, labels(n) AS labels, properties(n) AS properties
        """
        with self.driver.session(database=self.database) as session:
            row = session.execute_read(
                lambda tx: tx.run(query, id=node_id, graph_name=self.graph_name).single()
            )
        return dict(row) if row else None

    def get_neighbors(self, node_id: str, *, depth: int = 1, limit: int = 50) -> list[dict[str, Any]]:
        if depth < 1 or depth > 3:
            raise GraphStoreError("Neighbor depth must be between 1 and 3.")
        query = f"""
        MATCH path = (n:{DEFAULT_NODE_LABEL} {{id: $id, graph_name: $graph_name}})-[*1..{depth}]-(m:{DEFAULT_NODE_LABEL})
        WHERE m.graph_name = $graph_name
        WITH path, m
        LIMIT $limit
        RETURN m.id AS id, labels(m) AS labels, properties(m) AS properties,
               [rel IN relationships(path) | {{id: rel.id, type: type(rel), properties: properties(rel)}}] AS relationships
        """
        with self.driver.session(database=self.database) as session:
            rows = session.execute_read(
                lambda tx: list(
                    tx.run(query, id=node_id, graph_name=self.graph_name, limit=int(limit))
                )
            )
        return [dict(row) for row in rows]


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    rows: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise GraphStoreError(f"JSONL row must be an object: {file_path}:{line_number}")
            rows.append(item)
    return rows


def normalize_node_properties(
    node: dict[str, Any],
    *,
    graph_name: str,
    sub_graph_name: str | None = None,
) -> dict[str, Any]:
    required = ["id", "label", "name"]
    missing = [key for key in required if not node.get(key)]
    if missing:
        raise GraphStoreError(f"Node record is missing required fields: {missing}")

    props = {
        "id": str(node["id"]),
        "label": str(node["label"]),
        "name": str(node["name"]),
        "create_at": node.get("create_at"),
        "create_by": node.get("create_by"),
        "graph_name": graph_name,
        "sub_graph_name": sub_graph_name,
    }
    props.update(flatten_properties(node.get("properties") or {}))
    return clean_neo4j_properties(props)


def normalize_relationship_properties(
    relationship: dict[str, Any],
    *,
    graph_name: str,
    sub_graph_name: str | None = None,
) -> dict[str, Any]:
    required = ["id", "type", "source_id", "target_id"]
    missing = [key for key in required if not relationship.get(key)]
    if missing:
        raise GraphStoreError(f"Relationship record is missing required fields: {missing}")

    props = {
        "id": str(relationship["id"]),
        "type": str(relationship["type"]),
        "source_id": str(relationship["source_id"]),
        "target_id": str(relationship["target_id"]),
        "create_at": relationship.get("create_at"),
        "create_by": relationship.get("create_by"),
        "graph_name": graph_name,
        "sub_graph_name": sub_graph_name,
    }
    props.update(flatten_properties(relationship.get("properties") or {}))
    return clean_neo4j_properties(props)


def flatten_properties(properties: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in properties.items():
        safe_key = str(key)
        if prefix:
            safe_key = f"{prefix}_{safe_key}"
        safe_key = safe_property_key(safe_key)
        if isinstance(value, dict):
            flattened.update(flatten_properties(value, prefix=safe_key))
        else:
            flattened[safe_key] = value
    return flattened


def clean_neo4j_properties(properties: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in properties.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        elif isinstance(value, list):
            cleaned[key] = [stringify_property_item(item) for item in value if item is not None]
        else:
            cleaned[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return cleaned


def stringify_property_item(value: Any) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def safe_property_key(value: str) -> str:
    key = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value.strip())
    key = "_".join(part for part in key.split("_") if part)
    if not key:
        raise GraphStoreError("Neo4j property key cannot be empty.")
    if key[0].isdigit():
        key = f"p_{key}"
    return key


def batched(rows: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    batch_size = max(int(batch_size), 1)
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def get_graph_store_service(
    *,
    graph_name: str | None = None,
    sub_graph_name: str | None = None,
) -> GraphStoreService:
    return GraphStoreService(graph_name=graph_name, sub_graph_name=sub_graph_name)

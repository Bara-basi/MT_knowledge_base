from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Literal

from neo4j import Query, READ_ACCESS

from app.core.config import settings
from app.db.neo4j import (
    DEFAULT_NODE_LABEL,
    KNOWN_NODE_LABELS,
    KNOWN_RELATIONSHIP_TYPES,
    get_neo4j_driver,
    validate_neo4j_identifier,
)


GraphReturnType = Literal["records", "nodes", "relationships", "paths"]
GraphDirection = Literal["outgoing", "incoming", "both"]
DEFAULT_SUB_GRAPH_NAME = "standard_product_subgraph"
_RAW_QUERY_LIMIT_PARAMETER = "__graph_result_limit"
_BLOCKED_CYPHER_CLAUSES = {
    "ALTER",
    "CALL",
    "CREATE",
    "DELETE",
    "DENY",
    "DETACH",
    "DROP",
    "FOREACH",
    "GRANT",
    "LOAD",
    "MERGE",
    "REMOVE",
    "RENAME",
    "REVOKE",
    "SET",
    "SHOW",
    "START",
    "STOP",
    "TERMINATE",
    "USE",
}


class GraphQueryError(RuntimeError):
    """Base error for graph queries."""


class GraphQueryValidationError(GraphQueryError, ValueError):
    """Raised when an unsafe or invalid graph query is requested."""


class GraphQueryService:
    """Bounded read-only graph querying plus product/standard domain lookups."""

    def __init__(
        self,
        driver: Any | None = None,
        *,
        database: str | None = None,
        graph_name: str | None = None,
        sub_graph_name: str | None = DEFAULT_SUB_GRAPH_NAME,
    ) -> None:
        self.driver = driver or get_neo4j_driver()
        self.database = database or settings.neo4j_database
        self.graph_name = graph_name or settings.neo4j_graph_name
        self.sub_graph_name = sub_graph_name
        self._owns_driver = driver is None

    def close(self) -> None:
        if self._owns_driver:
            self.driver.close()

    def execute_cypher(
        self,
        cypher: str,
        *,
        parameters: dict[str, Any] | None = None,
        return_type: GraphReturnType = "records",
        limit: int = 50,
    ) -> list[Any]:
        """Execute caller-provided read-only Cypher and cap its outer result set."""

        query = validate_read_only_cypher(cypher)
        limit = validate_limit(limit)
        params = dict(parameters or {})
        if _RAW_QUERY_LIMIT_PARAMETER in params:
            raise GraphQueryValidationError(
                f"Parameter name {_RAW_QUERY_LIMIT_PARAMETER!r} is reserved"
            )
        params[_RAW_QUERY_LIMIT_PARAMETER] = limit
        wrapped_query = (
            f"CALL () {{\n{query}\n}}\n"
            f"RETURN * LIMIT ${_RAW_QUERY_LIMIT_PARAMETER}"
        )
        records = self._run_read(wrapped_query, params)
        return project_query_records(records, return_type=return_type, limit=limit)

    def query_graph(
        self,
        *,
        start_node_ids: list[str] | None = None,
        start_labels: list[str] | None = None,
        start_properties: dict[str, Any] | None = None,
        relationship_types: list[str] | None = None,
        target_labels: list[str] | None = None,
        target_properties: dict[str, Any] | None = None,
        direction: GraphDirection = "outgoing",
        min_depth: int = 1,
        max_depth: int = 1,
        return_type: GraphReturnType = "nodes",
        limit: int = 50,
    ) -> list[Any]:
        """Run a structured graph traversal scoped to the configured logical graph."""

        limit = validate_limit(limit)
        if direction not in {"outgoing", "incoming", "both"}:
            raise GraphQueryValidationError(f"Unsupported direction: {direction}")
        if not 1 <= int(min_depth) <= int(max_depth) <= 3:
            raise GraphQueryValidationError("Traversal depth must satisfy 1 <= min <= max <= 3")

        start_labels = validate_known_identifiers(
            start_labels or [], KNOWN_NODE_LABELS, "node label"
        )
        target_labels = validate_known_identifiers(
            target_labels or [], KNOWN_NODE_LABELS, "node label"
        )
        relationship_types = validate_known_identifiers(
            relationship_types or [], KNOWN_RELATIONSHIP_TYPES, "relationship type"
        )
        relationship_pattern = ""
        if relationship_types:
            relationship_pattern = ":" + "|".join(relationship_types)
        relationship_pattern += f"*{int(min_depth)}..{int(max_depth)}"
        if direction == "outgoing":
            pattern = f"(source)-[{relationship_pattern}]->(target)"
        elif direction == "incoming":
            pattern = f"(source)<-[{relationship_pattern}]-(target)"
        else:
            pattern = f"(source)-[{relationship_pattern}]-(target)"

        query = f"""
        MATCH path = {pattern}
        WHERE source:{DEFAULT_NODE_LABEL}
          AND target:{DEFAULT_NODE_LABEL}
          AND all(node IN nodes(path) WHERE node.graph_name = $graph_name)
          AND ($sub_graph_name IS NULL OR
               all(node IN nodes(path) WHERE node.sub_graph_name = $sub_graph_name))
          AND (size($start_node_ids) = 0 OR source.id IN $start_node_ids)
          AND (size($start_labels) = 0 OR
               any(label IN labels(source) WHERE label IN $start_labels))
          AND all(key IN keys($start_properties)
                  WHERE source[key] = $start_properties[key])
          AND (size($target_labels) = 0 OR
               any(label IN labels(target) WHERE label IN $target_labels))
          AND all(key IN keys($target_properties)
                  WHERE target[key] = $target_properties[key])
        """
        query += structured_return_clause(return_type)
        params = {
            "graph_name": self.graph_name,
            "sub_graph_name": self.sub_graph_name,
            "start_node_ids": [str(value) for value in (start_node_ids or [])],
            "start_labels": start_labels,
            "start_properties": dict(start_properties or {}),
            "target_labels": target_labels,
            "target_properties": dict(target_properties or {}),
            "limit": limit,
        }
        records = self._run_read(query, params)
        return [serialize_neo4j_value(record["item"]) for record in records]

    def find_nodes_by_keyword(
        self,
        keyword: str,
        *,
        label: str,
        limit: int = 5,
    ) -> tuple[Literal["exact", "fuzzy", "none"], list[dict[str, Any]]]:
        """Find nodes by name/code/aliases, trying exact matching before fuzzy matching."""

        safe_label = validate_known_identifiers([label], KNOWN_NODE_LABELS, "node label")[0]
        cleaned_keyword = str(keyword or "").strip()
        if not cleaned_keyword:
            raise GraphQueryValidationError("keyword cannot be empty")
        limit = validate_limit(limit)
        normalized_keyword = normalize_search_text(cleaned_keyword)
        params = {
            "graph_name": self.graph_name,
            "sub_graph_name": self.sub_graph_name,
            "keyword": cleaned_keyword.casefold(),
            "normalized_keyword": normalized_keyword,
            "limit": limit,
        }
        exact_query = keyword_match_query(safe_label, fuzzy=False)
        exact_rows = self._run_read(exact_query, params)
        if exact_rows:
            return "exact", [serialize_neo4j_value(row["item"]) for row in exact_rows]

        fuzzy_query = keyword_match_query(safe_label, fuzzy=True)
        fuzzy_rows = self._run_read(fuzzy_query, params)
        if fuzzy_rows:
            return "fuzzy", [serialize_neo4j_value(row["item"]) for row in fuzzy_rows]
        return "none", []

    def find_standards_for_product(
        self,
        keyword: str,
        *,
        limit: int = 50,
    ) -> dict[str, Any]:
        match_mode, products = self.find_nodes_by_keyword(
            keyword, label="Product", limit=min(limit, 10)
        )
        if not products:
            return {
                "keyword": keyword,
                "match_mode": "none",
                "matched_products": [],
                "standards": [],
            }
        standard_nodes = self.query_graph(
            start_node_ids=[node["id"] for node in products],
            start_labels=["Product"],
            relationship_types=["apply_to"],
            target_labels=["Standard"],
            direction="outgoing",
            return_type="nodes",
            limit=limit,
        )
        standard_names = sorted(
            {str(node.get("name") or "").strip() for node in standard_nodes if node.get("name")}
        )
        return {
            "keyword": keyword,
            "match_mode": match_mode,
            "matched_products": products,
            "standards": standard_names[:limit],
        }

    def find_standard_context(
        self,
        keyword: str,
        *,
        limit: int = 50,
    ) -> dict[str, Any]:
        match_mode, standards = self.find_nodes_by_keyword(
            keyword, label="Standard", limit=min(limit, 10)
        )
        empty = {
            "keyword": keyword,
            "match_mode": "none",
            "matched_standards": [],
            "referenced_standards": [],
            "versions": [],
            "documents": [],
            "products": [],
            "sections": [],
            "tables": [],
        }
        if not standards:
            return empty

        standard_ids = [node["id"] for node in standards]
        referenced_standards = self.query_graph(
            start_node_ids=standard_ids,
            relationship_types=["reference_to"],
            target_labels=["Standard"],
            direction="outgoing",
            limit=limit,
        )
        versions = self.query_graph(
            start_node_ids=standard_ids,
            relationship_types=["has_version"],
            target_labels=["StandardVersion"],
            direction="outgoing",
            limit=limit,
        )
        documents = self.query_graph(
            start_node_ids=standard_ids,
            relationship_types=["is_about"],
            target_labels=["Document"],
            direction="incoming",
            limit=limit,
        )
        products = self.query_graph(
            start_node_ids=standard_ids,
            relationship_types=["apply_to"],
            target_labels=["Product"],
            direction="incoming",
            limit=min(limit, 20),
        )
        document_ids = [node["id"] for node in documents]
        sections: list[dict[str, Any]] = []
        tables: list[dict[str, Any]] = []
        if document_ids:
            sections = self.query_graph(
                start_node_ids=document_ids,
                start_labels=["Document"],
                relationship_types=["has_section"],
                target_labels=["Section"],
                direction="outgoing",
                limit=limit,
            )
            tables = self.query_graph(
                start_node_ids=document_ids,
                start_labels=["Document"],
                relationship_types=["has_table"],
                target_labels=["Table"],
                direction="outgoing",
                limit=limit,
            )
        return {
            "keyword": keyword,
            "match_mode": match_mode,
            "matched_standards": deduplicate_nodes(standards),
            "referenced_standards": deduplicate_nodes(referenced_standards),
            "versions": deduplicate_nodes(versions),
            "documents": deduplicate_nodes(documents),
            "products": deduplicate_nodes(products),
            "sections": deduplicate_nodes(sections),
            "tables": deduplicate_nodes(tables),
        }

    def _run_read(self, query: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        with self.driver.session(
            database=self.database,
            default_access_mode=READ_ACCESS,
        ) as session:
            return [
                dict(record)
                for record in session.run(
                    Query(query, timeout=settings.neo4j_query_timeout),
                    **parameters,
                )
            ]


def get_graph_query_service() -> GraphQueryService:
    return GraphQueryService()


def validate_read_only_cypher(cypher: str) -> str:
    query = str(cypher or "").strip()
    if not query:
        raise GraphQueryValidationError("cypher cannot be empty")
    scrubbed = scrub_cypher_literals_and_comments(query)
    if ";" in scrubbed:
        raise GraphQueryValidationError("Multiple Cypher statements are not allowed")
    words = {word.upper() for word in re.findall(r"\b[A-Za-z_]+\b", scrubbed)}
    blocked = sorted(words & _BLOCKED_CYPHER_CLAUSES)
    if blocked:
        raise GraphQueryValidationError(
            f"Only read-only Cypher is allowed; blocked clause: {blocked[0]}"
        )
    first_match = re.search(r"\b([A-Za-z_]+)\b", scrubbed)
    first_word = first_match.group(1).upper() if first_match else ""
    if first_word not in {"MATCH", "OPTIONAL", "RETURN", "UNWIND", "WITH"}:
        raise GraphQueryValidationError(
            "Cypher must start with MATCH, OPTIONAL MATCH, WITH, UNWIND, or RETURN"
        )
    if "RETURN" not in words:
        raise GraphQueryValidationError("Read-only Cypher must contain RETURN")
    return query


def scrub_cypher_literals_and_comments(cypher: str) -> str:
    value = re.sub(r"/\*.*?\*/", " ", cypher, flags=re.DOTALL)
    value = re.sub(r"//[^\r\n]*", " ", value)
    value = re.sub(r"'(?:\\.|''|[^'])*'", "''", value)
    value = re.sub(r'"(?:\\.|""|[^"])*"', '""', value)
    value = re.sub(r"`(?:``|[^`])*`", "``", value)
    return value


def validate_limit(limit: int) -> int:
    value = int(limit)
    if value < 1 or value > 200:
        raise GraphQueryValidationError("limit must be between 1 and 200")
    return value


def validate_known_identifiers(
    values: list[str], allowed: set[str], kind: str
) -> list[str]:
    result: list[str] = []
    for value in values:
        safe_value = validate_neo4j_identifier(value)
        if safe_value not in allowed:
            raise GraphQueryValidationError(f"Unknown {kind}: {safe_value}")
        if safe_value not in result:
            result.append(safe_value)
    return result


def structured_return_clause(return_type: GraphReturnType) -> str:
    node_projection = (
        "{id: target.id, name: target.name, labels: labels(target), "
        "properties: properties(target)}"
    )
    relationship_projection = (
        "{id: relationship.id, type: type(relationship), "
        "source_id: startNode(relationship).id, target_id: endNode(relationship).id, "
        "properties: properties(relationship)}"
    )
    if return_type == "nodes":
        return f"\nWITH DISTINCT target LIMIT $limit\nRETURN {node_projection} AS item"
    if return_type == "relationships":
        return (
            "\nUNWIND relationships(path) AS relationship\n"
            "WITH DISTINCT relationship LIMIT $limit\n"
            f"RETURN {relationship_projection} AS item"
        )
    if return_type == "paths":
        return (
            "\nWITH DISTINCT path LIMIT $limit\n"
            "RETURN {"
            "nodes: [node IN nodes(path) | {id: node.id, name: node.name, "
            "labels: labels(node), properties: properties(node)}], "
            "relationships: [relationship IN relationships(path) | "
            f"{relationship_projection}]"
            "} AS item"
        )
    if return_type == "records":
        return (
            "\nWITH DISTINCT source, target, path LIMIT $limit\n"
            "RETURN {"
            "source: {id: source.id, name: source.name, labels: labels(source), "
            "properties: properties(source)}, "
            f"target: {node_projection}, "
            "relationships: [relationship IN relationships(path) | "
            f"{relationship_projection}]"
            "} AS item"
        )
    raise GraphQueryValidationError(f"Unsupported return_type: {return_type}")


def keyword_match_query(label: str, *, fuzzy: bool) -> str:
    normalized_name = cypher_normalized_text("node.name")
    normalized_code = cypher_normalized_text("node.code")
    normalized_alias = cypher_normalized_text("alias")
    if fuzzy:
        predicate = f"""
        ({normalized_name} CONTAINS $normalized_keyword OR
         $normalized_keyword CONTAINS {normalized_name} OR
         {normalized_code} CONTAINS $normalized_keyword OR
         any(alias IN search_aliases WHERE
             {normalized_alias} CONTAINS $normalized_keyword OR
             $normalized_keyword CONTAINS {normalized_alias}))
        """
        score = f"""
        CASE
          WHEN {normalized_name} STARTS WITH $normalized_keyword THEN 80
          WHEN {normalized_code} STARTS WITH $normalized_keyword THEN 75
          WHEN any(alias IN search_aliases WHERE
                   {normalized_alias} STARTS WITH $normalized_keyword) THEN 70
          WHEN {normalized_name} CONTAINS $normalized_keyword THEN 60
          WHEN {normalized_code} CONTAINS $normalized_keyword THEN 55
          ELSE 40
        END
        """
    else:
        predicate = f"""
        (toLower(trim(coalesce(node.name, ''))) = $keyword OR
         toLower(trim(coalesce(node.code, ''))) = $keyword OR
         any(alias IN search_aliases WHERE
             toLower(trim(alias)) = $keyword) OR
         {normalized_name} = $normalized_keyword OR
         {normalized_code} = $normalized_keyword OR
         any(alias IN search_aliases WHERE
             {normalized_alias} = $normalized_keyword))
        """
        score = f"""
        CASE
          WHEN toLower(trim(coalesce(node.name, ''))) = $keyword THEN 100
          WHEN toLower(trim(coalesce(node.code, ''))) = $keyword THEN 98
          WHEN any(alias IN search_aliases WHERE
                   toLower(trim(alias)) = $keyword) THEN 95
          ELSE 90
        END
        """
    return f"""
    MATCH (node:{DEFAULT_NODE_LABEL}:{label})
    WHERE node.graph_name = $graph_name
      AND ($sub_graph_name IS NULL OR node.sub_graph_name = $sub_graph_name)
    WITH node,
         coalesce(node.aliases, []) +
         CASE WHEN properties(node)['alias'] IS NULL
              THEN [] ELSE [toString(properties(node)['alias'])] END +
         CASE WHEN node.chinese_name IS NULL
              THEN [] ELSE [toString(node.chinese_name)] END +
         CASE WHEN node.category IS NULL
              THEN [] ELSE [toString(node.category)] END
         AS search_aliases
    WHERE {predicate}
    WITH node, {score} AS score
    ORDER BY score DESC, size(coalesce(node.name, '')) ASC, node.name ASC
    LIMIT $limit
    RETURN {{id: node.id, name: node.name, labels: labels(node),
             properties: properties(node)}} AS item
    """


def cypher_normalized_text(expression: str) -> str:
    result = f"toLower(trim(coalesce({expression}, '')))"
    for character in (" ", "-", "_", "/", ".", "(", ")"):
        result = f"replace({result}, '{character}', '')"
    return result


def normalize_search_text(value: str) -> str:
    return re.sub(r"[\s\-_/().]+", "", value.casefold())


def project_query_records(
    records: list[dict[str, Any]],
    *,
    return_type: GraphReturnType,
    limit: int,
) -> list[Any]:
    if return_type == "records":
        return [serialize_neo4j_value(record) for record in records[:limit]]
    singular = {
        "nodes": "node",
        "relationships": "relationship",
        "paths": "path",
    }[return_type]
    collected: list[Any] = []
    for record in records:
        for value in record.values():
            collect_neo4j_kind(value, singular, collected)
            if len(collected) >= limit:
                break
        if len(collected) >= limit:
            break
    return deduplicate_serialized(collected)[:limit]


def collect_neo4j_kind(value: Any, kind: str, output: list[Any]) -> None:
    value_kind = neo4j_value_kind(value)
    if value_kind == kind:
        output.append(serialize_neo4j_value(value))
        return
    if isinstance(value, Mapping):
        for child in value.values():
            collect_neo4j_kind(child, kind, output)
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            collect_neo4j_kind(child, kind, output)


def neo4j_value_kind(value: Any) -> str | None:
    if hasattr(value, "nodes") and hasattr(value, "relationships"):
        return "path"
    if hasattr(value, "start_node") and hasattr(value, "end_node"):
        return "relationship"
    if hasattr(value, "labels") and isinstance(value, Mapping):
        return "node"
    return None


def serialize_neo4j_value(value: Any) -> Any:
    kind = neo4j_value_kind(value)
    if kind == "node":
        properties = dict(value)
        return {
            "id": str(properties.get("id") or getattr(value, "element_id", "")),
            "name": str(properties.get("name") or ""),
            "labels": sorted(str(label) for label in value.labels),
            "properties": serialize_neo4j_value(properties),
        }
    if kind == "relationship":
        properties = dict(value)
        start_properties = dict(value.start_node)
        end_properties = dict(value.end_node)
        return {
            "id": str(properties.get("id") or getattr(value, "element_id", "")),
            "type": str(getattr(value, "type", value.__class__.__name__)),
            "source_id": str(
                properties.get("source_id")
                or start_properties.get("id")
                or getattr(value.start_node, "element_id", "")
            ),
            "target_id": str(
                properties.get("target_id")
                or end_properties.get("id")
                or getattr(value.end_node, "element_id", "")
            ),
            "properties": serialize_neo4j_value(properties),
        }
    if kind == "path":
        return {
            "nodes": [serialize_neo4j_value(node) for node in value.nodes],
            "relationships": [
                serialize_neo4j_value(relationship) for relationship in value.relationships
            ],
        }
    if isinstance(value, Mapping):
        return {str(key): serialize_neo4j_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialize_neo4j_value(child) for child in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "iso_format"):
        return value.iso_format()
    if hasattr(value, "to_native"):
        return serialize_neo4j_value(value.to_native())
    return str(value)


def deduplicate_serialized(values: list[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        marker = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result


def deduplicate_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node in nodes:
        node_id = str(node.get("id") or "")
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        result.append(node)
    return result

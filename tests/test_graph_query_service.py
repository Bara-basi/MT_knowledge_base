from __future__ import annotations

import pytest

from app.services.graph.query_service import (
    GraphQueryService,
    GraphQueryValidationError,
    keyword_match_query,
    normalize_search_text,
    validate_read_only_cypher,
)


class FakeTransaction:
    def __init__(self, rows):
        self.rows = rows
        self.query = ""
        self.parameters = {}

    def run(self, query, **parameters):
        self.query = getattr(query, "text", str(query))
        self.parameters = parameters
        return self.rows


class FakeSession:
    def __init__(self, transaction):
        self.transaction = transaction

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute_read(self, callback):
        return callback(self.transaction)

    def run(self, query, **parameters):
        return self.transaction.run(query, **parameters)


class FakeDriver:
    def __init__(self, rows):
        self.transaction = FakeTransaction(rows)

    def session(self, **kwargs):
        return FakeSession(self.transaction)


def test_read_only_cypher_rejects_mutation_and_multiple_statements() -> None:
    assert validate_read_only_cypher("MATCH (n) RETURN n") == "MATCH (n) RETURN n"
    with pytest.raises(GraphQueryValidationError, match="blocked clause: DELETE"):
        validate_read_only_cypher("MATCH (n) DELETE n RETURN n")
    with pytest.raises(GraphQueryValidationError, match="Multiple"):
        validate_read_only_cypher("MATCH (n) RETURN n; MATCH (m) RETURN m")


def test_read_only_cypher_ignores_keywords_inside_literals_and_comments() -> None:
    query = "MATCH (n) WHERE n.name = 'delete set call' // CREATE x\nRETURN n"
    assert validate_read_only_cypher(query) == query


def test_execute_cypher_wraps_query_with_hard_result_limit() -> None:
    driver = FakeDriver([{"value": 1}, {"value": 2}])
    service = GraphQueryService(driver=driver)

    result = service.execute_cypher("RETURN 1 AS value", limit=1)

    assert result == [{"value": 1}]
    assert "CALL () {" in driver.transaction.query
    assert "RETURN * LIMIT $__graph_result_limit" in driver.transaction.query
    assert driver.transaction.parameters["__graph_result_limit"] == 1


def test_normalized_search_ignores_common_standard_separators() -> None:
    assert normalize_search_text("SA-312 / SA-312M") == "sa312sa312m"


def test_keyword_search_includes_chinese_name_and_category() -> None:
    query = keyword_match_query("Product", fuzzy=False)

    assert "node.chinese_name" in query
    assert "node.category" in query


class StubDomainService:
    def __init__(self, label: str):
        self.label = label
        self.calls: list[dict] = []

    def find_nodes_by_keyword(self, keyword, *, label, limit):
        node = {"id": f"{label.lower()}:1", "name": keyword, "labels": [label], "properties": {}}
        return "exact", [node]

    def query_graph(self, **kwargs):
        self.calls.append(kwargs)
        relationship_types = kwargs.get("relationship_types") or []
        mapping = {
            "apply_to": [{"id": "standard:1", "name": "SA-312", "labels": ["Standard"], "properties": {}}],
            "reference_to": [{"id": "standard:2", "name": "SA-999", "labels": ["Standard"], "properties": {}}],
            "has_version": [{"id": "version:1", "name": "2023", "labels": ["StandardVersion"], "properties": {}}],
            "is_about": [{"id": "document:1", "name": "SA-312 PDF", "labels": ["Document"], "properties": {}}],
            "has_section": [{"id": "section:1", "name": "Scope", "labels": ["Section"], "properties": {}}],
            "has_table": [{"id": "table:1", "name": "Table 1", "labels": ["Table"], "properties": {}}],
        }
        if relationship_types == ["apply_to"] and kwargs.get("direction") == "incoming":
            return [{"id": "product:1", "name": "Pipe", "labels": ["Product"], "properties": {}}]
        return mapping.get(relationship_types[0], [])


def test_product_lookup_builds_on_bounded_general_traversal() -> None:
    service = StubDomainService("Product")

    result = GraphQueryService.find_standards_for_product(service, "Pipe", limit=20)

    assert result["standards"] == ["SA-312"]
    assert service.calls == [
        {
            "start_node_ids": ["product:1"],
            "start_labels": ["Product"],
            "relationship_types": ["apply_to"],
            "target_labels": ["Standard"],
            "direction": "outgoing",
            "return_type": "nodes",
            "limit": 20,
        }
    ]


def test_standard_context_uses_real_edge_directions_and_separate_content_limits() -> None:
    service = StubDomainService("Standard")

    result = GraphQueryService.find_standard_context(service, "SA-312", limit=10)

    assert [node["name"] for node in result["documents"]] == ["SA-312 PDF"]
    assert [node["name"] for node in result["sections"]] == ["Scope"]
    assert [node["name"] for node in result["tables"]] == ["Table 1"]
    directions = {
        call["relationship_types"][0]: call["direction"] for call in service.calls
    }
    assert directions["reference_to"] == "outgoing"
    assert directions["has_version"] == "outgoing"
    assert directions["is_about"] == "incoming"
    assert directions["apply_to"] == "incoming"
    assert directions["has_section"] == "outgoing"
    assert directions["has_table"] == "outgoing"

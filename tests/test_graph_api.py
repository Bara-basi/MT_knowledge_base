from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from neo4j.exceptions import ServiceUnavailable

from app.api.v1.graph import graph_query_service_dependency, router
from app.services.graph.query_service import GraphQueryValidationError


class FakeGraphQueryService:
    def execute_cypher(self, cypher, *, parameters, return_type, limit):
        if "DELETE" in cypher:
            raise GraphQueryValidationError("Only read-only Cypher is allowed")
        return [{"answer": 1}]

    def query_graph(self, **kwargs):
        return [{"id": "standard:1", "name": "SA-312", "labels": ["Standard"], "properties": {}}]

    def find_standards_for_product(self, keyword, *, limit):
        return {
            "keyword": keyword,
            "match_mode": "exact",
            "matched_products": [
                {"id": "product:1", "name": "Pipe", "labels": ["Product"], "properties": {}}
            ],
            "standards": ["SA-312"],
        }

    def find_standard_context(self, keyword, *, limit):
        return {
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


class UnavailableGraphQueryService(FakeGraphQueryService):
    def query_graph(self, **kwargs):
        raise ServiceUnavailable("connection refused")


def make_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    service = FakeGraphQueryService()
    app.dependency_overrides[graph_query_service_dependency] = lambda: service
    return TestClient(app)


def make_unavailable_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    service = UnavailableGraphQueryService()
    app.dependency_overrides[graph_query_service_dependency] = lambda: service
    return TestClient(app, raise_server_exceptions=False)


def test_graph_query_endpoint_supports_direct_cypher() -> None:
    response = make_client().post(
        "/graph/query",
        json={"cypher": "RETURN 1 AS answer", "return_type": "records", "limit": 5},
    )

    assert response.status_code == 200
    assert response.json() == {
        "return_type": "records",
        "count": 1,
        "items": [{"answer": 1}],
    }


def test_graph_query_endpoint_maps_validation_errors_to_400() -> None:
    response = make_client().post(
        "/graph/query",
        json={"cypher": "MATCH (n) DELETE n RETURN n", "return_type": "records"},
    )

    assert response.status_code == 400


def test_product_standard_endpoint_returns_only_standard_names() -> None:
    response = make_client().post(
        "/graph/product-standards",
        json={"keyword": "Pipe", "limit": 20},
    )

    assert response.status_code == 200
    assert response.json()["standards"] == ["SA-312"]


def test_graph_query_endpoint_maps_driver_connection_errors_to_503() -> None:
    response = make_unavailable_client().post(
        "/graph/query",
        json={
            "start_labels": ["Product"],
            "relationship_types": ["apply_to"],
            "target_labels": ["Standard"],
        },
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Neo4j graph query failed"}

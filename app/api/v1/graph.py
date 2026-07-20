from __future__ import annotations

from collections.abc import Generator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from neo4j.exceptions import DriverError, Neo4jError

from app.schemas.graph import (
    GraphQueryRequest,
    GraphQueryResponse,
    KeywordGraphRequest,
    ProductStandardsResponse,
    StandardContextResponse,
)
from app.services.graph.query_service import (
    GraphQueryService,
    GraphQueryValidationError,
    get_graph_query_service,
)


router = APIRouter(prefix="/graph", tags=["graph"])


def graph_query_service_dependency() -> Generator[GraphQueryService, None, None]:
    service = get_graph_query_service()
    try:
        yield service
    finally:
        service.close()


GraphQueryServiceDependency = Annotated[
    GraphQueryService, Depends(graph_query_service_dependency)
]


@router.post(
    "/query",
    response_model=GraphQueryResponse,
    response_model_exclude_none=True,
)
def query_graph(
    request: GraphQueryRequest,
    service: GraphQueryServiceDependency,
) -> GraphQueryResponse:
    """Execute a bounded structured traversal or caller-provided read-only Cypher."""

    try:
        if request.cypher:
            items = service.execute_cypher(
                request.cypher,
                parameters=request.parameters,
                return_type=request.return_type,
                limit=request.limit,
            )
        else:
            items = service.query_graph(
                start_node_ids=request.start_node_ids,
                start_labels=request.start_labels,
                start_properties=request.start_properties,
                relationship_types=request.relationship_types,
                target_labels=request.target_labels,
                target_properties=request.target_properties,
                direction=request.direction,
                min_depth=request.min_depth,
                max_depth=request.max_depth,
                return_type=request.return_type,
                limit=request.limit,
            )
    except (GraphQueryValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (DriverError, Neo4jError) as exc:
        raise HTTPException(status_code=503, detail="Neo4j graph query failed") from exc
    return GraphQueryResponse(
        return_type=request.return_type,
        count=len(items),
        items=items,
    )


@router.post(
    "/product-standards",
    response_model=ProductStandardsResponse,
    response_model_exclude_none=True,
)
def find_product_standards(
    request: KeywordGraphRequest,
    service: GraphQueryServiceDependency,
) -> ProductStandardsResponse:
    """Resolve a product (name or alias) and return its directly applicable standards."""

    try:
        result = service.find_standards_for_product(
            request.keyword,
            limit=request.limit,
        )
    except (GraphQueryValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (DriverError, Neo4jError) as exc:
        raise HTTPException(status_code=503, detail="Neo4j graph query failed") from exc
    return ProductStandardsResponse.model_validate(result)


@router.post(
    "/standard-context",
    response_model=StandardContextResponse,
    response_model_exclude_none=True,
)
def find_standard_context(
    request: KeywordGraphRequest,
    service: GraphQueryServiceDependency,
) -> StandardContextResponse:
    """Resolve a standard and return tightly bounded related graph content."""

    try:
        result = service.find_standard_context(
            request.keyword,
            limit=request.limit,
        )
    except (GraphQueryValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (DriverError, Neo4jError) as exc:
        raise HTTPException(status_code=503, detail="Neo4j graph query failed") from exc
    return StandardContextResponse.model_validate(result)

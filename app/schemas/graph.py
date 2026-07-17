from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


GraphReturnType = Literal["records", "nodes", "relationships", "paths"]
GraphDirection = Literal["outgoing", "incoming", "both"]


class GraphQueryRequest(BaseModel):
    """A read-only Cypher query or a bounded structured graph traversal."""

    cypher: str | None = Field(
        None,
        min_length=1,
        description="Optional read-only Cypher. Mutating clauses and procedures are rejected.",
    )
    parameters: dict[str, Any] = Field(default_factory=dict)
    start_node_ids: list[str] = Field(default_factory=list, max_length=100)
    start_labels: list[str] = Field(default_factory=list, max_length=10)
    start_properties: dict[str, Any] = Field(default_factory=dict)
    relationship_types: list[str] = Field(default_factory=list, max_length=20)
    target_labels: list[str] = Field(default_factory=list, max_length=10)
    target_properties: dict[str, Any] = Field(default_factory=dict)
    direction: GraphDirection = "outgoing"
    min_depth: int = Field(1, ge=1, le=3)
    max_depth: int = Field(1, ge=1, le=3)
    return_type: GraphReturnType = "nodes"
    limit: int = Field(50, ge=1, le=200)

    @model_validator(mode="after")
    def validate_query_mode(self) -> "GraphQueryRequest":
        if self.min_depth > self.max_depth:
            raise ValueError("min_depth cannot exceed max_depth")
        structured_fields = (
            self.start_node_ids,
            self.start_labels,
            self.start_properties,
            self.relationship_types,
            self.target_labels,
            self.target_properties,
        )
        if self.cypher and any(structured_fields):
            raise ValueError("cypher cannot be combined with structured traversal fields")
        if not self.cypher and self.parameters:
            raise ValueError("parameters can only be used with cypher")
        return self


class GraphQueryResponse(BaseModel):
    return_type: GraphReturnType
    count: int
    items: list[Any]


class KeywordGraphRequest(BaseModel):
    keyword: str = Field(..., min_length=1, max_length=200)
    limit: int = Field(20, ge=1, le=100)


class GraphNodeResult(BaseModel):
    id: str
    name: str
    labels: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)


class ProductStandardsResponse(BaseModel):
    keyword: str
    match_mode: Literal["exact", "fuzzy", "none"]
    matched_products: list[GraphNodeResult]
    standards: list[str]


class StandardContextResponse(BaseModel):
    keyword: str
    match_mode: Literal["exact", "fuzzy", "none"]
    matched_standards: list[GraphNodeResult]
    referenced_standards: list[GraphNodeResult]
    versions: list[GraphNodeResult]
    documents: list[GraphNodeResult]
    products: list[GraphNodeResult]
    sections: list[GraphNodeResult]
    tables: list[GraphNodeResult]

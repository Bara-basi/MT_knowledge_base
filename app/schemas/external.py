from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


QuoteDimensionName = Literal[
    "询价完整度",
    "询价供应商准确度",
    "询价命名规范度",
    "计算准确度",
    "报价完整度",
    "报价及时性",
]


class ExternalQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=8_000)
    user_id: str = Field(..., min_length=1, max_length=512)
    service_id: str = Field(..., min_length=1, max_length=128)
    session_id: str = Field(..., min_length=1, max_length=512)
    use_lark_document: bool = Field(
        False,
        description=(
            "Prefer mapped Feishu/Lark document URLs for knowledge sources. "
            "When false, sources use the backend document-download endpoint."
        ),
    )
    format_type: Literal["markdown", "json"] = "markdown"

    @field_validator("question", "user_id", "service_id", "session_id")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value must not be blank")
        return cleaned


class ExternalQueryResponse(BaseModel):
    question: str
    answer: str | dict[str, Any] | list[Any]
    user_id: str
    service_id: str
    session_id: str
    topic_id: str | None = None
    answer_format: Literal["markdown", "json"] = "markdown"
    status: Literal["success"] = "success"


class QuoteDeductionItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    dimension: QuoteDimensionName = Field(alias="评分维度")
    reason: str = Field(..., min_length=1, max_length=1_000, alias="扣分原因")
    points: int = Field(..., ge=-100, le=-1, alias="扣分")


class QuoteDimensionScores(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    inquiry_completeness: int = Field(..., ge=0, le=100, alias="询价完整度")
    supplier_accuracy: int = Field(..., ge=0, le=100, alias="询价供应商准确度")
    inquiry_naming: int = Field(..., ge=0, le=100, alias="询价命名规范度")
    calculation_accuracy: int = Field(..., ge=0, le=100, alias="计算准确度")
    quotation_completeness: int = Field(..., ge=0, le=100, alias="报价完整度")
    quotation_timeliness: int = Field(..., ge=0, le=100, alias="报价及时性")


class QuoteScoreResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    total_score: int = Field(..., ge=0, le=100, alias="总分")
    full_score: Literal[100] = Field(..., alias="满分")
    dimensions: QuoteDimensionScores = Field(..., alias="评分维度")
    deductions: list[QuoteDeductionItem] = Field(..., alias="扣分项")

    @model_validator(mode="after")
    def recalculate_scores(self) -> "QuoteScoreResult":
        # Timeliness is deliberately out of scope for this version.
        unique_deductions: list[QuoteDeductionItem] = []
        seen_deductions: set[tuple[str, str, int]] = set()
        for item in self.deductions:
            if item.dimension == "报价及时性":
                continue
            item.reason = item.reason.strip()
            key = (item.dimension, item.reason.casefold(), item.points)
            if key in seen_deductions:
                continue
            seen_deductions.add(key)
            unique_deductions.append(item)
        self.deductions = unique_deductions
        allowed_points = {
            "询价完整度": {-5, -1},
            "询价供应商准确度": {-5},
            "询价命名规范度": {-5},
            "计算准确度": {-2},
            "报价完整度": {-5, -3, -1},
        }
        for item in self.deductions:
            if item.points not in allowed_points[item.dimension]:
                raise ValueError(
                    f"{item.dimension} deduction {item.points} does not match the scoring rules"
                )
        deducted_by_dimension: dict[str, int] = {}
        for item in self.deductions:
            deducted_by_dimension[item.dimension] = (
                deducted_by_dimension.get(item.dimension, 0) + abs(item.points)
            )

        self.dimensions = QuoteDimensionScores(
            **{
                "询价完整度": max(0, 100 - deducted_by_dimension.get("询价完整度", 0)),
                "询价供应商准确度": max(
                    0,
                    100 - deducted_by_dimension.get("询价供应商准确度", 0),
                ),
                "询价命名规范度": max(
                    0,
                    100 - deducted_by_dimension.get("询价命名规范度", 0),
                ),
                "计算准确度": max(0, 100 - deducted_by_dimension.get("计算准确度", 0)),
                "报价完整度": max(0, 100 - deducted_by_dimension.get("报价完整度", 0)),
                "报价及时性": 100,
            }
        )
        self.total_score = max(
            0,
            100 - sum(abs(item.points) for item in self.deductions),
        )
        self.full_score = 100
        return self


class QuoteScoreResponse(QuoteScoreResult):
    user_id: str
    service_id: str
    session_id: str
    file_name: str | None = None
    topic_id: str | None = None
    format_type: Literal["json"] = "json"
    status: Literal["success"] = "success"

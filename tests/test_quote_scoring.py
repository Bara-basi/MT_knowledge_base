from __future__ import annotations

import asyncio
from io import BytesIO
import json

from fastapi import UploadFile
from openpyxl import Workbook
import pytest

from app.api.v1 import external
from app.schemas.external import ExternalQueryRequest
from app.schemas.query import QueryResponse
from app.services.quote_scoring import (
    QUOTE_SCORING_PROMPT,
    parse_json_answer,
    parse_quote_score,
    spreadsheet_to_compact_json,
)


def test_parse_json_answer_accepts_fenced_object() -> None:
    assert parse_json_answer('```json\n{"ok":true}\n```') == {"ok": True}


def test_quote_score_is_recalculated_from_deductions() -> None:
    result = parse_quote_score(
        json.dumps(
            {
                "总分": 99,
                "满分": 100,
                "评分维度": {
                    "询价完整度": 99,
                    "询价供应商准确度": 99,
                    "询价命名规范度": 99,
                    "计算准确度": 99,
                    "报价完整度": 99,
                    "报价及时性": 1,
                },
                "扣分项": [
                    {"评分维度": "计算准确度", "扣分原因": "公式错误", "扣分": -2},
                    {"评分维度": "报价完整度", "扣分原因": "缺少数量列", "扣分": -5},
                    {"评分维度": "报价及时性", "扣分原因": "超时", "扣分": -10},
                ],
            },
            ensure_ascii=False,
        )
    )

    assert result.total_score == 93
    assert result.dimensions.calculation_accuracy == 98
    assert result.dimensions.quotation_completeness == 95
    assert result.dimensions.quotation_timeliness == 100
    assert len(result.deductions) == 2
    payload = result.model_dump(by_alias=True)
    assert payload["总分"] == 93
    assert payload["评分维度"]["报价及时性"] == 100
    assert payload["扣分项"][0]["扣分"] == -2


def test_quote_score_rejects_non_rubric_deduction_value() -> None:
    with pytest.raises(ValueError, match="does not match the scoring rules"):
        parse_quote_score(
            json.dumps(
                {
                    "总分": 96,
                    "满分": 100,
                    "评分维度": {
                        "询价完整度": 100,
                        "询价供应商准确度": 100,
                        "询价命名规范度": 100,
                        "计算准确度": 96,
                        "报价完整度": 100,
                        "报价及时性": 100,
                    },
                    "扣分项": [
                        {"评分维度": "计算准确度", "扣分原因": "合并扣分", "扣分": -4}
                    ],
                },
                ensure_ascii=False,
            )
        )


def test_quote_score_deduplicates_identical_model_deductions() -> None:
    dimensions = {
        "询价完整度": 100,
        "询价供应商准确度": 100,
        "询价命名规范度": 100,
        "计算准确度": 100,
        "报价完整度": 100,
        "报价及时性": 100,
    }
    result = parse_quote_score(
        json.dumps(
            {
                "总分": 96,
                "满分": 100,
                "评分维度": dimensions,
                "扣分项": [
                    {"评分维度": "计算准确度", "扣分原因": "汇率错误", "扣分": -2},
                    {"评分维度": "计算准确度", "扣分原因": " 汇率错误 ", "扣分": -2},
                ],
            },
            ensure_ascii=False,
        )
    )

    assert result.total_score == 98
    assert result.dimensions.calculation_accuracy == 98
    assert len(result.deductions) == 1


def test_xlsx_is_converted_to_compact_json_with_formulas() -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "报价"
    sheet.append(["产品", "数量", "单价", "总价"])
    sheet.append(["钢管", 2, 10.5, "=B2*C2"])
    buffer = BytesIO()
    workbook.save(buffer)
    workbook.close()

    parsed = json.loads(
        spreadsheet_to_compact_json(
            filename="报价.xlsx",
            content=buffer.getvalue(),
        )
    )

    assert parsed["file_name"] == "报价.xlsx"
    assert parsed["sheets"][0]["rows"][0]["cells"]["A"] == "产品"
    assert parsed["sheets"][0]["rows"][1]["cells"]["D"] == {"formula": "=B2*C2"}


def test_external_json_format_returns_parsed_json(monkeypatch) -> None:
    async def fake_execute(request, **kwargs):
        assert request.format_type == "json"
        assert "合法 JSON" in kwargs["additional_system_prompt"]
        return QueryResponse(question=request.question, answer='{"score":88}')

    monkeypatch.setattr(external, "_execute_external_query", fake_execute)
    response = asyncio.run(
        external.query_external_knowledge_base(
            ExternalQueryRequest(
                question="返回评分",
                user_id="u1",
                service_id="crm",
                session_id="s1",
                format_type="json",
            )
        )
    )

    assert response.answer == {"score": 88}
    assert response.answer_format == "json"


def test_quote_score_endpoint_passes_uploaded_sheet_as_task_data(monkeypatch) -> None:
    workbook = Workbook()
    workbook.active.append(["产品", "单价"])
    workbook.active.append(["钢管", 10])
    buffer = BytesIO()
    workbook.save(buffer)
    workbook.close()
    upload = UploadFile(filename="quote.xlsx", file=BytesIO(buffer.getvalue()))
    captured: dict = {}

    async def fake_execute(request, **kwargs):
        captured["request"] = request
        captured.update(kwargs)
        return QueryResponse(
            question=request.question,
            answer=json.dumps(
                {
                    "总分": 98,
                    "满分": 100,
                    "评分维度": {
                        "询价完整度": 100,
                        "询价供应商准确度": 100,
                        "询价命名规范度": 100,
                        "计算准确度": 98,
                        "报价完整度": 100,
                        "报价及时性": 100,
                    },
                    "扣分项": [
                        {"评分维度": "计算准确度", "扣分原因": "公式错误", "扣分": -2}
                    ],
                },
                ensure_ascii=False,
            ),
        )

    monkeypatch.setattr(external, "_execute_external_query", fake_execute)
    response = asyncio.run(
        external.score_external_quote(
            question="请评分",
            user_id="u1",
            service_id="crm",
            session_id="s1",
            use_lark_document=False,
            file=upload,
        )
    )

    assert response.total_score == 98
    assert response.file_name == "quote.xlsx"
    assert captured["tool_name"] == "quote_scoring"
    assert captured["additional_system_prompt"] == QUOTE_SCORING_PROMPT
    task_input = json.loads(captured["task_input"])
    assert task_input["sheets"][0]["rows"][1]["cells"]["A"] == "钢管"

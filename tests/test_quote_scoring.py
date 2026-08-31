from __future__ import annotations

import asyncio
from io import BytesIO
import json
from uuid import UUID

from fastapi import HTTPException, UploadFile
from openpyxl import Workbook
import pytest

from app.api.v1 import external
from app.schemas.external import ExternalQueryRequest
from app.schemas.query import QueryResponse
from app.services.quote_scoring import (
    QUOTE_SCORING_PROMPT,
    _quote_formula_audit_context,
    finalize_quote_score_json,
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


def test_quote_score_drops_explicit_zero_point_non_deductions() -> None:
    result = parse_quote_score(
        json.dumps(
            {
                "扣分项": [
                    {
                        "评分维度": "报价完整度",
                        "扣分原因": "价格可见格式已经保留两位小数，不扣分",
                        "扣分": 0,
                    },
                    {
                        "评分维度": "计算准确度",
                        "扣分原因": "支数公式错误",
                        "扣分": -2,
                    },
                ]
            },
            ensure_ascii=False,
        )
    )

    assert result.total_score == 98
    assert len(result.deductions) == 1
    assert result.deductions[0].points == -2


def test_quote_score_builds_deterministic_scores_from_deductions_only() -> None:
    result = parse_quote_score(
        json.dumps(
            {
                "扣分项": [
                    {
                        "评分维度": "计算准确度",
                        "扣分原因": "总价与数量乘以单价不一致",
                        "扣分": -2,
                    },
                    {
                        "评分维度": "报价完整度",
                        "扣分原因": "付款方式遗漏",
                        "扣分": -3,
                    },
                ]
            },
            ensure_ascii=False,
        )
    )

    payload = result.model_dump(by_alias=True)
    assert payload["总分"] == 95
    assert payload["满分"] == 100
    assert payload["评分维度"]["计算准确度"] == 98
    assert payload["评分维度"]["报价完整度"] == 97
    assert payload["评分维度"]["报价及时性"] == 100


def test_quote_scoring_prompt_uses_visible_precision_and_material_terms() -> None:
    assert "以单元格的\n可见格式为准" in QUOTE_SCORING_PROMPT
    assert "不得因为底层 `value` 含更多小数而扣分" in QUOTE_SCORING_PROMPT
    assert "不得仅因英文拼写、语法" in QUOTE_SCORING_PROMPT
    assert "业务含义仍然明确" in QUOTE_SCORING_PROMPT
    assert "同一根因被复制到多个产品行" in QUOTE_SCORING_PROMPT
    assert "不得为了重复行而重复扣分" in QUOTE_SCORING_PROMPT
    assert "必须先逐一检查文件中提供的公式" in QUOTE_SCORING_PROMPT
    assert "不能只检查最终显示金额" in QUOTE_SCORING_PROMPT
    assert "不自动规定付款比例或尾款支付节点" in QUOTE_SCORING_PROMPT
    assert "不得推断 CIF 必须凭提单副本支付尾款" in QUOTE_SCORING_PROMPT
    assert "证据门控规则" in QUOTE_SCORING_PROMPT
    assert "不是要求逐项寻找理由扣分" in QUOTE_SCORING_PROMPT
    assert "就不能判断与这些外部基准不一致" in QUOTE_SCORING_PROMPT
    assert "绝不输出扣分为 0" in QUOTE_SCORING_PROMPT
    assert "报价单价×报价数量=报价合计" in QUOTE_SCORING_PROMPT
    assert "不能把公斤价直接填入以米或支为单位" in QUOTE_SCORING_PROMPT
    assert "只汇总管件、漏掉钢管" in QUOTE_SCORING_PROMPT


def test_quote_score_finalizer_uses_json_mode_and_recalculates_scores() -> None:
    captured: dict = {}

    class FakeClient:
        def chat(self, messages, **kwargs):
            captured["messages"] = messages
            captured.update(kwargs)
            return json.dumps(
                {
                    "扣分项": [
                        {
                            "评分维度": "计算准确度",
                            "扣分原因": "公式错误",
                            "扣分": -2,
                        }
                    ]
                },
                ensure_ascii=False,
            )

    finalized = finalize_quote_score_json(
        task_input='{"sheets":[]}',
        harness_answer="草稿可能错误",
        client=FakeClient(),
    )
    payload = json.loads(finalized)

    assert captured["extra_body"] == {
        "response_format": {"type": "json_object"},
        "enable_thinking": False,
    }
    assert captured["max_tokens"] == 4_096
    assert captured["temperature"] == 0
    assert "必须亲自核对" in captured["messages"][0]["content"]
    assert "草稿可能错误" in captured["messages"][1]["content"]
    assert "<mtsco-formula-audit-context>" in captured["messages"][1]["content"]
    assert payload["总分"] == 98
    assert payload["评分维度"]["计算准确度"] == 98


def test_quote_formula_audit_context_adds_headers_and_row_values() -> None:
    task_input = json.dumps(
        {
            "sheets": [
                {
                    "name": "报价",
                    "rows": [
                        {
                            "row": 11,
                            "cells": {
                                "H": "Length mm/PC",
                                "J": "Qty",
                                "K": "UOM",
                                "S": "支数",
                            },
                        },
                        {
                            "row": 12,
                            "cells": {
                                "H": 6096,
                                "J": 1250,
                                "K": "M",
                                "S": {"formula": "=J12/20", "value": 62.5},
                            },
                        },
                    ],
                }
            ]
        },
        ensure_ascii=False,
    )

    audit = json.loads(_quote_formula_audit_context(task_input))

    assert audit[0]["row"] == 12
    assert audit[0]["formulas"][0]["cell"] == "S12"
    assert audit[0]["formulas"][0]["header"] == "支数"
    context = {item["cell"]: item for item in audit[0]["cells"]}
    assert context["H12"] == {"cell": "H12", "header": "Length mm/PC", "value": 6096}
    assert context["J12"] == {"cell": "J12", "header": "Qty", "value": 1250}
    assert context["K12"] == {"cell": "K12", "header": "UOM", "value": "M"}


def test_xlsx_is_converted_to_compact_json_with_formulas() -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "报价"
    sheet.append(["产品", "数量", "单价", "总价"])
    sheet.append(["钢管", 2, 10.5, "=B2*C2"])
    sheet["C2"].number_format = "0.00"
    sheet["D2"].number_format = "#,##0.00"
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
    assert parsed["sheets"][0]["rows"][1]["cells"]["C"] == {
        "value": 10.5,
        "number_format": "0.00",
    }
    assert parsed["sheets"][0]["rows"][1]["cells"]["D"] == {
        "formula": "=B2*C2",
        "number_format": "#,##0.00",
    }


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

    monkeypatch.setattr(external, "_execute_external_quote_score", fake_execute)
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
    task_input = json.loads(captured["task_input"])
    assert task_input["sheets"][0]["rows"][1]["cells"]["A"] == "钢管"


def test_direct_quote_score_uses_one_model_call_without_harness(monkeypatch) -> None:
    calls: dict = {"finalizer": 0, "recorded": None}

    async def fake_create(**_kwargs):
        return {"message_id": UUID("00000000-0000-0000-0000-000000000030")}

    def fake_finalizer(**kwargs):
        calls["finalizer"] += 1
        calls["finalizer_kwargs"] = kwargs
        return json.dumps(
            {
                "扣分项": [
                    {"评分维度": "计算准确度", "扣分原因": "公式错误", "扣分": -2}
                ]
            },
            ensure_ascii=False,
        )

    async def fake_record(**kwargs):
        calls["recorded"] = kwargs

    async def fail_if_harness_runs(_request):
        raise AssertionError("the dedicated quote scorer must not start Harness")

    monkeypatch.setattr(external, "create_external_chat_record", fake_create)
    monkeypatch.setattr(external, "finalize_quote_score_json", fake_finalizer)
    monkeypatch.setattr(external, "record_external_chat_answer", fake_record)
    monkeypatch.setattr(external, "ask_knowledge_base", fail_if_harness_runs)

    request = ExternalQueryRequest(
        question="请评分",
        user_id="u1",
        service_id="crm",
        session_id="unique-score-1",
        format_type="json",
    )
    response = asyncio.run(
        external._execute_external_quote_score(request, task_input='{"sheets":[]}')
    )

    assert calls["finalizer"] == 1
    assert calls["finalizer_kwargs"]["harness_answer"] == ""
    assert calls["finalizer_kwargs"]["user_instruction"] == "请评分"
    assert json.loads(response.answer)["总分"] == 98
    assert calls["recorded"]["answer"] == response.answer


def test_invalid_json_is_repaired_in_same_harness_session_before_storage(
    monkeypatch,
) -> None:
    calls: list = []
    recorded: dict = {}

    async def fake_create(**_kwargs):
        return {"message_id": UUID("00000000-0000-0000-0000-000000000020")}

    async def fake_ask(request):
        calls.append(request)
        if len(calls) == 1:
            return QueryResponse(question=request.question, answer="这是普通文本")
        return QueryResponse(
            question=request.question,
            answer='```json\n{"ok": true, "items": [1, 2]}\n```',
        )

    async def fake_record(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(external, "create_external_chat_record", fake_create)
    monkeypatch.setattr(external, "ask_knowledge_base", fake_ask)
    monkeypatch.setattr(external, "record_external_chat_answer", fake_record)

    request = ExternalQueryRequest(
        question="返回 JSON",
        user_id="u1",
        service_id="crm",
        session_id="s1",
        format_type="json",
    )
    response = asyncio.run(
        external._execute_external_query(
            request,
            additional_system_prompt="只返回合法 JSON",
            task_input='{"source":"test"}',
            answer_parser=parse_json_answer,
        )
    )

    assert len(calls) == 2
    assert calls[0].session_id == calls[1].session_id
    assert calls[0].user_id == calls[1].user_id
    assert calls[0].task_input == '{"source":"test"}'
    assert calls[1].task_input == ""
    assert "上一条回答未通过" in calls[1].question
    assert "agent answer is not valid JSON" in calls[1].question
    assert "这是普通文本" in calls[1].question
    assert "<mtsco-invalid-structured-answer>" in calls[1].question
    assert response.answer == '{"ok":true,"items":[1,2]}'
    assert recorded["answer"] == response.answer


def test_invalid_json_exhausts_bounded_repairs_without_recording_success(
    monkeypatch,
) -> None:
    calls: list = []
    recorded: list[dict] = []

    async def fake_create(**_kwargs):
        return {"message_id": UUID("00000000-0000-0000-0000-000000000021")}

    async def fake_ask(request):
        calls.append(request)
        return QueryResponse(question=request.question, answer="仍然不是 JSON")

    async def fake_record(**kwargs):
        recorded.append(kwargs)

    monkeypatch.setattr(external, "create_external_chat_record", fake_create)
    monkeypatch.setattr(external, "ask_knowledge_base", fake_ask)
    monkeypatch.setattr(external, "record_external_chat_answer", fake_record)

    request = ExternalQueryRequest(
        question="返回 JSON",
        user_id="u1",
        service_id="crm",
        session_id="s1",
        format_type="json",
    )
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            external._execute_external_query(
                request,
                additional_system_prompt="只返回合法 JSON",
                answer_parser=parse_json_answer,
            )
        )

    assert exc_info.value.status_code == 502
    assert "after 3 attempts" in str(exc_info.value.detail)
    assert len(calls) == 3
    assert recorded == []


def test_structured_output_repair_prompt_stays_within_query_limit() -> None:
    prompt = external._structured_output_repair_prompt(
        "agent answer is not valid JSON",
        invalid_answer="x" * 50_000,
    )

    assert len(prompt) <= 8_000
    assert "x" * 5_000 in prompt

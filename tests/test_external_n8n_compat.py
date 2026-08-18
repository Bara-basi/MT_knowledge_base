from __future__ import annotations

import json
from pathlib import Path

from app.schemas.query import N8nQueryRequest


WORKFLOW_PATH = Path("n8n-files") / "知识库问答-prod.json"


def test_external_payload_is_accepted_by_n8n_contract() -> None:
    payload = N8nQueryRequest(
        question="制度是什么？",
        user_id="external:v1:u:" + "a" * 64,
        session_id="external:v1:s:" + "b" * 64,
        conversation_id="external:v1:s:" + "b" * 64,
        service_id="crm",
        use_lark_document=False,
        format_type="json",
        additional_system_prompt="Return JSON",
        task_input='{"rows":[]}',
        source="external",
    ).model_dump()

    assert payload["source"] == "external"
    assert payload["service_id"] == "crm"
    assert payload["use_lark_document"] is False
    assert payload["format_type"] == "json"
    assert payload["additional_system_prompt"] == "Return JSON"
    assert payload["task_input"] == '{"rows":[]}'
    assert payload["current_topic"] == "无近期对话"
    assert payload["history_topics"] == []


def test_prod_workflow_passes_isolated_ids_through_topic_callbacks() -> None:
    workflow = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    nodes = {node["name"]: node for node in workflow["nodes"]}

    webhook = nodes["Webhook1"]["parameters"]
    assert webhook["httpMethod"] == "POST"
    assert webhook["responseMode"] == "lastNode"

    for node_name in ("新建主题", "继承/切换对话"):
        parameters = nodes[node_name]["parameters"]
        assert parameters["method"] == "POST"
        body_parameters = {
            item["name"]: item["value"]
            for item in parameters["bodyParameters"]["parameters"]
        }
        assert "Webhook1" in body_parameters["user_id"]
        assert ".body.user_id" in body_parameters["user_id"]
        assert "Webhook1" in body_parameters["session_id"]
        assert ".body.session_id" in body_parameters["session_id"]

    create_topic_parameters = {
        item["name"]: item["value"]
        for item in nodes["新建主题"]["parameters"]["bodyParameters"]["parameters"]
    }
    assert ".body.service_id" in create_topic_parameters["metadata"]
    assert ".body.metadata?.source" in create_topic_parameters["metadata"]

    final_code = nodes["Code in JavaScript"]["parameters"]["jsCode"]
    assert "answer," in final_code
    assert "topic_id:" in final_code

    final_prompt = nodes["final_qa_agent"]["parameters"]["text"]
    assert ".body.additional_system_prompt" in final_prompt
    assert ".body.task_input" in final_prompt

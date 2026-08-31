# 外部知识库与报价评分 API 接口文档

本文档面向需要调用企业内部知识库的其他项目组，说明通用知识库问答接口和报价评分接口的请求、响应及推荐用法。

## 1. 接口概览

| 接口 | 方法 | Content-Type | 用途 |
| --- | --- | --- | --- |
| `/prod/api/v1/external/query` | `POST` | `application/json` | 通用知识库问答，支持 Markdown 或 JSON 输出 |
| `/prod/api/v1/external/quote-score` | `POST` | `multipart/form-data` | 对文本或 Excel 报价材料进行评分，固定返回 JSON |

生产环境基础地址由部署方提供，本文使用以下占位地址：

```text
http://47.96.9.22:8000
```

完整地址示例：

```text
http://47.96.9.22:8000/prod/api/v1/external/query
```

当前接口仅供企业内部系统使用，暂未要求鉴权 Header。调用方应设置不短于 600 秒的请求超时时间，避免知识检索或模型生成耗时较长时由客户端提前断开。

## 2. 公共调用约定

### 2.1 数据隔离与多轮对话

- `service_id` 用来标识调用系统或项目组，例如 `quotation-system`。
- 不同 `service_id` 的用户及会话会被隔离。不同项目组不要共用同一个 `service_id`。
- `user_id` 是调用系统中的用户标识，不要求使用飞书 Open ID。
- `session_id` 是调用方生成并维护的会话标识。同一个用户需要继续多轮对话时，应持续传入相同的 `service_id`、`user_id` 和 `session_id`。
- 新对话应生成新的 `session_id`。建议使用业务单号、UUID，或“业务类型 + 唯一编号”，例如 `quote-training-20260818-001`。
- 后端会使用 `service_id` 对外部 `user_id` 和 `session_id` 做稳定的命名空间处理，原始标识不会写入聊天表。

### 2.2 通用成功与失败规则

- 成功响应的 HTTP 状态码为 `200`。
- 除报价评分响应的中文业务字段外，接口字段名区分大小写，均使用文档中的小写蛇形格式。
- 返回 `4xx` 或 `5xx` 时，不应使用响应内容作为正常问答或评分结果。
- FastAPI 业务错误通常采用以下结构：

```json
{
  "detail": "错误说明"
}
```

- 参数校验错误 `422` 的 `detail` 通常是校验错误数组。

## 3. 通用知识库问答

### 3.1 接口信息

```http
POST /prod/api/v1/external/query
Content-Type: application/json
```

该接口复用现有知识检索、主题路由、多轮对话和 n8n 问答链路。它适合普通知识库问答/问答小助手，以及希望模型直接返回 JSON 对象的内部系统调用。

### 3.2 请求参数

请求 Body 为 JSON 对象：

| 字段 | 类型 | 必填 | 默认值 | 约束 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `question` | string | 是 | 无 | 去除首尾空格后长度 `1–8000` | 本次问题或任务内容 |
| `user_id` | string | 是 | 无 | 去除首尾空格后长度 `1–512` | 调用系统内的用户唯一标识 |
| `service_id` | string | 是 | 无 | 去除首尾空格后长度 `1–128` | 调用系统或项目组标识，也是数据隔离边界 |
| `session_id` | string | 是 | 无 | 去除首尾空格后长度 `1–512` | 多轮对话会话标识 |
| `use_lark_document` | boolean | 否 | `false` | `true` 或 `false` | Markdown 知识来源是否优先使用映射后的飞书文档链接,妙搭应用和飞书企业自建应用推荐使用`true`,其余使用`false` |
| `format_type` | string | 否 | `markdown` | `markdown` 或 `json` | 指定回答格式 |

请求示例：

```json
{
  "question": "公司的报价审批流程是什么？",
  "user_id": "employee-1001",
  "service_id": "quotation-system",
  "session_id": "ticket-20260818-001",
  "use_lark_document": false,
  "format_type": "markdown"
}
```

### 3.3 响应参数

| 字段 | 类型 | 一定返回 | 说明 |
| --- | --- | --- | --- |
| `question` | string | 是 | 原始问题 |
| `answer` | string、object 或 array | 是 | `markdown` 时为字符串；`json` 时为解析后的 JSON 对象或数组 |
| `user_id` | string | 是 | 请求中的原始用户标识 |
| `service_id` | string | 是 | 请求中的服务标识 |
| `session_id` | string | 是 | 请求中的会话标识 |
| `topic_id` | string | 否 | 后端为本次对话识别出的主题 ID；无法确定时省略 |
| `answer_format` | string | 是 | `markdown` 或 `json` |
| `status` | string | 是 | 成功时固定为 `success` |

#### Markdown 响应

```json
{
  "question": "公司的报价审批流程是什么？",
  "answer": "报价需要先完成成本核算，再提交负责人审批。[1]\n\n---\n### 知识来源\n1. [报价制度.docx](https://.../prod/api/v1/documents/download?path=...)",
  "user_id": "employee-1001",
  "service_id": "quotation-system",
  "session_id": "ticket-20260818-001",
  "topic_id": "550e8400-e29b-41d4-a716-446655440000",
  "answer_format": "markdown",
  "status": "success"
}
```

`use_lark_document` 的效果：

- `false`：知识来源默认指向后端文档下载接口，适合非飞书系统。
- `true`：如果系统中存在文件到飞书文档的映射，优先返回飞书文档链接；没有映射时仍返回后端下载链接。

### 3.4 多轮对话示例

第一次调用：

```json
{
  "question": "请说明公司的报价审批流程。",
  "user_id": "employee-1001",
  "service_id": "crm",
  "session_id": "approval-discussion-001"
}
```

继续追问时保持三个标识不变：

```json
{
  "question": "刚才流程中的审批负责人是谁？",
  "user_id": "employee-1001",
  "service_id": "crm",
  "session_id": "approval-discussion-001"
}
```

## 4. 报价评分工具

### 4.1 接口信息

```http
POST /prod/api/v1/external/quote-score
Content-Type: multipart/form-data
```

该接口使用报价专用的单次模型审核链路，根据内置评分规则和通用报价计算知识分析输入；不会先启动通用知识库 Harness。接口固定返回经过后端校验和重算的评分 JSON。

可以采用两种输入方式：

1. 纯文本评分：不上传 `file`，直接把待评分内容写入 `question`。
2. Excel 评分：`question` 写评分要求或背景，通过 `file` 上传 `.xlsx` 或 `.xls`。

### 4.2 请求参数

请求使用表单字段，不是 JSON Body：

| 字段 | 类型 | 必填 | 默认值 | 约束 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `question` | string | 是 | 无 | 去除首尾空格后长度 `1–8000` | 待评分文本，或对上传文件的评分要求/背景 |
| `user_id` | string | 是 | 无 | 去除首尾空格后长度 `1–512` | 调用系统内的用户标识 |
| `service_id` | string | 是 | 无 | 去除首尾空格后长度 `1–128` | 调用系统或项目组标识 |
| `session_id` | string | 是 | 无 | 去除首尾空格后长度 `1–512` | 本次评分会话标识 |
| `use_lark_document` | boolean | 否 | `false` | `true` 或 `false` | 为兼容旧调用方保留；当前报价专用链路不执行知识检索 |
| `file` | file | 否 | 无 | 仅 `.xlsx` 或 `.xls` | 待评分报价表格 |

评分接口不需要传 `format_type`，其输出格式固定为 `json`。服务端日志会分别记录文件解析、模型和请求总耗时，且不会记录报价文件正文。

#### Excel 文件限制

| 限制项 | 限制值 |
| --- | --- |
| 文件格式 | `.xlsx`、`.xls` |
| 文件大小 | 不超过 10 MB |
| 工作簿总行数 | 不超过 2,000 行 |
| 单个工作表列数 | 不超过 100 列 |
| 转换后的 JSON 长度 | 不超过 120,000 字符 |

处理方式：

- 所有工作表都会转换为结构化 JSON，包括隐藏工作表。
- `.xlsx` 中的公式会保留公式文本；如果文件中存在缓存计算结果，也会一并传入。
- `.xls` 读取单元格可用值。
- 上传文件原文和转换后的完整 JSON 不写入 `chat_messages_external`，仅作为本次 n8n 评分任务的输入。
- `question` 会正常写入外部聊天记录，因此不要在 `question` 中放置不需要持久化的敏感原始表格内容。


```

Python 示例：

```python
from pathlib import Path

import requests

url = "http://47.96.9.22:8000/prod/api/v1/external/quote-score"
workbook_path = Path("DAY 1 报价练习题.xlsx")
form_data = {
    "question": "请对上传的报价文件进行评分，并逐条说明扣分原因。",
    "user_id": "employee-1001",
    "service_id": "quote-training",
    "session_id": "excel-score-001",
    "use_lark_document": "false",
}

with workbook_path.open("rb") as workbook:
    files = {
        "file": (
            workbook_path.name,
            workbook,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    }
    response = requests.post(
        url,
        data=form_data,
        files=files,
        timeout=300,
    )

response.raise_for_status()
score = response.json()
print(f'{score["总分"]}/{score["满分"]}')
for item in score["扣分项"]:
    print(item["评分维度"], item["扣分原因"], item["扣分"])
```

不要手动设置 multipart 请求的 `Content-Type` 边界。使用 curl 的 `-F`、Python `requests` 的 `files`，或前端 `FormData` 时，由客户端自动生成正确的 `Content-Type`。

### 4.3 响应参数

| 字段 | 类型 | 一定返回 | 说明 |
| --- | --- | --- | --- |
| `总分` | integer | 是 | `0–100`，由后端根据有效扣分项重算 |
| `满分` | integer | 是 | 固定为 `100` |
| `评分维度` | object | 是 | 六个维度的分数对象 |
| `扣分项` | array | 是 | 每个可确认问题对应一个扣分对象；无扣分时为空数组 |
| `user_id` | string | 是 | 请求中的原始用户标识 |
| `service_id` | string | 是 | 请求中的服务标识 |
| `session_id` | string | 是 | 请求中的会话标识 |
| `file_name` | string | 否 | 上传文件名；未上传文件时省略 |
| `topic_id` | string | 否 | 本次问答主题 ID；无法确定时省略 |
| `format_type` | string | 是 | 固定为 `json` |
| `status` | string | 是 | 成功时固定为 `success` |

`评分维度` 固定包含：

| 字段 | 类型 | 范围 | 当前规则 |
| --- | --- | --- | --- |
| `询价完整度` | integer | `0–100` | 根据该维度扣分项从 100 分递减 |
| `询价供应商准确度` | integer | `0–100` | 根据该维度扣分项从 100 分递减 |
| `询价命名规范度` | integer | `0–100` | 根据该维度扣分项从 100 分递减 |
| `计算准确度` | integer | `0–100` | 根据该维度扣分项从 100 分递减 |
| `报价完整度` | integer | `0–100` | 根据该维度扣分项从 100 分递减 |
| `报价及时性` | integer | `0–100` | 当前不参与判断，始终为 `100` |

每个 `扣分项` 的结构：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `评分维度` | string | 必须是六个评分维度之一；当前不会返回报价及时性扣分 |
| `扣分原因` | string | 具体、可核对的违规说明 |
| `扣分` | integer | 负数形式的本项扣分，例如 `-1`、`-2`、`-5` |

完整响应示例：

```json
{
  "总分": 90,
  "满分": 100,
  "评分维度": {
    "询价完整度": 99,
    "询价供应商准确度": 100,
    "询价命名规范度": 100,
    "计算准确度": 96,
    "报价完整度": 95,
    "报价及时性": 100
  },
  "扣分项": [
    {
      "评分维度": "询价完整度",
      "扣分原因": "规格要素与询价内容不匹配",
      "扣分": -1
    },
    {
      "评分维度": "计算准确度",
      "扣分原因": "总价与数量乘以单价的计算结果不一致",
      "扣分": -2
    },
    {
      "评分维度": "计算准确度",
      "扣分原因": "汇率使用错误",
      "扣分": -2
    },
    {
      "评分维度": "报价完整度",
      "扣分原因": "未按照报价模板提供完整字段",
      "扣分": -5
    }
  ],
  "user_id": "employee-1001",
  "service_id": "quotation-system",
  "session_id": "excel-score-001",
  "file_name": "DAY 1 报价练习题.xlsx",
  "topic_id": "550e8400-e29b-41d4-a716-446655440000",
  "format_type": "json",
  "status": "success"
}
```

总分计算方式：

```text
总分 = max(0, 100 - 所有有效扣分项绝对值之和)
维度分 = max(0, 100 - 该维度所有有效扣分项绝对值之和)
```

接口会忽略模型生成的总分并在后端重新计算，同时去除重复的相同扣分项。报价及时性扣分会被移除，报价及时性分数固定为 100。

### 4.4 当前扣分值约束

| 评分维度 | 允许的单项扣分值 |
| --- | --- |
| 询价完整度 | `-5`、`-1` |
| 询价供应商准确度 | `-5` |
| 询价命名规范度 | `-5` |
| 计算准确度 | `-2` |
| 报价完整度 | `-5`、`-3`、`-1` |
| 报价及时性 | 不扣分 |

如果模型输出不符合上述约束的扣分项，后端不会返回未经校验的评分，而是返回 `502`。

## 5. 错误码

| HTTP 状态码 | 适用接口 | 含义 | 调用方处理建议 |
| --- | --- | --- | --- |
| `400` | 报价评分 | 文件格式不支持、文件损坏、超过大小/行列/转换长度限制，或 `.xls` 解析依赖不可用 | 修正或精简文件后重试，不要原样自动重试 |
| `422` | 两个接口 | 缺少必填参数、字符串为空或过长、类型错误、`format_type` 非法 | 根据响应 `detail` 修正请求 |
| `502` | 两个接口 | Harness/报价评分模型不可用、响应缺少答案，或模型没有生成符合要求的 JSON/评分结构 | 可有限次数重试；持续发生时联系知识库维护方并提供时间、`service_id`、`session_id` |
| `503` | 两个接口 | 外部聊天记录表不可用，或答案无法写入隔离表 | 稍后重试，并检查 PostgreSQL 状态 |
| `504` | 通用问答 | Harness 问答超时 | 可稍后重试；同时确认客户端自身超时设置 |
| `500` | 两个接口 | 未预期的服务端错误 | 联系维护方排查服务日志 |

错误示例：

```json
{
  "detail": "quote file must be .xlsx or .xls"
}
```

参数校验错误示例：

```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "question"],
      "msg": "String should have at least 1 character",
      "input": ""
    }
  ]
}
```

## 6. 接入建议

1. 每个项目组先申请或约定唯一的 `service_id`，并将其配置在服务端，不要由最终用户随意填写。
2. 使用业务用户 ID 作为 `user_id`；匿名业务也应生成稳定的内部用户标识。
3. 由调用方持久化 `session_id`。继续对话时复用，开始新话题时更换。
4. 普通页面展示优先使用 `format_type=markdown`；程序消费答案时使用 `json` 并自行校验业务 Schema。
5. 报价评分始终调用 `/external/quote-score`，不要使用通用 JSON 问答模拟评分。
6. 调用日志建议记录请求时间、`service_id`、`user_id`、`session_id`、HTTP 状态码和耗时，但不要记录完整报价文件或敏感问答正文。
7. 对 `502`、`503`、`504` 最多进行少量指数退避重试；对 `400`、`422` 不进行自动重试。

# API 手册

我以当前 FastAPI 路由为准列出全部接口的最小调用示例。以下命令使用 Bash 语法，默认生产前缀为 `/prod`；Windows 可在 Git Bash/WSL 中执行，或把续行符 `\` 换成 PowerShell 反引号。

```bash
BASE_URL='http://127.0.0.1:8000'
API_BASE="$BASE_URL/prod/api/v1"
```

除飞书验签与内部附件令牌外，当前代码没有统一 API 鉴权。我只在可信网络或完成网关认证后开放这些接口。

## 服务状态与开发代理

```bash
curl "$BASE_URL/"
curl "$BASE_URL/health"
curl "$BASE_URL/health/ready"
curl "$BASE_URL/metrics"
curl "$API_BASE/health"
curl "$BASE_URL/openapi.json"
curl "$BASE_URL/docs"
curl "$BASE_URL/redoc"
```

`/health` 表示进程存活；`/health/ready` 检查 PostgreSQL，并返回 Harness 和持久队列开关；`/metrics` 返回 Prometheus 文本。配置 `DEV_PROXY_TARGET` 后，`/dev` 与 `/dev/{path}` 会保留原方法、请求体和查询参数进行代理：

```bash
curl "$BASE_URL/dev/health"
curl -X POST "$BASE_URL/dev/example" -H 'Content-Type: application/json' -d '{"value":"demo"}'
```

## 问答

### `POST /query`

```bash
curl -X POST "$API_BASE/query" \
  -H 'Content-Type: application/json' \
  -d '{"question":"公司的报价审批流程是什么？","user_id":"demo-user","session_id":"demo-session","metadata":{"department":"demo"}}'
```

同一 `user_id + session_id` 会延续上下文。`additional_system_prompt` 和 `task_input` 只供受信任的服务端任务使用。

### `POST /external/query`

```bash
curl -X POST "$API_BASE/external/query" \
  -H 'Content-Type: application/json' \
  -d '{"question":"公司的报价审批流程是什么？","user_id":"demo-user","service_id":"demo-service","session_id":"demo-session","use_lark_document":false,"format_type":"markdown"}'
```

外部问答使用 `service_id + user_id + session_id` 生成隔离标识，并写入独立聊天表。`format_type` 可选 `markdown` 或 `json`；`use_lark_document=true` 时优先返回已映射的飞书文档链接。

### `POST /external/quote-score`

```bash
# 直接评分文本
curl -X POST "$API_BASE/external/quote-score" \
  -F 'question=请对以下报价内容进行评分：示例内容' \
  -F 'user_id=demo-user' -F 'service_id=demo-service' -F 'session_id=demo-quote'

# 上传 Excel
curl -X POST "$API_BASE/external/quote-score" \
  -F 'question=请对上传的报价文件进行评分' \
  -F 'user_id=demo-user' -F 'service_id=demo-service' -F 'session_id=demo-quote' \
  -F 'file=@./quote-example.xlsx'
```

文件仅支持 `.xlsx`、`.xls`，最大 10 MB、2,000 行、100 列；响应为后端重算后的严格 JSON。

## 混合检索与文档直连

### `POST /retrieval/flow`

```bash
curl -X POST "$API_BASE/retrieval/flow" \
  -H 'Content-Type: application/json' \
  -d '{"query":"报价审批流程","limit":8,"rerank":true,"debug":false}'
```

### `POST /retrieval/filtered`

```bash
curl -X POST "$API_BASE/retrieval/filtered" \
  -H 'Content-Type: application/json' \
  -d '{"query":"适用要求","file_path":"示例文档路径","path_prefix":"第1章","limit":5,"rerank":true}'
```

### `POST /retrieval/documents/search`

```bash
curl -X POST "$API_BASE/retrieval/documents/search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"报价制度","limit":10}'
```

### `POST /retrieval/document-direct`

```bash
curl -X POST "$API_BASE/retrieval/document-direct" \
  -H 'Content-Type: application/json' \
  -d '{"query":"报价制度","limit":2,"max_chars_per_document":120000}'
```

`flow` 执行混合召回与可选重排；`filtered` 把范围限制在指定文档；`documents/search` 只找文档；`document-direct` 返回命中解析文档的整块上下文。

## 图谱

### `POST /graph/query`

```bash
# 有界结构化遍历
curl -X POST "$API_BASE/graph/query" \
  -H 'Content-Type: application/json' \
  -d '{"start_labels":["Product"],"start_properties":{"name":"示例产品"},"relationship_types":["APPLIES_STANDARD"],"direction":"outgoing","max_depth":1,"return_type":"nodes","limit":20}'

# 只读 Cypher
curl -X POST "$API_BASE/graph/query" \
  -H 'Content-Type: application/json' \
  -d '{"cypher":"MATCH (n) WHERE n.name = $name RETURN n LIMIT 10","parameters":{"name":"示例标准"},"return_type":"records","limit":10}'
```

写入语句和过程调用会被拒绝，结构化遍历深度上限为 3。

### `POST /graph/product-standards`

```bash
curl -X POST "$API_BASE/graph/product-standards" \
  -H 'Content-Type: application/json' -d '{"keyword":"示例产品","limit":20}'
```

### `POST /graph/standard-context`

```bash
curl -X POST "$API_BASE/graph/standard-context" \
  -H 'Content-Type: application/json' -d '{"keyword":"示例标准","limit":20}'
```

## 文档与营销资料

### `GET /documents/download`

```bash
curl -G "$API_BASE/documents/download" \
  --data-urlencode 'path=minio://knowledge-raw-docs/examples/example.pdf' -o example.pdf
```

### `POST /documents/marketing-assets/search`

```bash
curl -X POST "$API_BASE/documents/marketing-assets/search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"产品样册","limit":10,"source":"api"}'
```

`source=harness` 返回紧凑模型文本，其他值返回结构化匹配列表。

### `POST /documents/agent-vision-payload`

```bash
curl -X POST "$API_BASE/documents/agent-vision-payload" \
  -H 'Content-Type: application/json' \
  -d '{"path":"minio://knowledge-raw-docs/examples/example.pdf","max_pages":12}'
```

### `POST /documents/agent-context-assets`

```bash
curl -X POST "$API_BASE/documents/agent-context-assets" \
  -H 'Content-Type: application/json' \
  -d '{"document_paths":["minio://knowledge-raw-docs/examples/example.docx"],"image_paths":["minio://knowledge-raw-docs/examples/example.png"]}'
```

## MinIO 管理

这些接口属于内部管理面。

### `GET /documents/minio/categories`

```bash
curl "$API_BASE/documents/minio/categories"
```

### `POST /documents/minio/init`

```bash
curl -X POST "$API_BASE/documents/minio/init?bucket=knowledge-raw-docs"
```

### `POST /documents/minio/upload`

```bash
curl -X POST "$API_BASE/documents/minio/upload" \
  -F 'file=@./example.pdf' -F 'object_name=examples/example.pdf'
```

也可以传 `bucket`，或先调用 `minio/categories` 后使用返回的有效 `category`。

## 飞书文档同步

这些接口会读取飞书内容并更新解析/索引，属于内部管理面。

### `POST /documents/sync/lark/scan`

```bash
curl -X POST "$API_BASE/documents/sync/lark/scan?dry_run=true&image_workers=3"
```

### `POST /documents/sync/lark/update`

```bash
curl -X POST "${API_BASE}/documents/sync/lark/update?document_name=%E7%A4%BA%E4%BE%8B%E6%96%87%E6%A1%A3&force=true&dry_run=false"
```

### `POST /documents/sync/lark/ingest`

```bash
curl -X POST "${API_BASE}/documents/sync/lark/ingest?document_link=https%3A%2F%2Fexample.feishu.cn%2Fwiki%2Fexample&bucket=knowledge-raw-docs&image_workers=3"
```

## Harness 内部附件

这些接口只供内部进程使用。设置 `HARNESS_ATTACHMENT_API_TOKEN` 后必须携带 `X-KB-Internal-Token`；未设置时只允许本机调用。支持 DOCX、XLSX、PPTX、PDF、旧版 DOC/XLS/PPT 和常见图片。XLS 使用 `xlrd + openpyxl` 在纯 Python 中升级；DOC/PPT 使用 LibreOffice headless，服务器可通过 `LIBREOFFICE_BIN` 指定可执行文件，不调用 Microsoft Office。

```bash
INTERNAL_TOKEN='<INTERNAL_TOKEN>'
```

### `POST /documents/harness-attachments/upload`

```bash
curl -X POST "$API_BASE/documents/harness-attachments/upload" \
  -H "X-KB-Internal-Token: $INTERNAL_TOKEN" \
  -F 'user_id=demo-user' -F 'internal_session_id=demo-session' -F 'file=@./example.pdf'
```

### `GET /documents/harness-attachments`

```bash
curl -G "$API_BASE/documents/harness-attachments" \
  -H "X-KB-Internal-Token: $INTERNAL_TOKEN" \
  --data-urlencode 'user_id=demo-user' --data-urlencode 'internal_session_id=demo-session'
```

### `POST /documents/harness-attachments/parse`

```bash
curl -X POST "$API_BASE/documents/harness-attachments/parse" \
  -H "X-KB-Internal-Token: $INTERNAL_TOKEN" -H 'Content-Type: application/json' \
  -d '{"user_id":"demo-user","internal_session_id":"demo-session","attachment_id":"<ATTACHMENT_ID>"}'
```

### `POST /documents/harness-attachments/read`

```bash
curl -X POST "$API_BASE/documents/harness-attachments/read" \
  -H "X-KB-Internal-Token: $INTERNAL_TOKEN" -H 'Content-Type: application/json' \
  -d '{"user_id":"demo-user","internal_session_id":"demo-session","attachment_id":"<ATTACHMENT_ID>","query":"关键结论","limit":3}'
```

## 飞书事件回调

标准生产回调是 `POST /prod/feishu/events`；同一处理器也挂载在 `/prod/api/v1/feishu/events`。

```bash
curl -X POST "$BASE_URL/prod/feishu/events" \
  -H 'Content-Type: application/json' \
  -d '{"token":"<FEISHU_VERIFICATION_TOKEN>","challenge":"demo-challenge","type":"url_verification"}'
```

正常返回 `{"challenge":"demo-challenge"}`。真实消息、卡片按钮和文档变更事件由飞书推送。当前实现不接受加密事件体；普通消息验签、去重后由 FastAPI `BackgroundTasks` 直接处理，不进入 PostgreSQL 回答队列。

## 常见状态码

- `400`：参数、对象引用、附件类型或图谱查询不合法；
- `401/403`：内部附件令牌错误或调用来源不允许；
- `404`：文档、模型文件或附件不存在；
- `429`：共享限流触发；
- `502`：Harness、模型或上游返回无效结果；
- `503`：Harness 未启用、依赖不可用或并发容量不足；
- `504`：上游超时。

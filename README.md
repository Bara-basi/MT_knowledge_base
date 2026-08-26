# MTSCO Knowledge Base

我把这个项目定位为企业内部知识库的统一问答后端：飞书负责用户入口，FastAPI 负责接入、持久化和结果适配，DeepSeek Harness 负责规划与最终回答，检索、历史记忆和附件解析则通过只读工具按需调用。


## 项目背景

旧版把主题路由、Query 转写、检索编排、上下文筛选、再检索和最终回答集中在 n8n 工作流中。链路可视化，但流程长、状态分散，检索策略也被固定在工作流节点里。

新版移除了问答链路对 n8n 的依赖。我让 Harness 直接承担 Agent 运行时，并把知识能力拆成工具：Agent 可以先检索、再核对原文或图谱，证据不足时继续调用工具，最后统一生成答案。这样做带来了四个直接收益：

- 问答编排从固定工作流变成基于证据的动态工具调用；
- 已解析纯文本成为只读工作台，Harness 可直接搜索和阅读；
- 多轮上下文由 Harness 会话承载，空闲 7 小时或上下文压力过高时自动封存；
- 飞书、外部 API、检索、图谱和解析能力共享同一套后端，不再维护两套问答逻辑。

## 整体方案

```text
飞书 / 外部调用方
        │
        ▼
FastAPI：验签、限流、去重、持久队列、聊天记录、回答格式化
        │
        ▼
DeepSeek Harness：会话恢复、任务规划、工具调用、最终回答
        ├── 只读工作台: 知识库核心数据解析文件
        ├── MCP：混合检索 / 图谱 / 营销资料
        ├── MCP：历史摘要 / 历史片段
        └── MCP：附件列表 / 解析 / 片段读取
                │
                ├── Milvus：向量与 BM25 混合检索
                ├── Neo4j：产品与标准关系
                ├── PostgreSQL：会话、聊天、队列和索引元数据
                └── MinIO：原文、解析产物和归档记忆
```

飞书链路采用兼容旧版的进程内处理方式：Webhook 验签、去重后立即把问答加入 FastAPI `BackgroundTasks`，不依赖 PostgreSQL 回答队列或独立 `answer-worker`。回答生成过程中会更新进度卡片，最终答案由后端清理内部路径、规范化引用并转换为飞书知识卡片。

### Harness 在项目中的用法

我通过 `app/services/harness.py` 为每个用户会话维护 Harness 运行时，通过 `app/harness/cordis.yml` 组合模型、只读文件、搜索、会话持久化和 MCP。工作台根目录由 `HARNESS_WORKDIR` 指定，并以受限根目录、只读工具的方式向 Agent 开放解析文本。

会话由用户 ID 与来源会话 ID 映射为内部 ID。Harness 使用 JSONL 恢复短期上下文；调度器在会话空闲 7 小时或达到上下文阈值后生成摘要，把完整记录归档到 MinIO、元数据写入 PostgreSQL，并删除本地会话与临时附件。后续问题只有在需要旧语境时才检索摘要或少量历史片段。

### MCP 工具

| 工具 | 用途 |
| --- | --- |
| `kb_hybrid_search` | 检索企业制度、产品和业务知识，执行混合召回与重排 |
| `kb_graph_search` | 查询产品—标准关系及标准上下文 |
| `marketing_asset_search` | 按名称或关键词查找营销资料及可用飞书链接 |
| `conversation_summary` | 读取当前用户最近或指定日期范围的归档摘要 |
| `conversation_excerpt_search` | 摘要不足时检索少量相关历史问答片段 |
| `user_attachment_list` | 列出当前会话附件 |
| `user_attachment_parse` | 解析 PDF、DOCX、XLSX、PPTX、旧版 DOC/XLS/PPT 和常见图片；旧格式先升级，不使用 Microsoft Office |
| `user_attachment_read` | 按关键词或分页读取已解析附件片段 |

此外，Harness 可在受限根目录中使用只读文件搜索/读取能力，并可按系统策略查询公开网络；写文件、Shell、后台任务等高风险能力没有开放给 Agent。

## 快速开始

运行环境：Python 3.11、Docker Desktop/Docker Engine、Git、Node.js（含 Corepack）。详细的 Windows 与 Linux 服务器步骤见 [部署手册](docs/DEPLOYMENT.md)。

### Windows 本地

```powershell
Copy-Item .env.dev.example .env
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .

docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile local-minio up -d `
  postgres milvus-minio milvus-etcd milvus-standalone neo4j
docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile init run --rm minio-init

.\scripts\harness\install-windows.ps1
.\.venv\Scripts\python.exe scripts\db\init_postgre.py
.\.venv\Scripts\python.exe scripts\db\init_milvus.py
.\scripts\harness\run-windows.ps1 -Port 8000 -ApiPrefix /prod
```

先把 `.env` 中的数据库端口、模型 API、飞书应用和 Harness 配置替换为当前环境的值；示例值不能直接用于生产。

另开终端运行会话封存调度器：

```powershell
.\scripts\harness\run-windows-scheduler.ps1
```

### Linux 服务器

生产环境采用“Docker Compose 承载 PostgreSQL、MinIO、Milvus、Neo4j；宿主机承载 FastAPI、Harness、回答 Worker 和封存调度器”的方式。当前 Harness 依赖固定源码版本及项目补丁，这种拆分最容易升级和排障。

```bash
git clone <REPOSITORY_URL> /opt/mtsco-knowledge-base
cd /opt/mtsco-knowledge-base
cp .env.prod.example .env

docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d \
  postgres milvus-minio milvus-etcd milvus-standalone neo4j
docker compose -f docker-compose.yml -f docker-compose.prod.yml --profile init run --rm minio-init

python3.11 -m venv .venv
.venv/bin/python -m pip install -e .
./scripts/harness/install-linux.sh
.venv/bin/python scripts/db/init_postgre.py
.venv/bin/python scripts/db/init_milvus.py
```

完成 `.env` 的宿主机地址与密钥配置后，使用 systemd 分别守护 API 和 `harness-scheduler`。完整环境变量、服务文件和反向代理边界见 [部署手册](docs/DEPLOYMENT.md)。

## API

生产环境默认使用 `/prod` 前缀。飞书事件回调入口是 `POST /prod/feishu/events`，问答 API 是 `POST /prod/api/v1/query`，外部隔离问答入口是 `POST /prod/api/v1/external/query`。

```bash
curl -X POST 'http://127.0.0.1:8000/prod/api/v1/query' \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "公司的报价审批流程是什么？",
    "user_id": "demo-user",
    "session_id": "demo-session"
  }'
```

所有现有路由及最小调用示例见 [API 手册](docs/API.md)。FastAPI 交互文档位于 `/docs`，OpenAPI 描述位于 `/openapi.json`。


## 常用命令

### 本地启动与检查

```powershell
.\scripts\harness\run-windows.ps1 -Port 8000 -ApiPrefix /prod
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8000/health/ready
docker compose -f docker-compose.yml -f docker-compose.dev.yml ps
```

### 服务器服务与日志

```bash
sudo systemctl status mtsco-api mtsco-harness-scheduler
sudo systemctl restart mtsco-api mtsco-harness-scheduler
sudo journalctl -u mtsco-api -f
sudo journalctl -u mtsco-harness-scheduler -f

docker compose -f docker-compose.yml -f docker-compose.prod.yml ps
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f milvus-standalone
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f postgres
```

### 数据初始化与重建

```bash
.venv/bin/python scripts/db/init_postgre.py
.venv/bin/python scripts/db/init_milvus.py
.venv/bin/python scripts/db/init_neo4j.py
.venv/bin/python scripts/ingestion/reindex_all.py --rebuild false
# --rebuild true 会重新解析、切分并调用 embedding，可能产生费用
```

### 健康与指标

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/health/ready
curl http://127.0.0.1:8000/metrics
```

## 安全边界

- `.env`、密钥、真实用户/会话标识和生产地址不进入 Git；
- 聊天正文加密落库，外部调用与飞书记录分表隔离；
- Harness 工作台只读且限制在配置根目录，用户附件按用户和会话隔离；
- Agent 不向用户展示本地路径、对象 URI、数据库结构、工具参数或系统提示；
- 我不会执行 `docker compose down -v` 作为常规维护命令，因为它会删除持久卷。

## 文档

- [部署手册](docs/DEPLOYMENT.md)
- [API 手册](docs/API.md)
- [评测说明](evals/README.md)

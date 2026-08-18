# MTSCO Knowledge Base

MTSCO 企业内部知识库后端项目，当前主要通过 Docker Compose 部署生产环境，包含 FastAPI、Milvus、PostgreSQL、MinIO、n8n 等服务。本文档记录本次服务器部署过程中沉淀下来的关键经验和从零部署步骤。

## 服务器配置教程

当前生产配置默认使用外部 embedding / rerank / LLM API，不在服务器本地跑大模型，因此不强制需要 GPU。

最小配置要求：

| 场景 | CPU | 内存 | 磁盘 | GPU |
| --- | --- | --- | --- | --- |
| 小范围试用 | 2 核 | 8GB | 40GB SSD | 不需要 |
当前使用 `2C / 8GB / 40GB SSD` 的 Linux 服务器。
更推荐使用 `4C / 16GB / 200GB SSD` 的 Linux 服务器。

## 部署方式

代码推荐直接从 GitLab 拉取，而不是本地整体打包上传：
地址: https://gitlab.com/wangyi3020231348/MT_knowledge_base.git

填入access_token: `glpat-70bhf7v1BmlcmVF-kt24O2M6MQpvOjEKdTpubXgzYQ8.01.171b49j3q`

> `data/`：数据目录下的文件由于太多太大，不会同步，需要通过 rsync或 scp 手动同步


## 服务器基础环境

以下示例以 Ubuntu 22.04/24.04 为准。

```bash
apt update
apt install -y ca-certificates curl gnupg git rsync unzip openssl
```

安装 Docker 和 Docker Compose：

```bash
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

. /etc/os-release
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $VERSION_CODENAME stable" > /etc/apt/sources.list.d/docker.list

apt update
apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

验证：

```bash
docker --version
docker compose version
```

## 拉取项目代码

建议部署到 `/opt/mtsco-knowledge-base`：

```bash
cd /opt
git clone https://gitlab.com/wangyi3020231348/MT_knowledge_base.git
cd /opt/mtsco-knowledge-base
```


## 创建生产配置

复制生产环境示例配置：

```bash
cd /opt/mtsco-knowledge-base
cp .env.prod.example .env
```

编辑 `.env`：

```bash
nano .env
```

Linux 服务器上需要重点确认这些配置：

```env
COMPOSE_FILE=docker-compose.yml:docker-compose.prod.yml
GENERIC_TIMEZONE=Asia/Shanghai

API_HOST_PORT=8000
API_WORKERS=4
API_ROUTE_PREFIX=/prod
FEISHU_ROUTE_PREFIX=/prod
PUBLIC_BASE_URL=https://your-domain.example/prod

POSTGRES_DB=mtsco_knowledge_base
POSTGRES_USER=mtsco
POSTGRES_PASSWORD=change-me
POSTGRES_CHAT_TABLE=chat_messages
POSTGRES_EXTERNAL_CHAT_TABLE=chat_messages_external
CHAT_MESSAGE_ENCRYPTION_KEY=change-me-to-a-long-random-string

MINIO_ACCESS_KEY_ID=change-me
MINIO_SECRET_ACCESS_KEY=change-me
MINIO_INTERNAL_ENDPOINT=http://milvus-minio:9000
MINIO_PUBLIC_ENDPOINT=https://your-domain.example/minio
APP_MINIO_BUCKET=knowledge-raw-docs
MINIO_SECURE=false

USE_LOCAL_EMBEDDING_MODEL=false
USE_LOCAL_RERANK_MODEL=false
SKIP_RETRIEVAL_WARMUP=true
MODEL_CACHE_DIR=/opt/mtsco-models

SILICONFLOW_API_KEY=your-api-key
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1

N8N_HOST=your-domain.example
N8N_HOST_PORT=5678
N8N_WEBHOOK_URL=https://your-domain.example/
N8N_QUERY_WEBHOOK_URL=http://n8n:5678/webhook/fastapi-prod
N8N_API_BASE_URL=http://n8n:5678
N8N_ENCRYPTION_KEY=change-me-to-a-long-random-string

NEO4J_PASSWORD=change-me
```

## 外部问答 API

面向调用方的完整请求、响应、错误码和接入示例见 [外部知识库与报价评分 API 接口文档](docs/EXTERNAL_API.md)。

生产入口为 `POST /prod/api/v1/external/query`。它复用现有 n8n 问答与 `conversation_topics` 主题路由，但消息只写入 `chat_messages_external`，不会进入飞书使用的 `chat_messages`。

请求示例：

```bash
curl -X POST 'https://your-domain.example/prod/api/v1/external/query' \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "公司的报价审批流程是什么？",
    "user_id": "employee-1001",
    "service_id": "crm",
    "session_id": "ticket-20260817-001",
    "use_lark_document": false,
    "format_type": "markdown"
  }'
```

响应中的 `answer` 是 Markdown 原文字符串。知识来源会以 `---` 分隔并列在回答底部，不包含飞书卡片的折叠面板、停止按钮或反馈按钮。`use_lark_document=true` 时优先返回已映射的飞书文档地址；为 `false` 时返回后端文档下载地址。

`format_type` 可选 `markdown` 或 `json`，默认 `markdown`。使用 `json` 时，后端会要求 n8n 最终 Agent 只输出合法 JSON，并把解析后的 JSON 对象放入响应的 `answer` 字段；若模型返回的内容无法解析，接口返回 `502`，不会把伪 JSON 当成功结果返回。

```json
{
  "question": "公司的报价审批流程是什么？",
  "answer": "回答正文。[1]\n\n---\n### 知识来源\n1. [报价制度.docx](https://your-domain.example/prod/api/v1/documents/download?path=...)",
  "user_id": "employee-1001",
  "service_id": "crm",
  "session_id": "ticket-20260817-001",
  "topic_id": "00000000-0000-0000-0000-000000000001",
  "answer_format": "markdown",
  "status": "success"
}
```

初始化或升级 PostgreSQL 表结构：

```bash
python scripts/db/init_postgre.py
```

外部 `user_id` 和 `session_id` 不以原文落库。后端会将它们与 `service_id` 一起生成稳定的 SHA-256 命名空间 ID，使不同项目组即使提交相同 ID 也不会共享多轮上下文。调用方应在同一 `service_id` 下持续复用同一 `session_id` 来保持多轮对话。

### 报价评分工具

入口为 `POST /prod/api/v1/external/quote-score`，请求类型是 `multipart/form-data`。`question` 可以直接包含待评分的文本；当报价内容是 Excel 时，通过可选 `file` 字段上传 `.xlsx` 或 `.xls`。文件最大 10 MB，最多解析 2,000 行、100 列，解析结果不会写入聊天表，只作为本次评分的结构化任务输入。

```bash
curl -X POST 'https://your-domain.example/prod/api/v1/external/quote-score' \
  -F 'question=请对上传的报价文件进行评分' \
  -F 'user_id=employee-1001' \
  -F 'service_id=crm' \
  -F 'session_id=quote-training-001' \
  -F 'use_lark_document=false' \
  -F 'file=@./DAY 1 报价练习题.xlsx'
```

评分接口始终返回严格 JSON。所有维度和总分均从 100 分开始，后端根据逐条扣分项重算，`报价及时性` 当前固定为 100 分且不产生扣分项：

```json
{
  "总分": 93,
  "满分": 100,
  "评分维度": {
    "询价完整度": 100,
    "询价供应商准确度": 100,
    "询价命名规范度": 100,
    "计算准确度": 98,
    "报价完整度": 95,
    "报价及时性": 100
  },
  "扣分项": [
    {"评分维度": "计算准确度", "扣分原因": "公式错误", "扣分": -2},
    {"评分维度": "报价完整度", "扣分原因": "缺少产品列", "扣分": -5}
  ],
  "user_id": "employee-1001",
  "service_id": "crm",
  "session_id": "quote-training-001",
  "file_name": "DAY 1 报价练习题.xlsx",
  "format_type": "json",
  "status": "success"
}
```

生成随机密钥：

```bash
openssl rand -hex 32
```

至少需要替换：

- `POSTGRES_PASSWORD`
- `CHAT_MESSAGE_ENCRYPTION_KEY`
- `MINIO_SECRET_ACCESS_KEY`
- `N8N_ENCRYPTION_KEY`
- `NEO4J_PASSWORD`

重要经验：

- Linux 上 `COMPOSE_FILE` 使用冒号 `:` 分隔，不使用 Windows 的分号 `;`
- Linux 上 `MODEL_CACHE_DIR` 不要使用 `E:/models`，建议使用 `/opt/mtsco-models`
- 项目使用`.env`路由到不同配置,`.env` 不要提交到 GitHub

创建模型缓存目录：

```bash
mkdir -p /opt/mtsco-models
```

## 启动生产环境

构建 API 镜像：

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml build api
```

启动基础服务：

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d postgres milvus-etcd milvus-minio milvus-standalone neo4j
```

查看状态：

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps
```

初始化 MinIO bucket：

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml --profile init run --rm minio-init
```

启动 API 和 n8n：

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d api n8n
```

检查 API：

```bash
curl http://127.0.0.1:8000/health
```

正常返回：

```json
{"status":"ok"}
```

常用访问地址：

- FastAPI 文档：`http://服务器IP:8000/docs`
- 
- n8n：`http://服务器IP:5678`
- MinIO 控制台：`http://服务器IP:9001`
- Milvus：`http://服务器IP:19530`
若访问不畅通，请到服务器安全组中添加对应端口，不同厂商提供的入口有所不同，请自行检索。

## 初始化数据库和向量库

如果在 Docker API 容器内执行脚本：

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec api python scripts/db/init_postgre.py
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec api python scripts/db/init_milvus.py
```

如果需要导入或重建知识库向量：

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec api python scripts/ingestion/reindex_all.py --rebuild false
```

需要重新解析、切分、embedding、写入 Milvus 时：

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec api python scripts/ingestion/reindex_all.py --rebuild true
```

注意：当前生产配置为远程 embedding / rerank，因此重建 embedding 会调用外部 API，可能产生费用，也可能受 API 限速影响。

## 在服务器宿主机创建 `.venv` 跑 scripts

如果希望在宿主机直接运行 `scripts/`，而不是进入 Docker API 容器执行，可以在项目根目录创建 `.venv`。

安装 Python 3.11 和 uv：

```bash
apt install -y python3.11 python3.11-venv python3-pip
pip install uv
```

创建虚拟环境：

```bash
cd /opt/mtsco-knowledge-base
uv venv --python 3.11 .venv
uv pip install -e .
```

验证：

```bash
.venv/bin/python -c "import fastapi, pymilvus, torch; print('ok')"
```

宿主机运行脚本时，`.env` 里的服务地址必须是宿主机可访问地址，而不是 Docker 容器内部地址。

容器内地址示例：

```env
MILVUS_URI=http://milvus-standalone:19530
POSTGRES_HOST=postgres
MINIO_ENDPOINT=http://milvus-minio:9000
N8N_API_BASE_URL=http://n8n:5678
```

宿主机 `.venv` 跑脚本时应改为：

```env
MILVUS_URI=http://127.0.0.1:19530
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5432
MINIO_ENDPOINT=http://127.0.0.1:9000
N8N_API_BASE_URL=http://127.0.0.1:5678
```

更推荐保留两套环境文件：

- `.env`：给 Docker Compose / API 容器使用
- `.env.host`：给宿主机 `.venv` 跑 scripts 使用

运行脚本：

```bash
cd /opt/mtsco-knowledge-base
. .venv/bin/activate

python scripts/db/init_postgre.py
python scripts/db/init_milvus.py
python scripts/ingestion/reindex_all.py --rebuild false
```

## 同步 data 目录

如果 `data/` 没有进入 git，需要单独同步。Linux / macOS / WSL 推荐：

```bash
rsync -av --progress ./data/ root@服务器IP:/opt/mtsco-knowledge-base/data/
```

Windows 没有 rsync 时，可以压缩后上传：

```powershell
cd E:\MTSCO_knowledge_base
tar -czf data.tar.gz data
scp .\data.tar.gz root@服务器IP:/opt/mtsco-knowledge-base/
```

服务器解压：

```bash
cd /opt/mtsco-knowledge-base
tar -xzf data.tar.gz
```

## 同步 data/raw 到minio
```bash
.\.venv\Scripts\python.exe scripts\storage\upload_raw_documents.py data\raw --bucket knowledge-raw-docs --raw-root data\raw --init-folders --continue-on-error
```

## 更新部署

代码更新后：

```bash
cd /opt/mtsco-knowledge-base
git pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml build api
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

只重启 API：

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml restart api
```

查看日志：

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f api
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f n8n
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f milvus-standalone
```

## 上线前检查清单

- `.env` 已改为服务器生产配置
- Linux `COMPOSE_FILE` 使用 `:` 分隔
- `MODEL_CACHE_DIR=/opt/mtsco-models`
- `USE_LOCAL_EMBEDDING_MODEL=false`
- `USE_LOCAL_RERANK_MODEL=false`
- `SILICONFLOW_API_KEY` 已配置
- PostgreSQL、MinIO、n8n、Neo4j 密码已替换
- `docker compose ps` 中核心服务健康
- `curl http://127.0.0.1:8000/health` 正常
- MinIO bucket 已初始化
- Milvus collection 已创建
- 知识库数据已导入或确认可检索
- n8n workflow 已导入并启用
- 飞书回调公网地址可访问
- 重要 volume 和 `data/` 已规划备份

## 常见坑

1. Linux 服务器继续使用 `E:/models` 会导致模型缓存挂载异常。
2. Linux 上 `COMPOSE_FILE=docker-compose.yml;docker-compose.prod.yml` 不生效，应使用 `:`。
3. 宿主机 `.venv` 跑 scripts 时，不能使用 `postgres`、`milvus-standalone` 这类容器内 DNS 地址。
4. `.env` 不能提交到 GitHub/GitLab，生产密钥必须留在服务器。
5. 执行 `docker compose down -v` 会删除 Docker volumes，可能清空 PostgreSQL、Milvus、MinIO、n8n 数据。
6. 重建 embedding 会调用外部 API，需要关注费用、限速和耗时。

# 常见维护问题
- docker api 重启
```bash 
cd E:\MTSCO_knowledge_base
docker rm -f api
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --no-deps --force-recreate --no-build api
```

## 其它脚本执行指令
Linux服务器操作：
1. 进入crontab
```bash
crontab -e
```
```bash
CRON_TZ=Asia/Shanghai
0 9 * * * cd /path/to/MTSCO_knowledge_base && .venv/bin/python scripts/reports/send_daily_report.py --department 迈拓思学园 >> logs/daily_report.log 2>&1
35 17 * * 5 cd /path/to/MTSCO_knowledge_base && .venv/bin/python scripts/reports/send_weekly_report.py --department 迈拓思学园 >> logs/weekly_report.log 2>&1
```

周报统计最近一个已结束的“上周五 17:30—本周五 17:30”区间。需要立即验证发送效果时，可单次强制触发：

```bash
.venv/bin/python scripts/reports/send_daily_report.py --department 迈拓思学园 --force
.venv/bin/python scripts/reports/send_weekly_report.py --department 迈拓思学园 --force
```

也可以在 `.env` 中设置 `DAILY_REPORT_DEPARTMENTS` 和 `WEEKLY_REPORT_DEPARTMENTS`；多个部门使用英文逗号分隔。命令行 `--department` 优先于环境变量，并可重复传入。

本地window操作：
```powershell 
.\.venv\Scripts\python.exe scripts\reports\send_daily_report.py --loop
.\.venv\Scripts\python.exe scripts\reports\send_weekly_report.py --loop
.\.venv\Scripts\python.exe scripts\reports\send_weekly_report.py --force
```
> 查看任务状态
```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -like "*send_daily_report.py*" } |
  Select-Object ProcessId, CommandLined
```

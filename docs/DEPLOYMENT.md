# 部署手册

我在生产环境使用 Docker 承载 PostgreSQL、MinIO、Milvus、Neo4j，使用宿主机 systemd 承载 API、DeepSeek Harness 和归档调度器。飞书问答使用 FastAPI 进程内后台任务，不启用独立 `answer-worker`。

## 1. Windows 本地部署

我使用 Python 3.11、Git、Docker Desktop 和带 Corepack 的 Node.js。先创建环境并启动数据服务：

```powershell
Copy-Item .env.dev.example .env
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .

docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile local-minio up -d `
  postgres milvus-minio milvus-etcd milvus-standalone neo4j
docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile init run --rm minio-init
```

API 在宿主机运行，因此我把 `.env` 中的连接地址改为宿主机映射端口，并启用 Harness 与持久队列：

```env
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=15432
MILVUS_URI=http://127.0.0.1:19531
MINIO_ENDPOINT=http://127.0.0.1:19000
NEO4J_URI=bolt://127.0.0.1:17687
API_ROUTE_PREFIX=/prod
FEISHU_ROUTE_PREFIX=/prod
HARNESS_ENABLED=true
FEISHU_DURABLE_QUEUE_ENABLED=false
```

安装 Harness 并初始化存储：

```powershell
.\scripts\harness\install-windows.ps1
.\.venv\Scripts\python.exe scripts\db\init_postgre.py
.\.venv\Scripts\python.exe scripts\db\init_milvus.py
.\.venv\Scripts\python.exe scripts\db\init_neo4j.py
```

我分别打开三个 PowerShell 终端：

```powershell
# 终端 1：API
.\scripts\harness\run-windows.ps1 -Port 8000 -ApiPrefix /prod

# 终端 2：持久回答队列
.\scripts\harness\run-windows-worker.ps1 -Concurrency 1

# 终端 3：上下文封存
.\scripts\harness\run-windows-scheduler.ps1
```

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8000/health/ready
docker compose -f docker-compose.yml -f docker-compose.dev.yml ps
```

## 2. Linux 基础服务

```bash
cd /opt/mtsco-knowledge-base
cp .env.prod.example .env
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d \
  postgres milvus-minio milvus-etcd milvus-standalone neo4j
docker compose -f docker-compose.yml -f docker-compose.prod.yml --profile init run --rm minio-init
```

我不把 `docker compose down -v` 作为维护命令，因为它会删除持久卷。旧版 n8n 不属于新版运行服务。

## 3. Linux 宿主机运行时

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e .
./scripts/harness/install-linux.sh
cp .env.host.example .env.host
.venv/bin/python scripts/db/init_postgre.py
.venv/bin/python scripts/db/init_milvus.py
.venv/bin/python scripts/db/init_neo4j.py
```

`.env` 保存密钥和通用设置；`.env.host` 只覆盖容器 DNS、工作目录和生产并发控制。我会按实际安装路径调整 `.env.host`，并确认：

- `HARNESS_WORKDIR` 指向解析文本目录；
- `HARNESS_CONTEXT_ARCHIVE_TOKENS=90000`；
- `HARNESS_GLOBAL_CONCURRENCY=2`；
- `FEISHU_DURABLE_QUEUE_ENABLED=false`。

## 4. systemd

服务模板位于 `deploy/systemd`，默认项目用户为 `mtsco`、安装目录为 `/opt/mtsco-knowledge-base`。实际值不同时，我先修改三个模板。

```bash
sudo cp deploy/systemd/mtsco-*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl disable --now mtsco-answer-worker 2>/dev/null || true
sudo systemctl enable --now mtsco-api mtsco-harness-scheduler
```

检查：

```bash
systemctl status mtsco-api mtsco-harness-scheduler
curl http://127.0.0.1:8000/health/ready
curl http://127.0.0.1:8000/metrics
journalctl -u mtsco-api -f
journalctl -u mtsco-harness-scheduler -f
```

`/health/ready` 必须返回 PostgreSQL ready。`/metrics` 应包含 `mtsco_answer_jobs`、`mtsco_answer_job_oldest_queued_seconds` 和 `mtsco_harness_sessions`。

## 5. 更新与数据维护

```bash
cd /opt/mtsco-knowledge-base
git pull --ff-only
.venv/bin/python -m pip install -e .
sudo systemctl restart mtsco-api mtsco-harness-scheduler

# 同步现有解析结果；true 会重新解析并调用 embedding，可能产生费用
.venv/bin/python scripts/ingestion/reindex_all.py --rebuild false
```

Harness 版本或补丁变化时，我重新运行 `scripts/harness/install-linux.sh`，再重启 API 和 Worker。

## 6. 上线顺序

我先启动基础服务，再启动 API、Worker、调度器，最后切换飞书回调。Webhook 成功入队后会返回 `job_id`；同一飞书事件重复投递只保留一个任务。调度器同时处理 7 小时空闲归档和 90,000 token 长度归档，长度归档的新会话只消费一次旧会话摘要。

公网只转发飞书回调、经过鉴权的外部 API 和业务需要的文档下载；`/metrics`、`/docs`、内部附件、检索、图谱和存储管理路由留在可信网络。上线前我确认 `.env` 未进入 Git、数据卷已有备份、附件读取权限已开通，并且 `/health/ready` 返回预期状态。

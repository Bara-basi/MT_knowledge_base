# 新版服务器部署与迁移手册

本文面向 Linux（以 Ubuntu/Debian 为例）服务器，部署新版 DeepSeek Harness 架构。运行时由 Docker Compose 承载 PostgreSQL、MinIO、Milvus、Neo4j；宿主机由 systemd 承载 FastAPI 和 Harness 上下文封存调度器。飞书消息由 FastAPI `BackgroundTasks` 处理，**不启动 n8n，也不启动 `mtsco-answer-worker`**。

## 1. 先确认目标架构

```text
飞书回调 -> Nginx/HTTPS -> FastAPI -> DeepSeek Harness
                                      |- 只读工作台：data/processing
                                      |- MCP：混合检索、图谱、营销资料、历史、附件
                                      `- PostgreSQL / Milvus / Neo4j / MinIO
```

Harness 的源码固定在 `/opt/mtsco/deepseek-harness`，项目通过 `scripts/harness/install-linux.sh` 拉取指定版本、应用安全与会话恢复补丁、构建 Node 运行时并安装 Python SDK。不要把 Harness 当成单独的 HTTP 服务启动。

## 2. 从旧版升级前：停旧链路并备份

以下操作保留数据库与对象存储数据；不要执行 `docker compose down -v`，也不要删除 Docker volumes、`data/processing` 或 `.env`。

```bash
sudo systemctl disable --now mtsco-answer-worker n8n 2>/dev/null || true
cd /opt/mtsco-knowledge-base
sudo mkdir -p /opt/mtsco-backups/$(date +%F)
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec -T postgres \
  sh -c 'pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB"' \
  | sudo tee /opt/mtsco-backups/$(date +%F)/postgres.sql >/dev/null
```

另外备份 MinIO bucket（原始资料、解析产物和对话记忆）及宿主机 `data/processing`。原 n8n 工作流和凭据可导出存档，但不再作为新版依赖。

## 3. 服务器前置条件

安装 Docker Engine/Compose、Git、Python 3.11、Node.js（含 Corepack）及构建工具。旧格式 DOC/PPT 附件解析还需要 LibreOffice；若不处理这两种旧格式，可不装。

```bash
sudo apt-get update
sudo apt-get install -y git python3.11 python3.11-venv python3-pip build-essential
# 可选：仅旧版 DOC/PPT 附件解析需要
sudo apt-get install -y libreoffice
node --version
corepack --version
docker compose version
```

创建专用服务账户和安装目录；已有安装可跳过创建账户。

```bash
sudo useradd --system --create-home --shell /usr/sbin/nologin mtsco 2>/dev/null || true
sudo mkdir -p /opt/mtsco-knowledge-base /opt/mtsco
sudo chown -R mtsco:mtsco /opt/mtsco-knowledge-base /opt/mtsco
sudo -u mtsco git clone <REPOSITORY_URL> /opt/mtsco-knowledge-base
cd /opt/mtsco-knowledge-base
```

## 4. 配置环境变量

```bash
cp .env.prod.example .env
cp .env.host.example .env.host
chmod 600 .env .env.host
```

在 `.env` 填入所有真实密钥，至少包括 `POSTGRES_PASSWORD`、`NEO4J_PASSWORD`、`MINIO_ACCESS_KEY_ID`、`MINIO_SECRET_ACCESS_KEY`、`CHAT_MESSAGE_ENCRYPTION_KEY`、`DEEPSEEK_API_KEY`、飞书应用的 `FEISHU_APP_ID` / `FEISHU_APP_SECRET` / `FEISHU_VERIFICATION_TOKEN`，以及实际公网 `PUBLIC_BASE_URL`。示例中的 `change-me` 和示例 URL 不可用于生产。

`.env.host` 只覆盖宿主机进程必须使用的地址与绝对路径。按实际目录检查以下值：

```env
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5432
MILVUS_URI=http://127.0.0.1:19530
MINIO_ENDPOINT=http://127.0.0.1:9000
NEO4J_URI=bolt://127.0.0.1:7687
HARNESS_ENABLED=true
HARNESS_WORKDIR=/opt/mtsco-knowledge-base/data/processing
HARNESS_SESSION_ROOT=/opt/mtsco-knowledge-base/data/harness_sessions
HARNESS_GLOBAL_CONCURRENCY=2
FEISHU_DURABLE_QUEUE_ENABLED=false
```

`HARNESS_WORKDIR` 必须是已解析纯文本的目录，并对 `mtsco` 可读；Harness 只以受限、只读方式访问该目录。`HARNESS_SESSION_ROOT` 与 `HARNESS_ATTACHMENT_ROOT` 必须对 `mtsco` 可读写。不要把容器内地址（如 `postgres`、`milvus-standalone`）写入 `.env.host`。

## 5. 启动基础服务与初始化

只启动数据服务；不要在这里启动 Compose 中的 `api` 服务，因为生产 API 使用宿主机的 Harness 运行时。

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d \
  postgres milvus-minio milvus-etcd milvus-standalone neo4j
docker compose -f docker-compose.yml -f docker-compose.prod.yml --profile init run --rm minio-init
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps
```

然后安装项目依赖、固定版本的 Harness 与数据库 schema：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
DHS_REPO=/opt/mtsco/deepseek-harness ./scripts/harness/install-linux.sh
.venv/bin/python scripts/db/init_postgre.py
.venv/bin/python scripts/db/init_milvus.py
.venv/bin/python scripts/db/init_neo4j.py
```

若这是从旧版迁移且解析文本已经在 `data/processing`，先执行不重建索引的同步检查：

```bash
.venv/bin/python scripts/ingestion/reindex_all.py --rebuild false
```

只有需要重新解析全部资料或重建 Milvus collection 时才执行 `--rebuild true`；该操作会调用 embedding，可能耗时并产生费用。产品标准 TXT 如需供下载服务使用，可再运行 `scripts/ingestion/sync_standard_text_assets.py`。

## 6. 启动 systemd 服务

服务模板默认使用用户 `mtsco`、项目目录 `/opt/mtsco-knowledge-base` 和 Harness 目录 `/opt/mtsco/deepseek-harness`。路径不同则先修改 `deploy/systemd/mtsco-api.service` 与 `deploy/systemd/mtsco-harness-scheduler.service`。

```bash
sudo cp deploy/systemd/mtsco-api.service deploy/systemd/mtsco-harness-scheduler.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl disable --now mtsco-answer-worker 2>/dev/null || true
sudo systemctl enable --now mtsco-api mtsco-harness-scheduler
sudo systemctl status mtsco-api mtsco-harness-scheduler --no-pager
```

验证时，`/health/ready` 中应显示 `"harness_enabled": true` 且 `"durable_queue_enabled": false`：

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/health/ready
curl --fail http://127.0.0.1:8000/metrics | grep mtsco_harness_sessions
journalctl -u mtsco-api -n 100 --no-pager
journalctl -u mtsco-harness-scheduler -n 100 --no-pager
```

## 7. 反向代理与飞书切换

只向公网暴露 HTTPS 反向代理；FastAPI 的 `127.0.0.1:8000`、Docker 数据库端口、`/metrics`、`/docs`、检索/图谱/附件内部接口和 MinIO 控制台均应留在可信网络。飞书事件订阅 URL 设为：

```text
https://<your-domain>/prod/feishu/events
```

飞书事件加密目前未实现，因此事件订阅不要启用加密；验证令牌必须与 `FEISHU_VERIFICATION_TOKEN` 一致。先在飞书开发者后台通过 URL 校验，再发送一条实际消息。正常 webhook 会立即返回 `accepted`，随后由进程内后台任务更新进度卡片并发送最终答案；不会返回 `job_id`。

## 8. 升级与回滚边界

常规代码升级：

```bash
cd /opt/mtsco-knowledge-base
git pull --ff-only
.venv/bin/python -m pip install -e .
DHS_REPO=/opt/mtsco/deepseek-harness ./scripts/harness/install-linux.sh
.venv/bin/python scripts/db/init_postgre.py
sudo systemctl restart mtsco-api mtsco-harness-scheduler
```

Harness 版本或补丁变化时必须重新运行安装脚本。回滚时恢复已验证的项目 Git 版本并重启 API、调度器；数据回滚应使用第 2 节备份单独处理，不能靠删除 volumes 实现。会话调度器会在用户空闲 7 小时或上下文达到阈值后，将摘要写入 PostgreSQL、完整记录归档到 MinIO，并清理对应本地 JSONL 与临时附件。

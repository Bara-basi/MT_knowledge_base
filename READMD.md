# MTSCO Knowledge Base 部署说明

## 依赖安装与 Docker Compose 启动

### 1. 准备环境

需要提前安装：

- Python 3.11
- uv
- Docker Desktop / Docker Engine
- Docker Compose

进入项目根目录：

```powershell
cd <project-root>
```

### 2. 配置环境变量

复制示例配置：

```powershell
Copy-Item .env.example .env
```

根据实际环境修改 `.env` 中的数据库、对象存储、模型路径、API Key 等配置。

开发或首次部署时，建议先保留：

```env
SKIP_RETRIEVAL_WARMUP=true
```

这样 FastAPI 可以先启动，避免模型或向量库未准备好时阻塞启动流程。

### 3. 安装本地 Python 依赖

项目依赖声明在 `pyproject.toml` 中

```powershell
uv venv --python 3.11
uv pip install -e .
```

如果需要验证安装，请运行：

```powershell
uv run python -c "import importlib.metadata as m; print(m.version('mtsco-knowledge-base'))"
```

### 4. 本地启动 FastAPI

仅启动后端 API：

```powershell
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

测试接口状态请访问：

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

### 5. 使用 Docker Compose 启动完整服务

启动全部服务，在根目录运行：

```powershell
docker compose up -d
```

查看状态：

```powershell
docker compose ps
```

查看 API 日志：

```powershell
docker compose logs --tail=100 api
```

常用地址：

```text
FastAPI: http://127.0.0.1:8000/docs
n8n:     http://127.0.0.1:5678
MinIO:   http://127.0.0.1:9001
Neo4j:   http://127.0.0.1:7474
```

### 6. 启用完整检索能力

当 Milvus、模型缓存、数据入库流程和相关外部服务都准备好后，可在 `.env` 中改为：

```env
SKIP_RETRIEVAL_WARMUP=false
```

然后重新创建 API 容器：

```powershell
docker compose up -d --force-recreate api
```

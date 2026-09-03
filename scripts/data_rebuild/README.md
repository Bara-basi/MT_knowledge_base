# 知识库重建：第一步（可恢复清空）

本目录的脚本用于知识库重建的第一步：先把旧数据完整备份到
`data/backup/<run-id>`，验证备份，再清空旧知识库数据。

## 范围

- 本地：`data/raw`、`data/processing`、`data/metadata/local2lark_mapping`
- MinIO：`knowledge-raw-docs`、`knowledge-processed-docs`
- PostgreSQL：`ingestion_registry`、`lark_document_catalog`、`marketing_asset_catalog`
- Milvus：当前知识向量集合 `mtsco_knowledge_chunks`
- Neo4j：当前数据库中的全部知识图谱节点和关系

不会触及 `data/src/vector_src.json`、聊天/会话/用户数据或 MinIO 标准素材桶。

## 执行

先只生成备份并打印计划：

```powershell
.\.venv\Scripts\python.exe scripts\data_rebuild\backup_and_clear_knowledge.py --dry-run
```

执行备份、校验与清空：

```powershell
.\.venv\Scripts\python.exe scripts\data_rebuild\backup_and_clear_knowledge.py --apply
```

如果备份成功但清空阶段因连接/运行错误中断，可直接使用该份已就绪备份继续清空，
不会重复创建备份：

```powershell
.\.venv\Scripts\python.exe scripts\data_rebuild\clear_knowledge_from_backup.py `
  data\backup\knowledge-reset-YYYYMMDDTHHMMSSZ
```

脚本完成后会输出备份目录，并在其中写入 `manifest.json` 和
`READY_TO_RESTORE.json`。没有该就绪文件的备份不得用于恢复。

## 恢复

恢复前目标知识库必须保持为空；恢复脚本会拒绝覆盖非空目标。

```powershell
.\.venv\Scripts\python.exe scripts\data_rebuild\restore_knowledge_backup.py `
  data\backup\knowledge-reset-YYYYMMDDTHHMMSSZ
```

恢复后，运行 `verify_knowledge_reset.py --expect-restored <backup-dir>` 校验数量。

## 复现和审计

- `backup_and_clear_knowledge.py`：备份、校验、清空。
- `restore_knowledge_backup.py`：从备份恢复。
- `verify_knowledge_reset.py`：检查空库或恢复后的数量。
- `clear_knowledge_from_backup.py`：使用已验证备份继续已中断的清空。

所有脚本均会读取项目 `.env` 中现有连接配置；不会在日志或清单中写出密钥。

## 飞书 → OSS 初始同步

先验证全部 `vector_src.json` 文档能否下载。失败项只写入简洁 JSON：

```powershell
.\.venv\Scripts\python.exe scripts\data_rebuild\sync_feishu_knowledge_to_oss.py --check-only
```

确认后执行下载、OSS 上传和 `lark_document_catalog` 写入：

```powershell
.\.venv\Scripts\python.exe scripts\data_rebuild\sync_feishu_knowledge_to_oss.py
```

`data/src/vector_src.json` 的键同时是 OSS 的相对存储路径约定：

- `single_file` 的键必须含“目录/文件名”。例如
  `迈拓思学园（内部）/报价锦囊（希蒙雷斯）.docx` 会写为
  `knowledge/迈拓思学园（内部）/报价锦囊（希蒙雷斯）.docx`。
- `wiki` 的键是所选 Wiki 文件夹的目录路径。例如
  `质检部文档/工厂交接` 会作为 `knowledge/质检部文档/工厂交接/`。
  同步只下载该目录树中没有子节点的叶子文档；飞书中代表目录本身的文档和所有
  中间目录文档均不下载。

飞书知识库的首级页面可能显示 `has_child=false`，但以 `sub_page_list` 指向
整个知识空间的首层节点。同步脚本会自动通过空间首层节点接口展开这种入口。
可先进行只读诊断：

```powershell
.\.venv\Scripts\python.exe scripts\data_rebuild\diagnose_feishu_wiki_roots.py `
  --url "https://tmqhw1h9zt.feishu.cn/wiki/RAGDwSQbpi50M4knMjlcyqVUnih" `
  --url "https://tmqhw1h9zt.feishu.cn/wiki/SL1kw1pDziNnuOkWYiPcXaNDnxe"
```

素材仅同步飞书路径和链接，不下载内容：

```powershell
.\.venv\Scripts\python.exe scripts\data_rebuild\sync_feishu_material_catalog.py --source data\src\material.json
```

## OSS → 解析 → 向量库（开发验证）

新版入库只读取 `lark_document_catalog` 中带有 `oss_object_key` 的文档，不使用
MinIO 或 `ingestion_registry`。原始文件只在 `data/processing` 临时下载给解析器，
不会保留在 Harness 工作区。Harness 仅保留按飞书目录组织的 TXT 和图片：
`$HARNESS_WORKDIR/knowledge/<飞书目录>/<完整文档名>/txt/` 与 `img/`。目录名保留源文件
扩展名，便于与 OSS/飞书源文件一一匹配。例如
`$HARNESS_WORKDIR/knowledge/迈拓思学院/成长手册使用说明.docx/txt/成长手册使用说明.txt`。
chunk 和 embedding 只在
`data/processing/lark/` 暂存，Milvus 写入成功后会自动删除。

Linux 服务器应确保 `EMBEDDING_CACHE_DIR` 和 `RERANKER_CACHE_DIR` 是 `mtsco`
用户可写的 Linux 路径；推荐在 `.env.host` 中设置为
`/opt/mtsco-knowledge-base/data/models`。不要沿用旧 Windows 配置 `E:\\models`。
同样，首次部署或曾以 `root` 运行重建脚本后，应将运行期目录交给 `mtsco`：

```bash
sudo install -d -o mtsco -g mtsco -m 0750 \
  /opt/mtsco-knowledge-base/data/processing \
  /opt/mtsco-knowledge-base/data/harness \
  /opt/mtsco-knowledge-base/data/models
sudo chown -R mtsco:mtsco \
  /opt/mtsco-knowledge-base/data/processing \
  /opt/mtsco-knowledge-base/data/harness \
  /opt/mtsco-knowledge-base/data/models
```

先用五份文档验证全链路：

```powershell
.\.venv\Scripts\python.exe -u scripts\data_rebuild\ingest_oss_knowledge.py `
  --limit 5 --continue-on-error
```

如只检查解析、清洗和切块，不写入向量库且保留临时产物：

```powershell
.\.venv\Scripts\python.exe -u scripts\data_rebuild\ingest_oss_knowledge.py `
  --limit 5 --no-upsert --continue-on-error
```

若解析已完成、`data/processing/lark/*/chunk/*.chunks.json` 仍在，但 Milvus 暂时
不可用，恢复后可从 chunk 直接继续 embedding 和向量写入，不会重新下载或解析：

```powershell
.\.venv\Scripts\python.exe -u scripts\data_rebuild\ingest_oss_knowledge.py `
  --resume-from-chunks --continue-on-error
```

该模式优先复用已有 `data/processing/global.bm25.json`；每份文档成功写入 Milvus 后，
只删除该文档的 chunk 和 embedding 临时文件。

## 日常飞书增量刷新

飞书文档后续变更时，运行下面脚本。它会暴力扫描全部映射来源、逐份下载并计算
SHA-256；只有新增、内容哈希变化或 OSS 路径变化的文档才会上传 OSS、更新 catalog
并重新解析、嵌入、覆盖 Milvus 中该飞书文档原有的 chunks。

```powershell
.\.venv\Scripts\python.exe -u scripts\data_rebuild\refresh_feishu_knowledge.py `
  --image-workers 3
```

首次可先用 `--dry-run` 检查变更数量；它不会写 OSS、PostgreSQL 或 Milvus：

```powershell
.\.venv\Scripts\python.exe -u scripts\data_rebuild\refresh_feishu_knowledge.py --dry-run
```

自动入库只接受 `.docx`、`.xlsx`、`.pptx` 与原生文本充足的 PDF。扫描件 PDF、
低文本 PDF、不可读文件和其他格式会被拒绝，名称写入
`data/metadata/lark_incremental_failures.json`。增量入库会加载已有全局 BM25 模型，
以避免改变旧向量的稀疏向量词表。

## 服务器每日增量刷新（systemd）

仓库包含 `deploy/systemd/mtsco-feishu-refresh.service` 和对应的 timer。部署代码和
项目 Python 依赖后，在服务器执行：

```bash
sudo install -m 0644 deploy/systemd/mtsco-feishu-refresh.service /etc/systemd/system/
sudo install -m 0644 deploy/systemd/mtsco-feishu-refresh.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now mtsco-feishu-refresh.timer
systemctl list-timers mtsco-feishu-refresh.timer
```

timer 每天 02:30（`Asia/Shanghai`）运行；`Persistent=true` 表示服务器在计划时间
关机时，会在下次启动后补跑一次。任务读取与 API 相同的 `.env`、`.env.host`，不需要
重启 API 服务。首次建议手动验证一次：

```bash
sudo systemctl start mtsco-feishu-refresh.service
sudo journalctl -u mtsco-feishu-refresh.service -n 200 --no-pager
```

查看后续执行状态：

```bash
systemctl list-timers mtsco-feishu-refresh.timer
sudo journalctl -u mtsco-feishu-refresh.service -f
```

同一时刻只允许一个刷新任务运行；若需要手动运行，请先确认该 service 不在 `active`
状态，避免与定时任务争用 OSS、数据库和 Milvus 写入。

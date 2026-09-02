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

# 基础评测

`basic_cases.draft.jsonl` 是待内部用户确认的初始案例，不应被视为正式金标准。

只校验案例格式，不调用模型：

```powershell
.\.venv\Scripts\python.exe scripts\evaluation\run_basic_eval.py
```

显式执行真实 Harness 评测：

```powershell
.\.venv\Scripts\python.exe scripts\evaluation\run_basic_eval.py --live
```

真实运行会为每个案例创建隔离会话并产生模型费用，结果写入 `evals/results/`。
HTML 报告同时展示问题、完整回答、自动检查和人工复核标准。自动通过只代表“非空、关键词和泄密模式”等底线检查通过，不代表业务答案正确；内部用户应补充每条案例的 `review_criteria` 并做最终判断。

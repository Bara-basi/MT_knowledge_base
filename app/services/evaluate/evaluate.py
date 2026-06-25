from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from app.services.llm import LLMAPIError, LLMClient


DEFAULT_DATASET_FILE = Path("data") / "dataset" / "test.json"
DEFAULT_OUTPUT_FILE = Path("data") / "dataset" / "evaluation.json"
DEFAULT_TABLE_FILE = Path("data") / "dataset" / "evaluation_table.png"
DEFAULT_BAD_CASES_CSV_FILE = Path("data") / "dataset" / "evaluation_bad_cases.csv"
DEFAULT_QUERY_URL = "http://localhost:8000/api/v1/query"
DEFAULT_RETRIEVAL_PATH = "/retrieval/flow"
DEFAULT_RETRIEVAL_LIMIT = 15

SCORE_THRESHOLDS = {
    "readability": 3.0,
    "correctness": 4.0,
    "completeness": 4.0,
}


@dataclass(frozen=True)
class QueryCallResult:
    answer: str
    raw_response: Any


@dataclass(frozen=True)
class RecallResult:
    score: int
    reason: str
    matched_chunk_index: int | None
    chunk_count: int
    latency_ms: float
    error: str = ""


def log(message: str, *, verbose: bool = True) -> None:
    if verbose:
        timestamp = datetime.now().strftime("%H:%M:%S")
        try:
            from tqdm import tqdm

            tqdm.write(f"[{timestamp}] [评测] {message}")
        except ImportError:
            print(f"[{timestamp}] [评测] {message}", flush=True)


JUDGE_SYSTEM_PROMPT = """你是企业内部知识库回答质量评测员。你需要基于题目、标准答案、文档依据和系统实际回答，给出严格但实用的评分。

评分维度：
1. readability 可读性：站在普通员工视角，预测用户愿意完整读完回答的概率。回答应清楚、简洁、有结构。图片等内容目前可能仍是占位符，不应因占位符本身扣分，但如果回答把占位符当成最终内容则应扣分。
2. correctness 正确性：回答是否与参考答案和文档依据一致，由于回答的模型可能能比出题模型看到更多的信息，仅可以通过参考答案和实际回答重叠部分判断是否正确，以及是否有幻觉。如果实际回答涵盖更多内容，不能直接判定为幻觉。关于隐私信息：目前暂不考虑此合规性问题，知识库中出现的账密等均为公司内部公开信息，不存在隐私泄露问题。
3. completeness 完整性：回答是否覆盖参考答案的核心内容；回答多于参考答案但不矛盾，可以视为完整。回答少于参考答案但提问未要求回答完整，也视为正确。只有当答案与参考答案明显冲突时，才应该扣分。

核心通过指标
1. 以实际提问为准，只要回答符合用户提问就可以通过，即使和参考答案原文有差异
2. 参考答案也可能出错，以是否解决实际问题为准

泛化指标
1. 实际回答如果给出额外信息，不算错，因为回答模型可以看到更多内容
2. 实际回答没有给出完整的参考答案，但符合用户提问，不算错
3. 实际回答包含知识库中不存在的内容，如果明确说明不是知识库内容或回答的内容是常见知识（如数理化知识、常见网站地址、开放性问题的解答）等，不算错

环境说明：
- 参考答案是出题模型给出的答案，由AI生成，并非权威
- 出题模型可以看到完整文档，因此可以判断某些不存在，但实际回答模型得到的是知识库的知识片段，即使找到了全部内容，也无法判定是否完整，比如提问：“XXX图片组中的四号图片说了什么？”，参考答案：“根本没有四号图片”，如果实际回答类似：“未找到四号图片”，属于正常情况。
输出要求：
你的输出必须是 JSON，不要输出 Markdown 代码块。"""


JUDGE_USER_PROMPT_TEMPLATE = """请评测下面这条企业知识库问答结果。

题目类型：{question_types}
用户问题：
{question}

标准答案：
{reference_answer}

文档依据：
{evidence}

系统实际回答：
{actual_answer}

请输出 JSON 对象，字段如下：
- scores: 对象，包含 readability、correctness、completeness 三个 0-5 分数字。
- pass: 布尔值。若 correctness >= 4 且 completeness >= 4 且 readability >= 3，通常为 true。
- reasons: 对象，分别用一句中文解释 readability、correctness、completeness 的评分。
- missing_points: 字符串数组，列出遗漏的关键点；没有则为空数组。
- hallucinations: 字符串数组，列出与文档或标准答案冲突的内容；没有则为空数组。
- suggested_answer: 如果实际回答不合格，给出更好的简短答案；合格时可以为空字符串。
"""


def evaluate_dataset(
    dataset_file: str | Path = DEFAULT_DATASET_FILE,
    output_file: str | Path = DEFAULT_OUTPUT_FILE,
    *,
    table_output_file: str | Path = DEFAULT_TABLE_FILE,
    bad_cases_csv_file: str | Path = DEFAULT_BAD_CASES_CSV_FILE,
    query_url: str = DEFAULT_QUERY_URL,
    retrieval_url: str | None = None,
    retrieval_limit: int = DEFAULT_RETRIEVAL_LIMIT,
    model: str | None = None,
    limit: int | None = None,
    seed: int | None = None,
    sleep_seconds: float = 0.0,
    user_id: str = "evaluation",
    verbose: bool = True,
    live_dashboard: bool = True,
) -> dict[str, Any]:
    dataset_path = Path(dataset_file)
    log(f"正在加载测试数据集：{dataset_path}", verbose=verbose)
    dataset = load_dataset(dataset_path)
    if limit is not None:
        original_count = len(dataset)
        dataset = limit_dataset(dataset, limit)
        log(
            f"已随机抽样 {len(dataset)} 条，共 {original_count} 条"
            + (f"，随机种子={seed}" if seed is not None else ""),
            verbose=verbose,
        )
    else:
        log(f"已加载 {len(dataset)} 条测试用例", verbose=verbose)

    retrieval_url = retrieval_url or default_retrieval_url(query_url)
    log(f"问答接口地址：{query_url}", verbose=verbose)
    log(f"召回接口地址：{retrieval_url}", verbose=verbose)
    log("正在初始化裁判模型客户端", verbose=verbose)
    llm = LLMClient()
    log(f"裁判模型：{model or llm.settings.model}", verbose=verbose)
    results: list[dict[str, Any]] = []
    skipped_query_failures = 0
    bad_case_writer = BadCaseCsvWriter(Path(bad_cases_csv_file))
    dashboard = LiveEvaluationDashboard(total=len(dataset), enabled=live_dashboard)
    if dashboard.enabled:
        log(f"已启动实时评估看板：{dashboard.backend}", verbose=verbose)

    progress = build_progress_bar(total=len(dataset), verbose=verbose)
    for index, item in enumerate(dataset, start=1):
        question = str(item.get("question", "")).strip()
        if not question:
            log(f"[{index}/{len(dataset)}] 跳过空问题", verbose=verbose)
            progress.update(1)
            continue

        log(
            f"[{index}/{len(dataset)}] 开始测试：{shorten(question, 80)}",
            verbose=verbose,
        )
        started = time.perf_counter()
        actual_answer = ""
        raw_query_response: Any = None
        query_error = ""
        try:
            query_result = call_query_api_with_retries(
                query_url,
                question,
                user_id=user_id,
                metadata={
                    "evaluation_item_id": item.get("id"),
                    "document_name": item.get("document_name"),
                    "question_types": item.get("question_types", []),
                },
                verbose=verbose,
            )
            actual_answer = query_result.answer
            raw_query_response = query_result.raw_response
        except Exception as exc:  # noqa: BLE001 - keep batch evaluation moving.
            query_error = str(exc)

        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        if query_error:
            log(
                f"[{index}/{len(dataset)}] 问答接口失败，耗时 {latency_ms}ms：{query_error}",
                verbose=verbose,
            )
            progress.update(1)
            skipped_query_failures += 1
            continue
        else:
            log(
                f"[{index}/{len(dataset)}] 问答接口返回成功，耗时 {latency_ms}ms，答案长度 {len(actual_answer)} 字符",
                verbose=verbose,
            )

        log(f"[{index}/{len(dataset)}] 正在检查近似召回", verbose=verbose)
        recall = evaluate_recall(
            item=item,
            question=question,
            retrieval_url=retrieval_url,
            retrieval_limit=retrieval_limit,
            raw_query_response=raw_query_response,
            verbose=verbose,
        )
        log(
            f"[{index}/{len(dataset)}] 召回结果={recall.score}，片段数={recall.chunk_count}，"
            f"耗时={recall.latency_ms}ms，原因={recall.reason}",
            verbose=verbose,
        )

        log(f"[{index}/{len(dataset)}] 正在评估答案质量", verbose=verbose)
        judge_started = time.perf_counter()
        try:
            judgment = judge_answer_logged(
                llm,
                item=item,
                actual_answer=actual_answer,
                query_error=query_error,
                model=model,
                verbose=verbose,
            )
        except Exception as exc:  # noqa: BLE001 - keep batch evaluation moving.
            judgment = failed_judgment(f"裁判模型评估失败：{exc}")
            log(f"[{index}/{len(dataset)}] 裁判模型评估失败：{exc}", verbose=verbose)
        log(
            f"[{index}/{len(dataset)}] 答案质量评估完成，耗时 "
            f"{round((time.perf_counter() - judge_started) * 1000, 2)}ms",
            verbose=verbose,
        )

        result = {
            "item_id": item.get("id"),
            "document_name": item.get("document_name"),
            "document_path": item.get("document_path"),
            "question_types": item.get("question_types", []),
            "question": question,
            "reference_answer": item.get("reference_answer", ""),
            "evidence": item.get("evidence", ""),
            "actual_answer": actual_answer,
            "query_error": query_error,
            "latency_ms": latency_ms,
            "recall": {
                "score": recall.score,
                "reason": recall.reason,
                "matched_chunk_index": recall.matched_chunk_index,
                "chunk_count": recall.chunk_count,
                "latency_ms": recall.latency_ms,
                "error": recall.error,
            },
            "judgment": judgment,
        }
        results.append(result)
        if is_bad_quality_result(result):
            bad_case_writer.append(result)
            log(f"[{index}/{len(dataset)}] 已记录质量不佳案例：{bad_case_writer.path}", verbose=verbose)
        dashboard.update(results)
        log(format_item_result(index, len(dataset), result), verbose=verbose)
        update_progress_bar(progress, result)

        if sleep_seconds > 0 and index < len(dataset):
            log(f"[{index}/{len(dataset)}] 等待 {sleep_seconds}s 后继续", verbose=verbose)
            time.sleep(sleep_seconds)

    progress.close()
    log(f"正在汇总 {len(results)} 条测试结果", verbose=verbose)
    table_path = Path(table_output_file)
    report = {
        "metadata": {
            "dataset_file": str(dataset_path),
            "table_output_file": str(table_path),
            "bad_cases_csv_file": str(Path(bad_cases_csv_file)),
            "query_url": query_url,
            "retrieval_url": retrieval_url,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "judge_model": model or llm.settings.model,
            "total": len(results),
            "skipped_query_failures": skipped_query_failures,
        },
        "summary": summarize_results(results),
        "results": results,
    }

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"正在写入 JSON 评测报告：{output_path}", verbose=verbose)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    try:
        save_evaluation_dashboard_image(results, total=len(dataset), output_file=table_path)
        log(f"已生成 matplotlib 评测图表：{table_path}", verbose=verbose)
    except Exception as exc:  # noqa: BLE001 - table rendering should not hide JSON output.
        log(f"生成 matplotlib 评测图表失败：{exc}", verbose=verbose)
    log("评测完成", verbose=verbose)
    dashboard.close()
    return report


BAD_CASE_CSV_FIELDS = [
    "evaluated_at",
    "item_id",
    "document_name",
    "document_path",
    "question_types",
    "question",
    "reference_answer",
    "actual_answer",
    "query_error",
    "recall_score",
    "recall_reason",
    "readability",
    "correctness",
    "completeness",
    "error_categories",
    "reason",
    "missing_points",
    "hallucinations",
    "suggested_answer",
    "latency_ms",
]


class BadCaseCsvWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8-sig", newline="") as file:
            csv.DictWriter(file, fieldnames=BAD_CASE_CSV_FIELDS).writeheader()

    def append(self, result: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=BAD_CASE_CSV_FIELDS)
            writer.writerow(bad_case_csv_row(result))


class LiveEvaluationDashboard:
    def __init__(self, *, total: int, enabled: bool = True) -> None:
        self.total = total
        self.enabled = False
        self.backend = ""
        self.plt: Any = None
        self.fig: Any = None
        if not enabled:
            return
        try:
            import matplotlib

            if not select_interactive_matplotlib_backend(matplotlib):
                return
            import matplotlib.pyplot as plt

            configure_matplotlib_chinese_font(matplotlib)
            plt.ion()
            self.fig = plt.figure(figsize=(9.5, 4.8), dpi=120)
            self.fig.patch.set_facecolor("#f7f8fb")
            self.plt = plt
            self.backend = matplotlib.get_backend()
            self.enabled = True
            plt.show(block=False)
            self.update([])
        except Exception:
            self.enabled = False

    def update(self, results: list[dict[str, Any]]) -> None:
        if not self.enabled:
            return
        draw_evaluation_dashboard(self.fig, results, total=self.total)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        return
        metric_labels = {
            "readability": "可读性",
            "correctness": "正确性",
            "completeness": "完整性",
        }
        metric_colors = {
            "readability": "#2563eb",
            "correctness": "#16a34a",
            "completeness": "#f59e0b",
        }
        for axis, metric in zip(self.axes[:3], metric_labels, strict=True):
            axis.clear()
            passed, failed = metric_pass_fail_counts(results, metric)
            total_count = passed + failed
            values = [passed, failed] if total_count else [0, 1]
            axis.pie(
                values,
                startangle=90,
                counterclock=False,
                colors=[metric_colors[metric], "#e5e7eb"],
                wedgeprops={"width": 0.34, "edgecolor": "white", "linewidth": 2},
            )
            percent = passed / total_count * 100 if total_count else 0.0
            axis.text(0, 0.06, f"{percent:.1f}%", ha="center", va="center", fontsize=18, fontweight="bold", color="#111827")
            axis.text(0, -0.18, f"{metric_labels[metric]}达标率", ha="center", va="center", fontsize=11, color="#374151")
            axis.set_aspect("equal")

        bar_axis = self.axes[3]
        bar_axis.clear()
        category_counts = failed_question_type_counts(results)
        labels = list(category_counts.keys()) or ["暂无问题"]
        values = list(category_counts.values()) or [0]
        colors = ["#dc2626", "#ea580c", "#7c3aed", "#0891b2", "#4f46e5", "#be123c"]
        bar_axis.bar(labels, values, color=[colors[index % len(colors)] for index in range(len(labels))])
        bar_axis.set_title("不通过案例的问题类型分布", fontsize=14, fontweight="bold", color="#111827", pad=12)
        bar_axis.set_ylabel("数量", fontsize=10, color="#374151")
        bar_axis.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        bar_axis.spines["top"].set_visible(False)
        bar_axis.spines["right"].set_visible(False)
        bar_axis.tick_params(axis="x", labelrotation=18, labelsize=9)
        for index, value in enumerate(values):
            bar_axis.text(index, value + 0.05, str(value), ha="center", va="bottom", fontsize=9)

        evaluated_count = len(results)
        pass_count = sum(1 for result in results if result.get("judgment", {}).get("pass"))
        self.fig.suptitle(
            f"知识库问答评估实时看板  {evaluated_count}/{self.total}  通过 {pass_count}",
            fontsize=18,
            fontweight="bold",
            color="#111827",
        )
    def close(self) -> None:
        if not self.enabled:
            return
        try:
            self.plt.ioff()
            self.plt.close(self.fig)
            return
        except Exception:
            pass


def configure_matplotlib_chinese_font(matplotlib: Any) -> None:
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False


def select_interactive_matplotlib_backend(matplotlib: Any) -> bool:
    current_backend = matplotlib.get_backend().lower()
    if "agg" not in current_backend:
        return True

    for backend in ("TkAgg", "QtAgg", "Qt5Agg", "WXAgg"):
        try:
            matplotlib.use(backend, force=True)
            return True
        except Exception:
            continue
    return False


def save_evaluation_dashboard_image(
    results: list[dict[str, Any]],
    *,
    total: int,
    output_file: str | Path,
) -> None:
    import matplotlib

    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    configure_matplotlib_chinese_font(matplotlib)
    import matplotlib.pyplot as plt

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9.5, 4.8), dpi=180)
    try:
        draw_evaluation_dashboard(fig, results, total=total)
        fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)


def draw_evaluation_dashboard(fig: Any, results: list[dict[str, Any]], *, total: int) -> None:
    fig.clear()
    fig.patch.set_facecolor("#f7f8fb")
    pie_panel, bar_axis = fig.subplots(
        1,
        2,
        gridspec_kw={"width_ratios": [1.05, 1.35], "wspace": 0.26},
    )

    pie_panel.set_title("三项质量达标率", fontsize=13, fontweight="bold", color="#111827", pad=10)
    pie_panel.set_facecolor("#ffffff")
    pie_panel.set_xticks([])
    pie_panel.set_yticks([])
    for spine in pie_panel.spines.values():
        spine.set_visible(False)

    metric_specs = [
        ("readability", "可读性", "#2563eb", (0.03, 0.22, 0.30, 0.56)),
        ("correctness", "正确性", "#16a34a", (0.35, 0.22, 0.30, 0.56)),
        ("completeness", "完整性", "#f59e0b", (0.67, 0.22, 0.30, 0.56)),
    ]
    for metric, label, color, bounds in metric_specs:
        axis = pie_panel.inset_axes(bounds)
        passed, failed = metric_pass_fail_counts(results, metric)
        total_count = passed + failed
        values = [passed, failed] if total_count else [0, 1]
        axis.pie(
            values,
            startangle=90,
            counterclock=False,
            colors=[color, "#e5e7eb"],
            wedgeprops={"width": 0.34, "edgecolor": "white", "linewidth": 1.5},
        )
        percent = passed / total_count * 100 if total_count else 0.0
        axis.text(0, 0.05, f"{percent:.0f}%", ha="center", va="center", fontsize=13, fontweight="bold", color="#111827")
        axis.text(0, -1.24, label, ha="center", va="center", fontsize=9.5, color="#374151")
        axis.set_aspect("equal")

    category_counts = failed_question_type_counts(results)
    labels = list(category_counts.keys()) or ["暂无问题"]
    values = list(category_counts.values()) or [0]
    colors = ["#dc2626", "#ea580c", "#7c3aed", "#0891b2", "#4f46e5", "#be123c"]
    bar_axis.set_facecolor("#ffffff")
    bar_axis.bar(labels, values, color=[colors[index % len(colors)] for index in range(len(labels))], width=0.62)
    bar_axis.set_title("不通过案例的问题类型分布", fontsize=13, fontweight="bold", color="#111827", pad=10)
    bar_axis.set_ylabel("数量", fontsize=9.5, color="#374151")
    bar_axis.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    bar_axis.spines["top"].set_visible(False)
    bar_axis.spines["right"].set_visible(False)
    bar_axis.spines["left"].set_color("#d1d5db")
    bar_axis.spines["bottom"].set_color("#d1d5db")
    bar_axis.tick_params(axis="x", labelrotation=18, labelsize=8.5)
    max_value = max(values) if values else 0
    bar_axis.set_ylim(0, max(1.0, max_value * 1.25))
    for index, value in enumerate(values):
        bar_axis.text(index, value + max(0.05, max_value * 0.03), str(value), ha="center", va="bottom", fontsize=8.5, color="#111827")

    evaluated_count = len(results)
    pass_count = sum(1 for result in results if result.get("judgment", {}).get("pass"))
    fig.suptitle(
        f"知识库问答评估  {evaluated_count}/{total}  通过 {pass_count}",
        fontsize=15,
        fontweight="bold",
        color="#111827",
        y=0.98,
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.84, bottom=0.16, wspace=0.26)


def is_bad_quality_result(result: dict[str, Any]) -> bool:
    return not bool(result.get("judgment", {}).get("pass"))


def failed_reason_categories(result: dict[str, Any]) -> list[str]:
    judgment = result.get("judgment", {})
    if judgment.get("pass"):
        return []
    categories = ["模型判定不通过"]
    if result.get("query_error"):
        categories.append("问答接口失败")
    if int(result.get("recall", {}).get("score", 0) or 0) <= 0:
        categories.append("召回未命中")
    if ensure_str_list(judgment.get("hallucinations")):
        categories.append("疑似幻觉")
    if ensure_str_list(judgment.get("missing_points")):
        categories.append("关键点遗漏")
    return list(dict.fromkeys(categories))


def error_categories(result: dict[str, Any]) -> list[str]:
    categories: list[str] = []
    judgment = result.get("judgment", {})
    scores = judgment.get("scores", {})
    if result.get("query_error"):
        categories.append("问答接口失败")
    if int(result.get("recall", {}).get("score", 0) or 0) <= 0:
        categories.append("召回未命中")
    if float(scores.get("readability", 0.0) or 0.0) < SCORE_THRESHOLDS["readability"]:
        categories.append("可读性不足")
    if float(scores.get("correctness", 0.0) or 0.0) < SCORE_THRESHOLDS["correctness"]:
        categories.append("正确性不足")
    if float(scores.get("completeness", 0.0) or 0.0) < SCORE_THRESHOLDS["completeness"]:
        categories.append("完整性不足")
    if ensure_str_list(judgment.get("hallucinations")):
        categories.append("疑似幻觉")
    if ensure_str_list(judgment.get("missing_points")):
        categories.append("关键点遗漏")
    return list(dict.fromkeys(categories))


def bad_category_counts(results: list[dict[str, Any]]) -> Counter[str]:
    return failed_question_type_counts(results)


def failed_question_type_counts(results: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for result in results:
        if not is_bad_quality_result(result):
            continue
        question_types = result.get("question_types") or []
        if isinstance(question_types, str):
            question_types = [question_types]
        normalized_types = [
            str(value).strip()
            for value in question_types
            if str(value).strip()
        ]
        counts.update(normalized_types or ["未分类"])
    return counts


def metric_pass_fail_counts(results: list[dict[str, Any]], metric: str) -> tuple[int, int]:
    threshold = SCORE_THRESHOLDS[metric]
    passed = 0
    failed = 0
    for result in results:
        score = float(result.get("judgment", {}).get("scores", {}).get(metric, 0.0) or 0.0)
        if score >= threshold:
            passed += 1
        else:
            failed += 1
    return passed, failed


def bad_case_csv_row(result: dict[str, Any]) -> dict[str, Any]:
    judgment = result.get("judgment", {})
    scores = judgment.get("scores", {})
    recall = result.get("recall", {})
    return {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "item_id": result.get("item_id", ""),
        "document_name": result.get("document_name", ""),
        "document_path": result.get("document_path", ""),
        "question_types": "; ".join(str(value) for value in result.get("question_types", [])),
        "question": result.get("question", ""),
        "reference_answer": result.get("reference_answer", ""),
        "actual_answer": result.get("actual_answer", ""),
        "query_error": result.get("query_error", ""),
        "recall_score": recall.get("score", ""),
        "recall_reason": recall.get("reason", ""),
        "readability": scores.get("readability", ""),
        "correctness": scores.get("correctness", ""),
        "completeness": scores.get("completeness", ""),
        "error_categories": "; ".join(failed_reason_categories(result)),
        "reason": judgment.get("reason", ""),
        "missing_points": "; ".join(ensure_str_list(judgment.get("missing_points"))),
        "hallucinations": "; ".join(ensure_str_list(judgment.get("hallucinations"))),
        "suggested_answer": judgment.get("suggested_answer", ""),
        "latency_ms": result.get("latency_ms", ""),
    }


def build_progress_bar(*, total: int, verbose: bool) -> Any:
    try:
        from tqdm import tqdm
    except ImportError:
        return NullProgressBar()
    return tqdm(
        total=total,
        desc="评测进度",
        unit="条",
        dynamic_ncols=True,
        disable=not verbose,
    )


class NullProgressBar:
    def update(self, amount: int) -> None:
        return None

    def set_postfix(self, *args: Any, **kwargs: Any) -> None:
        return None

    def close(self) -> None:
        return None


def update_progress_bar(progress: Any, result: dict[str, Any]) -> None:
    scores = result.get("judgment", {}).get("scores", {})
    progress.set_postfix(
        {
            "通过": "是" if result.get("judgment", {}).get("pass") else "否",
            "召回": result.get("recall", {}).get("score", 0),
            "正确性": scores.get("correctness", 0.0),
        }
    )
    progress.update(1)


def format_item_result(index: int, total: int, result: dict[str, Any]) -> str:
    judgment = result.get("judgment", {})
    scores = judgment.get("scores", {})
    status = "通过" if judgment.get("pass") else "未通过"
    query_status = "成功" if not result.get("query_error") else "失败"
    return (
        f"[{index}/{total}] 当前用例结果：{status}；"
        f"问答={query_status}；"
        f"召回={result.get('recall', {}).get('score', 0)}；"
        f"可读性={scores.get('readability', 0.0)}；"
        f"正确性={scores.get('correctness', 0.0)}；"
        f"完整性={scores.get('completeness', 0.0)}；"
        f"耗时={result.get('latency_ms', 0.0)}ms"
    )


def render_evaluation_table(report: dict[str, Any], output_file: str | Path) -> None:
    try:
        import matplotlib

        if "matplotlib.pyplot" not in sys.modules:
            matplotlib.use("Agg")
        configure_matplotlib_chinese_font(matplotlib)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("未安装 matplotlib，请先安装项目依赖。") from exc

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = report.get("results", [])
    summary = report.get("summary", {})
    average_scores = summary.get("average_scores", {})

    percent_metrics = [
        ("Pass Rate", clamp_percentage(summary.get("pass_rate", 0.0) * 100)),
        ("Recall Rate", clamp_percentage(summary.get("average_recall", 0.0) * 100)),
        ("Accuracy Rate", clamp_percentage(float(average_scores.get("correctness", 0.0)) / 5 * 100)),
        ("Readability Rate", clamp_percentage(float(average_scores.get("readability", 0.0)) / 5 * 100)),
        ("Completeness Rate", clamp_percentage(float(average_scores.get("completeness", 0.0)) / 5 * 100)),
    ]

    max_latency = max(
        [float(result.get("latency_ms", 0.0) or 0.0) for result in results] + [1.0]
    )
    x_labels = ["Readability", "Correctness", "Completeness", "Latency"]
    average_line = [
        clamp_percentage(float(average_scores.get("readability", 0.0)) / 5 * 100),
        clamp_percentage(float(average_scores.get("correctness", 0.0)) / 5 * 100),
        clamp_percentage(float(average_scores.get("completeness", 0.0)) / 5 * 100),
        clamp_percentage(float(summary.get("average_latency_ms", 0.0)) / max_latency * 100),
    ]

    case_lines: list[list[float]] = []
    for result in results:
        scores = result.get("judgment", {}).get("scores", {})
        case_lines.append(
            [
                clamp_percentage(float(scores.get("readability", 0.0)) / 5 * 100),
                clamp_percentage(float(scores.get("correctness", 0.0)) / 5 * 100),
                clamp_percentage(float(scores.get("completeness", 0.0)) / 5 * 100),
                clamp_percentage(float(result.get("latency_ms", 0.0) or 0.0) / max_latency * 100),
            ]
        )

    fig = plt.figure(figsize=(14, 8.4), dpi=180)
    fig.patch.set_facecolor("#f7f8fb")
    grid = fig.add_gridspec(2, 5, height_ratios=[1.0, 1.35], hspace=0.32, wspace=0.28)
    fig.suptitle(
        "Knowledge Base QA Evaluation",
        fontsize=20,
        fontweight="bold",
        color="#111827",
        y=0.975,
    )

    pie_colors = ("#2563eb", "#e5e7eb")
    for index, (label, value) in enumerate(percent_metrics):
        ax = fig.add_subplot(grid[0, index])
        ax.pie(
            [value, 100 - value],
            startangle=90,
            counterclock=False,
            colors=pie_colors,
            wedgeprops={"width": 0.34, "edgecolor": "white", "linewidth": 2},
        )
        ax.text(
            0,
            0.06,
            f"{value:.1f}%",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color="#111827",
        )
        ax.text(
            0,
            -0.18,
            label,
            ha="center",
            va="center",
            fontsize=9.2,
            color="#374151",
        )
        ax.set_aspect("equal")

    line_ax = fig.add_subplot(grid[1, :])
    line_ax.set_facecolor("#ffffff")
    x_positions = list(range(len(x_labels)))
    for case_index, values in enumerate(case_lines, start=1):
        line_ax.plot(
            x_positions,
            values,
            color="#93c5fd",
            linewidth=1.1,
            alpha=0.38,
            marker="o",
            markersize=3,
        )
    line_ax.plot(
        x_positions,
        average_line,
        color="#dc2626",
        linewidth=2.8,
        marker="o",
        markersize=7,
        label="Average",
    )
    for x_pos, value in zip(x_positions, average_line, strict=True):
        line_ax.annotate(
            f"{value:.1f}%",
            xy=(x_pos, value),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#991b1b",
            fontweight="bold",
        )

    line_ax.set_title(
        "Case Metrics Trend",
        fontsize=14,
        fontweight="bold",
        color="#111827",
        pad=12,
    )
    line_ax.set_xticks(x_positions)
    line_ax.set_xticklabels(x_labels, fontsize=10)
    line_ax.set_ylabel("Normalized Value (%)", fontsize=10, color="#374151")
    line_ax.set_ylim(0, 108)
    line_ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    line_ax.spines["top"].set_visible(False)
    line_ax.spines["right"].set_visible(False)
    line_ax.spines["left"].set_color("#d1d5db")
    line_ax.spines["bottom"].set_color("#d1d5db")
    line_ax.legend(loc="upper left", frameon=False)
    line_ax.text(
        0.99,
        -0.22,
        "Latency is normalized against the slowest evaluated case.",
        transform=line_ax.transAxes,
        ha="right",
        va="center",
        fontsize=8.5,
        color="#6b7280",
    )

    fig.subplots_adjust(left=0.055, right=0.985, top=0.905, bottom=0.12)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def clamp_percentage(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of QA items.")
    return [item for item in data if isinstance(item, dict)]


def limit_dataset(
    dataset: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    if limit >= len(dataset):
        return list(dataset)
    return list(dataset[:limit])


def call_query_api(
    query_url: str,
    question: str,
    *,
    user_id: str,
    metadata: dict[str, Any],
    verbose: bool = True,
) -> QueryCallResult:
    payload = {
        "question": question,
        "user_id": user_id,
        "metadata": metadata,
    }
    timeout = httpx.Timeout(timeout=180.0, connect=10.0)
    log("正在发送问答请求", verbose=verbose)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(query_url, json=payload)
        response.raise_for_status()

    data: Any
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type.lower():
        data = response.json()
    else:
        text = response.text.strip()
        return QueryCallResult(answer=text, raw_response=text)

    answer = extract_answer(data)
    if not answer:
        raise ValueError(f"Query API response did not contain an answer: {data}")
    return QueryCallResult(answer=answer, raw_response=data)


def call_query_api_with_retries(
    query_url: str,
    question: str,
    *,
    user_id: str,
    metadata: dict[str, Any],
    verbose: bool = True,
    retries: int = 3,
) -> QueryCallResult:
    last_error: Exception | None = None
    max_attempts = retries + 1
    for attempt in range(1, max_attempts + 1):
        try:
            return call_query_api(
                query_url,
                question,
                user_id=user_id,
                metadata=metadata,
                verbose=verbose,
            )
        except Exception as exc:  # noqa: BLE001 - evaluation should skip unstable query calls.
            last_error = exc
            if attempt >= max_attempts:
                break
            sleep_seconds = min(2.0 * attempt, 5.0)
            log(
                f"query API failed, retrying {attempt + 1}/{max_attempts} after {sleep_seconds:.0f}s: {exc}",
                verbose=verbose,
            )
            time.sleep(sleep_seconds)
    raise RuntimeError(f"query API failed after {max_attempts} attempts: {last_error}")


def extract_answer(data: Any) -> str:
    if isinstance(data, str):
        return data.strip()
    if isinstance(data, list):
        for item in data:
            answer = extract_answer(item)
            if answer:
                return answer
        return ""
    if isinstance(data, dict):
        for key in ("answer", "output", "text", "message", "result", "data"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, (dict, list)):
                answer = extract_answer(value)
                if answer:
                    return answer
    return ""


def evaluate_recall(
    *,
    item: dict[str, Any],
    question: str,
    retrieval_url: str,
    retrieval_limit: int,
    raw_query_response: Any,
    verbose: bool = True,
) -> RecallResult:
    started = time.perf_counter()
    chunks = extract_chunks(raw_query_response)
    source = "query_response"
    error = ""

    if not chunks:
        try:
            chunks = call_retrieval_api(
                retrieval_url,
                question,
                limit=retrieval_limit,
                document_name=processing_document_name(item),
                verbose=verbose,
            )
            source = "retrieval_api"
        except Exception as exc:  # noqa: BLE001 - recall should not block answer judging.
            error = str(exc)

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    if error:
        return RecallResult(
            score=0,
            reason=f"recall check failed: {error}",
            matched_chunk_index=None,
            chunk_count=0,
            latency_ms=latency_ms,
            error=error,
        )

    score, reason, matched_index = approximate_recall_score(item, chunks, source=source)
    return RecallResult(
        score=score,
        reason=reason,
        matched_chunk_index=matched_index,
        chunk_count=len(chunks),
        latency_ms=latency_ms,
    )


def call_retrieval_api(
    retrieval_url: str,
    question: str,
    *,
    limit: int,
    document_name: str | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    payload = {
        "query": question,
        "limit": limit,
        "rerank": True,
    }
    if document_name:
        payload["document_name"] = document_name
    timeout = httpx.Timeout(timeout=180.0, connect=10.0)
    log(
        f"正在发送召回请求，数量上限={limit}，文档名={document_name or '<自动>'}",
        verbose=verbose,
    )
    with httpx.Client(timeout=timeout) as client:
        response = client.post(retrieval_url, json=payload)
        response.raise_for_status()

    data = response.json()
    chunks = data.get("chunks", []) if isinstance(data, dict) else []
    return [chunk for chunk in chunks if isinstance(chunk, dict)]


def extract_chunks(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        for key in ("chunks", "contexts", "documents", "retrieved_chunks", "references"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        for value in data.values():
            nested = extract_chunks(value)
            if nested:
                return nested
    if isinstance(data, list):
        for item in data:
            nested = extract_chunks(item)
            if nested:
                return nested
    return []


def approximate_recall_score(
    item: dict[str, Any],
    chunks: list[dict[str, Any]],
    *,
    source: str,
) -> tuple[int, str, int | None]:
    if not chunks:
        return 0, f"no chunks available from {source}", None

    targets = recall_targets(item)
    evidence = str(item.get("evidence", "")).strip()
    reference_answer = str(item.get("reference_answer", "")).strip()

    for index, chunk in enumerate(chunks, start=1):
        chunk_text = normalize_text(json.dumps(chunk, ensure_ascii=False))
        for target in targets:
            if target and normalize_text(target) in chunk_text:
                return 1, f"matched document target '{target}' in chunk from {source}", index

        if evidence and fuzzy_text_overlap(evidence, chunk_text) >= 0.45:
            return 1, f"matched evidence overlap in chunk from {source}", index

        if reference_answer and fuzzy_text_overlap(reference_answer, chunk_text) >= 0.45:
            return 1, f"matched reference-answer overlap in chunk from {source}", index

    return 0, f"no chunk matched document/evidence/reference from {source}", None


def recall_targets(item: dict[str, Any]) -> list[str]:
    raw_values = [
        item.get("document_name"),
        item.get("document_path"),
        item.get("source_document_name"),
        item.get("source_document_path"),
    ]
    targets: list[str] = []
    for value in raw_values:
        if not value:
            continue
        text = str(value).strip()
        targets.append(text)
        stem = Path(text).stem
        if stem and stem != text:
            targets.append(stem)
    return [target for target in dict.fromkeys(targets) if len(normalize_text(target)) >= 2]


def processing_document_name(item: dict[str, Any]) -> str | None:
    for key in ("source_document_name", "document_name", "source_document_path", "document_path"):
        value = item.get(key)
        if not value:
            continue
        stem = Path(str(value)).stem.strip()
        if stem:
            return stem
    return None


def fuzzy_text_overlap(reference: str, candidate: str) -> float:
    reference_tokens = text_tokens(reference)
    if not reference_tokens:
        return 0.0
    candidate_tokens = text_tokens(candidate)
    if not candidate_tokens:
        return 0.0
    matched = sum(1 for token in reference_tokens if token in candidate_tokens)
    return matched / len(reference_tokens)


def text_tokens(text: str) -> set[str]:
    normalized = normalize_text(text)
    ascii_tokens = re.findall(r"[a-zA-Z0-9_]{2,}", normalized)
    cjk_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", normalized)
    cjk_bigrams: list[str] = []
    for token in cjk_tokens:
        cjk_bigrams.extend(token[index : index + 2] for index in range(len(token) - 1))
    return set(ascii_tokens + cjk_bigrams)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def default_retrieval_url(query_url: str) -> str:
    if query_url.rstrip("/").endswith("/query"):
        return f"{query_url.rstrip('/')[:-len('/query')]}{DEFAULT_RETRIEVAL_PATH}"
    return f"{query_url.rstrip('/')}{DEFAULT_RETRIEVAL_PATH}"


def shorten(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


def format_optional_reference_answer(reference_answer: Any) -> str:
    text = str(reference_answer or "").strip()
    return text or "未提供标准答案，请仅根据用户问题、文档依据和实际回答评估回答质量。"


def judge_answer_logged(
    llm: LLMClient,
    *,
    item: dict[str, Any],
    actual_answer: str,
    query_error: str,
    model: str | None,
    verbose: bool = True,
) -> dict[str, Any]:
    if query_error:
        return failed_judgment("query api failed")

    prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
        question_types=", ".join(item.get("question_types", [])),
        question=item.get("question", ""),
        reference_answer=format_optional_reference_answer(item.get("reference_answer")),
        evidence=item.get("evidence", ""),
        actual_answer=actual_answer,
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        log("正在发送裁判模型请求，优先使用 JSON 输出模式", verbose=verbose)
        reply = llm.chat(
            messages,
            model=model,
            temperature=0.0,
            max_tokens=2500,
            extra_body=build_judge_extra_body(
                llm,
                model=model,
                response_format=True,
            ),
        )
    except LLMAPIError as exc:
        if not looks_like_response_format_error(exc):
            log(f"裁判模型请求失败，无法降级重试：{exc}", verbose=verbose)
            raise
        log(
            f"裁判模型 JSON 模式失败，正在不带 response_format 重试：{exc}",
            verbose=verbose,
        )
        reply = llm.chat(
            messages,
            model=model,
            temperature=0.0,
            max_tokens=2500,
            extra_body=build_judge_extra_body(
                llm,
                model=model,
                response_format=False,
            ),
        )
    data = parse_json_object(reply)
    return normalize_judgment(data)


def failed_judgment(reason: str) -> dict[str, Any]:
    return {
        "scores": {
            "readability": 0,
            "correctness": 0,
            "completeness": 0,
        },
        "pass": False,
        "reason": reason,
        "reasons": {
            "readability": reason,
            "correctness": reason,
            "completeness": reason,
        },
        "missing_points": [reason],
        "hallucinations": [],
        "suggested_answer": "",
    }


def looks_like_response_format_error(exc: LLMAPIError) -> bool:
    message = str(exc).lower()
    return (
        "response_format" in message
        or "json_object" in message
        or "json mode" in message
    )


def build_judge_extra_body(
    llm: LLMClient,
    *,
    model: str | None,
    response_format: bool,
) -> dict[str, Any]:
    selected_model = model or llm.settings.model
    extra: dict[str, Any] = {}
    if response_format:
        extra["response_format"] = {"type": "json_object"}
    if is_kimi_thinking_model(selected_model, base_url=llm.settings.base_url):
        extra["thinking"] = {"type": "disabled"}
    return extra


def is_kimi_thinking_model(model: str, *, base_url: str) -> bool:
    normalized_model = model.strip().lower()
    normalized_base_url = base_url.strip().lower()
    return normalized_model.startswith("kimi-") or "moonshot" in normalized_base_url


def judge_answer(
    llm: LLMClient,
    *,
    item: dict[str, Any],
    actual_answer: str,
    query_error: str,
    model: str | None,
) -> dict[str, Any]:
    if query_error:
        return {
            "scores": {
                "readability": 0,
                "correctness": 0,
                "completeness": 0,
            },
            "pass": False,
            "reason": "查询接口调用失败，用户无法获得可读答案。",
            "reasons": {
                "readability": "查询接口调用失败，用户无法获得可读答案。",
                "correctness": "查询接口调用失败，无法验证答案正确性。",
                "completeness": "查询接口调用失败，未覆盖标准答案。",
            },
            "missing_points": ["查询接口调用失败"],
            "hallucinations": [],
            "suggested_answer": "",
        }

    prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
        question_types=", ".join(item.get("question_types", [])),
        question=item.get("question", ""),
        reference_answer=format_optional_reference_answer(item.get("reference_answer")),
        evidence=item.get("evidence", ""),
        actual_answer=actual_answer,
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        reply = llm.chat(
            messages,
            model=model,
            temperature=0.0,
            max_tokens=2500,
            extra_body=build_judge_extra_body(
                llm,
                model=model,
                response_format=True,
            ),
        )
    except LLMAPIError:
        reply = llm.chat(
            messages,
            model=model,
            temperature=0.0,
            max_tokens=2500,
            extra_body=build_judge_extra_body(
                llm,
                model=model,
                response_format=False,
            ),
        )
    data = parse_json_object(reply)
    return normalize_judgment(data)


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if match is None:
            raise
        data = json.loads(match.group(0))

    if not isinstance(data, dict):
        raise ValueError("Judge response JSON must be an object.")
    return data


def normalize_judgment(data: dict[str, Any]) -> dict[str, Any]:
    raw_scores = data.get("scores", {})
    if not isinstance(raw_scores, dict):
        raw_scores = {}

    scores = {
        "readability": clamp_score(raw_scores.get("readability")),
        "correctness": clamp_score(raw_scores.get("correctness")),
        "completeness": clamp_score(raw_scores.get("completeness")),
    }
    default_pass = (
        scores["readability"] >= 3
        and scores["correctness"] >= 4
        and scores["completeness"] >= 4
    )
    reasons = ensure_dict(data.get("reasons"))
    reason = str(data.get("reason") or "").strip() or summarize_judgment_reason(
        reasons=reasons,
        missing_points=ensure_str_list(data.get("missing_points")),
        hallucinations=ensure_str_list(data.get("hallucinations")),
    )

    return {
        "scores": scores,
        "pass": bool(data.get("pass", default_pass)),
        "reason": reason,
        "reasons": reasons,
        "missing_points": ensure_str_list(data.get("missing_points")),
        "hallucinations": ensure_str_list(data.get("hallucinations")),
        "suggested_answer": str(data.get("suggested_answer", "")).strip(),
    }


def clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(5.0, score))


def ensure_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(val) for key, val in value.items()}


def summarize_judgment_reason(
    *,
    reasons: dict[str, str],
    missing_points: list[str],
    hallucinations: list[str],
) -> str:
    parts: list[str] = []
    for key in ("readability", "correctness", "completeness"):
        value = reasons.get(key)
        if value:
            parts.append(f"{key}: {value}")
    if missing_points:
        parts.append(f"missing_points: {'; '.join(missing_points)}")
    if hallucinations:
        parts.append(f"hallucinations: {'; '.join(hallucinations)}")
    return " | ".join(parts)


def ensure_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "pass_rate": 0.0,
            "average_recall": 0.0,
            "average_scores": {
                "readability": 0.0,
                "correctness": 0.0,
                "completeness": 0.0,
            },
            "average_latency_ms": 0.0,
            "failed_query_count": 0,
        }

    score_names = ("readability", "correctness", "completeness")
    average_scores = {}
    for name in score_names:
        values = [
            result.get("judgment", {}).get("scores", {}).get(name, 0.0)
            for result in results
        ]
        average_scores[name] = round(statistics.mean(values), 2)

    passed = [result for result in results if result.get("judgment", {}).get("pass")]
    latencies = [float(result.get("latency_ms", 0.0)) for result in results]
    recall_scores = [
        int(result.get("recall", {}).get("score", 0))
        for result in results
    ]
    return {
        "pass_rate": round(len(passed) / len(results), 4),
        "average_recall": round(statistics.mean(recall_scores), 4),
        "recalled_count": sum(recall_scores),
        "passed_count": len(passed),
        "total_count": len(results),
        "average_scores": average_scores,
        "average_latency_ms": round(statistics.mean(latencies), 2),
        "failed_query_count": sum(1 for result in results if result.get("query_error")),
    }


def evaluate_answer_fallback(
    *,
    question: str,
    answer: str,
    user_id: str | None = None,
    session_id: str | None = None,
    conversation_id: str | None = None,
    reference_answer: str | None = None,
    evidence: str | None = None,
    question_types: list[str] | None = None,
    model: str | None = None,
    persist: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Evaluate a live QA answer and return fallback metadata."""

    llm = LLMClient()
    item = {
        "question": question,
        "reference_answer": reference_answer,
        "evidence": evidence or "",
        "question_types": question_types or [],
    }
    try:
        judgment = judge_answer_logged(
            llm,
            item=item,
            actual_answer=answer,
            query_error="",
            model=model,
            verbose=verbose,
        )
    except Exception as exc:  # noqa: BLE001 - fallback evaluation must not break replies.
        judgment = failed_judgment(f"fallback evaluation failed: {exc}")

    passed = bool(judgment.get("pass"))
    fallback = not passed
    reason = str(judgment.get("reason") or "").strip()

    result = {
        "fallback": fallback,
        "reason": reason,
        "judgment": judgment,
    }
    if persist:
        result["stored_row"] = None

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FastAPI query answers with a QA dataset.")
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_FILE),
        help=f"Dataset JSON file. Default: {DEFAULT_DATASET_FILE}",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_FILE),
        help=f"Evaluation report output JSON file. Default: {DEFAULT_OUTPUT_FILE}",
    )
    parser.add_argument(
        "--table-output",
        default=str(DEFAULT_TABLE_FILE),
        help=f"Evaluation chart PNG file. Default: {DEFAULT_TABLE_FILE}",
    )
    parser.add_argument(
        "--bad-cases-csv",
        default=str(DEFAULT_BAD_CASES_CSV_FILE),
        help=f"CSV file for bad-quality cases. Default: {DEFAULT_BAD_CASES_CSV_FILE}",
    )
    parser.add_argument(
        "--query-url",
        default=DEFAULT_QUERY_URL,
        help=f"FastAPI query endpoint. Default: {DEFAULT_QUERY_URL}",
    )
    parser.add_argument(
        "--retrieval-url",
        default=None,
        help="Retrieval endpoint for approximate recall. Defaults to query-url base + /retrieval/flow.",
    )
    parser.add_argument(
        "--retrieval-limit",
        type=int,
        default=DEFAULT_RETRIEVAL_LIMIT,
        help="Number of chunks to request for approximate recall.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override judge LLM model name. Defaults to KIMI_MODEL/LLM_MODEL or kimi-k2.6.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate the first N items in dataset order.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Deprecated compatibility option; --limit now keeps dataset order.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between query requests.",
    )
    parser.add_argument(
        "--user-id",
        default="evaluation",
        help="User id sent to the query API.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress logs.",
    )
    parser.add_argument(
        "--no-live-dashboard",
        action="store_true",
        help="Disable the live matplotlib dashboard.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_dataset(
        args.dataset,
        args.output,
        table_output_file=args.table_output,
        bad_cases_csv_file=args.bad_cases_csv,
        query_url=args.query_url,
        retrieval_url=args.retrieval_url,
        retrieval_limit=args.retrieval_limit,
        model=args.model,
        limit=args.limit,
        seed=args.seed,
        sleep_seconds=args.sleep,
        user_id=args.user_id,
        verbose=not args.quiet,
        live_dashboard=not args.no_live_dashboard,
    )
    summary = report["summary"]
    print(
        "评测完成："
        f"{summary.get('passed_count', 0)}/{summary.get('total_count', 0)} 通过，"
        f"通过率={summary.get('pass_rate', 0.0)}，"
        f"平均召回={summary.get('average_recall', 0.0)}，"
        f"平均评分={summary.get('average_scores', {})}。"
        f"JSON 报告：{args.output}；图表图片：{args.table_output}"
    )


if __name__ == "__main__":
    main()

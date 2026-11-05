"""data_sythesis/statistics.py -- 统计 search_agent 查找到的文件与 workflow 需要文件的相关性。

统计指标：
1. 相关文件（高度相关+部分相关）占总共查找到的文件的百分比
2. 每个类型（image, ppt, xlsx, pdf, docx）中相关文件占该类型总共的百分比
3. 出题使用的文件类型分布
4. 出题要求的产出文件类型分布
5. Rubric 统计与 Category 分布
6. Solver 工具调用统计
"""

import argparse
import json
import yaml
from collections import defaultdict, Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 文件扩展名 -> 类型映射
EXT_TO_TYPE = {
    ".pdf": "pdf",
    ".pptx": "ppt",
    ".ppt": "ppt",
    ".xlsx": "xlsx",
    ".xls": "xlsx",
    ".csv": "xlsx",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".bmp": "image",
    ".webp": "image",
    ".docx": "docx",
    ".doc": "docx",
}

# required_files 中的类型名 -> 标准类型名
REQUIRED_TYPE_MAP = {
    "pdf": "pdf",
    "pptx": "ppt",
    "ppt": "ppt",
    "xlsx": "xlsx",
    "image": "image",
    "docx": "docx",
    "doc": "docx",
}


def _load_solver_prefix() -> str:
    """从 config.yaml 读取 solver_agent family，返回文件前缀。"""
    config_path = Path(__file__).resolve().parent / "config.yaml"
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        family = cfg.get("agents", {}).get("solver_agent", {}).get(
            "family", cfg.get("defaults", {}).get("family", "")
        )
        return family
    except Exception:
        return ""


SOLVER_PREFIX = "claude"


def get_file_type(sandbox_path: str) -> str:
    """从 sandbox_path 提取文件类型。"""
    ext = Path(sandbox_path).suffix.lower()
    return EXT_TO_TYPE.get(ext, "other")


def collect_stats(output_dir: Path, sample_range: tuple[int, int] | None = None) -> dict:
    """遍历所有 sample 目录，收集统计数据。

    Args:
        output_dir: output 目录路径
        sample_range: 样本范围，如 (0, 10) 表示只统计 sample_0 到 sample_9，
                      (20, 100) 表示只统计 sample_20 到 sample_99。
                      None 表示统计全部样本。
    """
    total_highly_relevant = 0
    total_partially_relevant = 0
    total_irrelevant = 0
    total_required = 0

    # 按类型统计：found (highly + partially + irrelevant), highly_relevant, partially_relevant, irrelevant, relevant
    found_by_type = defaultdict(int)
    highly_relevant_by_type = defaultdict(int)
    partially_relevant_by_type = defaultdict(int)
    irrelevant_by_type = defaultdict(int)
    relevant_by_type = defaultdict(int)

    # question_agent 统计
    qa_total_attachments = 0
    qa_attachment_by_type = defaultdict(int)
    qa_attachment_counts = []  # 每个样本的附件数量
    deliverable_by_type = defaultdict(int)

    # rubric 统计
    total_rubrics = 0
    rubric_list = []
    category_counter = Counter()

    # solver 工具调用统计
    tool_counter = Counter()
    total_tool_calls = 0
    total_tool_steps = 0  # assistant 轮次（步数，一步可含多个并行调用）
    max_parallel_calls = 0  # 单步内最多并行工具调用次数
    parallel_calls_counter = Counter()  # 单步并行调用次数分布
    solver_tool_calls_per_sample = []  # 每个样本的工具调用总次数
    solver_tool_steps_per_sample = []  # 每个样本的工具调用步数

    # solver token 统计
    solver_total_tokens_per_sample = []  # 每个样本最后一条 assistant 的 prompt_tokens + completion_tokens

    sample_count = 0
    error_count = 0
    qa_count = 0

    for sample_dir in sorted(output_dir.iterdir()):
        if not sample_dir.is_dir() or not sample_dir.name.startswith("sample_"):
            continue

        # 按 sample_range 过滤
        if sample_range is not None:
            try:
                sample_idx = int(sample_dir.name.split("_", 1)[1])
            except (ValueError, IndexError):
                continue
            if not (sample_range[0] <= sample_idx < sample_range[1]):
                continue

        scenario_path = sample_dir / "scenario_result.json"
        search_path = sample_dir / "search_result.json"
        _result_name = f"{SOLVER_PREFIX}_solver_result.json" if SOLVER_PREFIX else "solver_result.json"
        _agent_name = f"{SOLVER_PREFIX}_solver_agent.json" if SOLVER_PREFIX else "solver_agent.json"
        solver_result_path = sample_dir / _result_name
        solver_agent_path = sample_dir / "conversation" / _agent_name

        if not search_path.exists():
            error_count += 1
            continue

        sample_count += 1

        # 读取 search_result.json
        search_data = json.loads(search_path.read_text(encoding="utf-8"))
        highly_relevant_files = search_data.get("highly_relevant_files", [])
        partially_relevant_files = search_data.get("partially_relevant_files", [])
        irrelevant_files = search_data.get("irrelevant_files", [])

        total_highly_relevant += len(highly_relevant_files)
        total_partially_relevant += len(partially_relevant_files)
        total_irrelevant += len(irrelevant_files)

        for f in highly_relevant_files:
            ftype = get_file_type(f.get("sandbox_path", ""))
            highly_relevant_by_type[ftype] += 1
            relevant_by_type[ftype] += 1
            found_by_type[ftype] += 1

        for f in partially_relevant_files:
            ftype = get_file_type(f.get("sandbox_path", ""))
            partially_relevant_by_type[ftype] += 1
            relevant_by_type[ftype] += 1
            found_by_type[ftype] += 1

        for f in irrelevant_files:
            ftype = get_file_type(f.get("sandbox_path", ""))
            irrelevant_by_type[ftype] += 1
            found_by_type[ftype] += 1

        # 读取 scenario_result.json（如果存在）
        if scenario_path.exists():
            scenario_data = json.loads(scenario_path.read_text(encoding="utf-8"))
            # 新格式: required_files 是字符串列表，无法提取类型
            required_files = scenario_data.get("required_files", [])
            total_required += len(required_files)

        # 读取 question_result.json（如果存在）
        question_path = sample_dir / "question_result.json"
        if question_path.exists():
            question_data = json.loads(question_path.read_text(encoding="utf-8"))
            qa_count += 1
            # 新格式: attachments 在 question.attachments 下
            question_obj = question_data.get("question", {})
            atts = question_obj.get("attachments", [])
            qa_attachment_counts.append(len(atts))
            # 统计 deliverable 类型
            deliverable = question_obj.get("deliverable", {})
            if deliverable:
                filename = deliverable.get("filename", "")
                # 多个文件名用逗号/分号分隔时归为 mixed
                if "," in filename or ";" in filename:
                    deliverable_by_type["mixed"] += 1
                else:
                    raw_type = deliverable.get("type", "")
                    std_type = REQUIRED_TYPE_MAP.get(raw_type, raw_type)
                    deliverable_by_type[std_type] += 1
            for att in atts:
                qa_total_attachments += 1
                filename = att.get("filename", "")
                ftype = get_file_type(filename)
                qa_attachment_by_type[ftype] += 1

            # 统计 rubric
            rubric = question_data.get("rubric", [])
            if isinstance(rubric, list):
                total_rubrics += len(rubric)
                rubric_list.append(len(rubric))
                for item in rubric:
                    if isinstance(item, dict):
                        category_counter[item.get("category", "unknown")] += 1

        # 统计 solver 工具调用 & token
        try:
            if not solver_result_path.exists() or not solver_agent_path.exists():
                raise FileNotFoundError
            solver_agent_data = json.loads(solver_agent_path.read_text(encoding="utf-8"))
            sample_tool_calls = 0
            sample_tool_steps = 0
            last_prompt_tokens = None
            for item in solver_agent_data:
                if item.get("role") == "assistant":
                    tcs = item.get("tool_calls", [])
                    if tcs:
                        sample_tool_steps += 1
                        max_parallel_calls = max(max_parallel_calls, len(tcs))
                        parallel_calls_counter[len(tcs)] += 1
                    for tc in tcs:
                        name = tc.get("function", {}).get("name", "unknown")
                        tool_counter[name] += 1
                        total_tool_calls += 1
                        sample_tool_calls += 1
                    usage = item.get("usage", {})
                    if usage and "prompt_tokens" in usage:
                        last_prompt_tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
            total_tool_steps += sample_tool_steps
            solver_tool_calls_per_sample.append(sample_tool_calls)
            solver_tool_steps_per_sample.append(sample_tool_steps)
            if last_prompt_tokens is not None:
                solver_total_tokens_per_sample.append(last_prompt_tokens)
        except Exception:
            pass

    total_relevant = total_highly_relevant + total_partially_relevant
    total_found = total_relevant + total_irrelevant

    return {
        "sample_count": sample_count,
        "error_count": error_count,
        "total_highly_relevant": total_highly_relevant,
        "total_partially_relevant": total_partially_relevant,
        "total_irrelevant": total_irrelevant,
        "total_relevant": total_relevant,
        "total_found": total_found,
        "total_required": total_required,
        "found_by_type": dict(found_by_type),
        "highly_relevant_by_type": dict(highly_relevant_by_type),
        "partially_relevant_by_type": dict(partially_relevant_by_type),
        "irrelevant_by_type": dict(irrelevant_by_type),
        "relevant_by_type": dict(relevant_by_type),
        # question_agent 统计
        "qa_count": qa_count,
        "qa_total_attachments": qa_total_attachments,
        "qa_attachment_by_type": dict(qa_attachment_by_type),
        "qa_attachment_counts": qa_attachment_counts,
        "deliverable_by_type": dict(deliverable_by_type),
        # rubric 统计
        "total_rubrics": total_rubrics,
        "rubric_list": rubric_list,
        "category_counter": dict(category_counter),
        # solver 工具调用统计
        "tool_counter": dict(tool_counter),
        "total_tool_calls": total_tool_calls,
        "total_tool_steps": total_tool_steps,
        "max_parallel_calls": max_parallel_calls,
        "parallel_calls_counter": dict(parallel_calls_counter),
        "solver_tool_calls_per_sample": solver_tool_calls_per_sample,
        "solver_tool_steps_per_sample": solver_tool_steps_per_sample,
        # solver token 统计
        "solver_total_tokens_per_sample": solver_total_tokens_per_sample,
    }


def print_stats(stats: dict) -> None:
    """打印统计结果。"""
    print("=" * 60)
    print("Search Agent 文件相关性统计")
    print("=" * 60)

    # 基本信息
    print(f"\n总样本数: {stats['sample_count']}")
    print(f"缺失 search_result 的样本数: {stats['error_count']}")

    # (1) 总体相关性
    total_found = stats["total_found"]
    total_highly_relevant = stats["total_highly_relevant"]
    total_partially_relevant = stats["total_partially_relevant"]
    total_irrelevant = stats["total_irrelevant"]
    total_required = stats["total_required"]

    print(f"\n{'─' * 60}")
    print("(1) 相关文件占总共查找到文件的百分比")
    print(f"{'─' * 60}")
    print(f"  Workflow 需要的文件总数:       {total_required}")
    print(f"  Search Agent 查找到的文件总数: {total_found}")
    print(f"    - 高度相关:                   {total_highly_relevant}")
    print(f"    - 部分相关:                   {total_partially_relevant}")
    print(f"    - 不相关:                     {total_irrelevant}")
    if total_found > 0:
        pct_highly = total_highly_relevant / total_found * 100
        pct_partially = total_partially_relevant / total_found * 100
        pct_irrelevant = total_irrelevant / total_found * 100
        print(f"    - 高度相关占比:               {pct_highly:.2f}%  ({total_highly_relevant}/{total_found})")
        print(f"    - 部分相关占比:               {pct_partially:.2f}%  ({total_partially_relevant}/{total_found})")
        print(f"    - 不相关占比:                 {pct_irrelevant:.2f}%  ({total_irrelevant}/{total_found})")
    else:
        print(f"  相关文件占比:                   N/A (未查找到文件)")

    # (2) 按类型统计
    all_types = sorted(set(stats["found_by_type"].keys()))

    print(f"\n{'─' * 60}")
    print("(2) 每个类型中相关文件占该类型总共的百分比")
    print(f"{'─' * 60}")
    print(f"  {'类型':<8} {'高度相关':>8} {'部分相关':>8} {'不相关':>8} {'总查找到':>8} {'高度相关占比':>12} {'部分相关占比':>12} {'不相关占比':>12}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*12} {'-'*12} {'-'*12}")

    for t in all_types:
        highly = stats["highly_relevant_by_type"].get(t, 0)
        partially = stats["partially_relevant_by_type"].get(t, 0)
        irrelevant = stats["irrelevant_by_type"].get(t, 0)
        found = stats["found_by_type"].get(t, 0)

        if found > 0:
            pct_highly = highly / found * 100
            pct_partially = partially / found * 100
            pct_irrelevant = irrelevant / found * 100
            print(f"  {t:<8} {highly:>8} {partially:>8} {irrelevant:>8} {found:>8} {pct_highly:>11.2f}% {pct_partially:>11.2f}% {pct_irrelevant:>11.2f}%")
        else:
            print(f"  {t:<8} {highly:>8} {partially:>8} {irrelevant:>8} {found:>8} {'N/A':>12} {'N/A':>12} {'N/A':>12}")

    print(f"\n{'=' * 60}")

    # (3) 出题使用的文件类型分布
    print(f"\n{'─' * 60}")
    print("(3) 出题使用的文件类型分布")
    print(f"{'─' * 60}")
    qa_total = stats["qa_total_attachments"]
    qa_attachment_counts = stats["qa_attachment_counts"]
    qa_types = sorted(t for t in stats["qa_attachment_by_type"].keys() if t != "other")
    if qa_attachment_counts:
        avg_attachments = sum(qa_attachment_counts) / len(qa_attachment_counts)
        min_attachments = min(qa_attachment_counts)
        max_attachments = max(qa_attachment_counts)
        print(f"  平均使用附件数:               {avg_attachments:.2f}")
        print(f"  最少使用附件数:               {min_attachments}")
        print(f"  最多使用附件数:               {max_attachments}")
    print(f"  {'类型':<8} {'使用数量':>8} {'占比':>10}")
    print(f"  {'-'*8} {'-'*8} {'-'*10}")
    for t in qa_types:
        cnt = stats["qa_attachment_by_type"].get(t, 0)
        if qa_total > 0:
            pct = cnt / qa_total * 100
            print(f"  {t:<8} {cnt:>8} {pct:>9.2f}%")
        else:
            print(f"  {t:<8} {cnt:>8} {'N/A':>10}")

    # (4) 出题要求的产出文件类型分布
    print(f"\n{'─' * 60}")
    print("(4) 出题要求的产出文件类型分布")
    print(f"{'─' * 60}")
    deliverable_by_type = stats["deliverable_by_type"]
    total_deliverable = sum(deliverable_by_type.values())
    print(f"  {'类型':<8} {'数量':>8} {'占比':>10}")
    print(f"  {'-'*8} {'-'*8} {'-'*10}")
    for t in sorted(deliverable_by_type.keys()):
        cnt = deliverable_by_type[t]
        pct = cnt / total_deliverable * 100 if total_deliverable > 0 else 0
        print(f"  {t:<8} {cnt:>8} {pct:>9.2f}%")

    print(f"\n{'═' * 60}")

    # (5) Rubric 统计
    rubric_list = stats["rubric_list"]
    if rubric_list:
        total_rubrics = stats["total_rubrics"]
        avg_rubrics = total_rubrics / len(rubric_list)
        print(f"\n{'─' * 60}")
        print("(5) Rubric 统计")
        print(f"{'─' * 60}")
        print(f"  总样本数:                     {len(rubric_list)}")
        print(f"  总 rubric 条目:               {total_rubrics}")
        print(f"  平均 rubric 个数:             {avg_rubrics:.2f}")
        print(f"  最小 rubric 个数:             {min(rubric_list)}")
        print(f"  最大 rubric 个数:             {max(rubric_list)}")

        category_counter = stats["category_counter"]
        if category_counter:
            total_categories = sum(category_counter.values())
            print(f"\n  Category 分布:")
            print(f"  {'类型':<20} {'数量':>8} {'占比':>10}")
            print(f"  {'-'*20} {'-'*8} {'-'*10}")
            for category, count in sorted(category_counter.items(), key=lambda x: -x[1]):
                pct = count / total_categories * 100
                print(f"  {category:<20} {count:>8} {pct:>9.2f}%")

    # (6) Solver 工具调用统计
    tool_counter = stats["tool_counter"]
    total_tool_calls = stats["total_tool_calls"]
    if tool_counter:
        print(f"\n{'─' * 60}")
        print("(6) Solver 工具调用统计")
        print(f"{'─' * 60}")
        print(f"  总调用次数 (tool calls):      {total_tool_calls}")
        print(f"  总调用步数 (tool steps):      {stats.get('total_tool_steps', 'N/A')}")
        print(f"  单步最多并行调用次数:         {stats.get('max_parallel_calls', 'N/A')}")
        parallel_dist = stats.get("parallel_calls_counter", {})
        if parallel_dist:
            total_steps = sum(parallel_dist.values())
            print(f"\n  单步并行调用次数分布:")
            print(f"  {'并行调用数':<12} {'步数':>8} {'占比':>10}")
            print(f"  {'-'*12} {'-'*8} {'-'*10}")
            for n in sorted(parallel_dist.keys()):
                cnt = parallel_dist[n]
                pct = cnt / total_steps * 100
                print(f"  {n:<12} {cnt:>8} {pct:>9.2f}%")
        print(f"  {'工具名称':<30} {'次数':>8} {'占比':>10}")
        print(f"  {'-'*30} {'-'*8} {'-'*10}")
        for tool, count in sorted(tool_counter.items(), key=lambda x: -x[1]):
            pct = count / total_tool_calls * 100
            print(f"  {tool:<30} {count:>8} {pct:>9.2f}%")

        # 每个样本工具调用次数区间分布（每5次一个区间）
        per_sample = stats.get("solver_tool_calls_per_sample", [])
        if per_sample:
            bucket_size = 5
            max_calls = max(per_sample)
            num_buckets = (max_calls // bucket_size) + 1
            buckets: dict[int, int] = defaultdict(int)
            for v in per_sample:
                buckets[v // bucket_size] += 1
            print(f"\n  每个样本工具调用次数分布（区间大小=5，平均={sum(per_sample)/len(per_sample):.2f}）:")
            print(f"  {'区间':<12} {'样本数':>8} {'占比':>10}")
            print(f"  {'-'*12} {'-'*8} {'-'*10}")
            for b in range(num_buckets):
                lo, hi = b * bucket_size, (b + 1) * bucket_size
                cnt = buckets.get(b, 0)
                pct = cnt / len(per_sample) * 100
                label = f"[{lo}, {hi})"
                print(f"  {label:<12} {cnt:>8} {pct:>9.2f}%")

        # 每个样本工具调用步数区间分布（每5步一个区间）
        per_steps = stats.get("solver_tool_steps_per_sample", [])
        if per_steps:
            bucket_size = 5
            max_steps = max(per_steps)
            num_buckets = (max_steps // bucket_size) + 1
            step_buckets: dict[int, int] = defaultdict(int)
            for v in per_steps:
                step_buckets[v // bucket_size] += 1
            print(f"\n  每个样本工具调用步数分布（区间大小=5，平均={sum(per_steps)/len(per_steps):.2f}）:")
            print(f"  {'区间':<12} {'样本数':>8} {'占比':>10}")
            print(f"  {'-'*12} {'-'*8} {'-'*10}")
            for b in range(num_buckets):
                lo, hi = b * bucket_size, (b + 1) * bucket_size
                cnt = step_buckets.get(b, 0)
                pct = cnt / len(per_steps) * 100
                label = f"[{lo}, {hi})"
                print(f"  {label:<12} {cnt:>8} {pct:>9.2f}%")

    # (7) Solver prompt_tokens + completion_tokens 统计
    prompt_tokens_list = stats.get("solver_total_tokens_per_sample", [])
    if prompt_tokens_list:
        bucket_size = 10000
        max_tokens = max(prompt_tokens_list)
        num_buckets = (max_tokens // bucket_size) + 1
        token_buckets: dict[int, int] = defaultdict(int)
        for v in prompt_tokens_list:
            token_buckets[v // bucket_size] += 1
        print(f"\n{'─' * 60}")
        print("(7) Solver 最后一条 assistant (prompt_tokens+completion_tokens) 分布（区间大小=10000）")
        print(f"{'─' * 60}")
        print(f"  样本数:   {len(prompt_tokens_list)}")
        print(f"  平均值:   {sum(prompt_tokens_list)/len(prompt_tokens_list):.0f}")
        print(f"  最小值:   {min(prompt_tokens_list)}")
        print(f"  最大值:   {max(prompt_tokens_list)}")
        print(f"\n  {'区间':<20} {'样本数':>8} {'占比':>10}")
        print(f"  {'-'*20} {'-'*8} {'-'*10}")
        for b in range(num_buckets):
            lo, hi = b * bucket_size, (b + 1) * bucket_size
            cnt = token_buckets.get(b, 0)
            pct = cnt / len(prompt_tokens_list) * 100
            label = f"[{lo}, {hi})"
            print(f"  {label:<20} {cnt:>8} {pct:>9.2f}%")

    print(f"\n{'═' * 60}")


# 饼状图配色
COLORS = {
    "highly_relevant": "#4CAF50",
    "partially_relevant": "#FF9800",
    "irrelevant": "#F44336",
}

TYPE_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4", "#E91E63", "#FFEB3B", "#795548"]


def save_pie_charts(stats: dict, view_dir: Path) -> None:
    """保存各统计指标的饼状图。"""
    view_dir.mkdir(parents=True, exist_ok=True)

    total_found = stats["total_found"]
    total_highly = stats["total_highly_relevant"]
    total_partially = stats["total_partially_relevant"]
    total_irrelevant = stats["total_irrelevant"]

    all_types = sorted(set(stats["found_by_type"].keys()))

    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "STHeiti", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    # ── (1) 总体相关性饼状图 ──
    if total_found > 0:
        labels = ["高度相关", "部分相关", "不相关"]
        sizes = [total_highly, total_partially, total_irrelevant]
        colors = [COLORS["highly_relevant"], COLORS["partially_relevant"], COLORS["irrelevant"]]
        # 过滤掉值为 0 的部分
        filtered = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
        labels, sizes, colors = zip(*filtered) if filtered else (labels, sizes, colors)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(
            sizes, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 14},
        )
        ax.set_title(f"(1) 总体相关性分布  (共 {total_found} 个文件)", fontsize=16)
        fig.tight_layout()
        fig.savefig(view_dir / "1_overall_relevance.png", dpi=150)
        plt.close(fig)

    # ── (2) 每个类型的相关性饼状图 ──
    types_with_data = [t for t in all_types if t != "other" and stats["found_by_type"].get(t, 0) > 0]
    if types_with_data:
        n = len(types_with_data)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        if n == 1:
            axes = [[axes]]
        else:
            axes = axes.reshape(nrows, ncols) if axes.ndim == 2 else axes.reshape(1, -1)

        for idx, t in enumerate(types_with_data):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            highly = stats["highly_relevant_by_type"].get(t, 0)
            partially = stats["partially_relevant_by_type"].get(t, 0)
            irrelevant = stats["irrelevant_by_type"].get(t, 0)
            found = highly + partially + irrelevant

            labels = ["高度相关", "部分相关", "不相关"]
            sizes = [highly, partially, irrelevant]
            colors = [COLORS["highly_relevant"], COLORS["partially_relevant"], COLORS["irrelevant"]]
            filtered = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
            if filtered:
                labels, sizes, colors = zip(*filtered)

            ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
                   startangle=90, textprops={"fontsize": 11})
            ax.set_title(f"{t}  (共 {found} 个)", fontsize=13)

        # 隐藏多余子图
        for idx in range(len(types_with_data), nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        fig.suptitle("(2) 每个类型的相关性分布", fontsize=16, y=1.02)
        fig.tight_layout()
        fig.savefig(view_dir / "2_relevance_by_type.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── (3) 出题使用的文件类型分布饼状图 ──
    qa_types_sorted = sorted(
        t for t in stats["qa_attachment_by_type"].keys() if t != "other"
    )
    type_sizes = [stats["qa_attachment_by_type"].get(t, 0) for t in qa_types_sorted]
    type_labels = [f"{t} ({s})" for t, s in zip(qa_types_sorted, type_sizes) if s > 0]
    type_sizes_filtered = [s for s in type_sizes if s > 0]
    type_colors_filtered = TYPE_COLORS[:len(type_sizes_filtered)]

    if type_sizes_filtered:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(type_sizes_filtered, labels=type_labels, colors=type_colors_filtered,
               autopct="%1.1f%%", startangle=90, textprops={"fontsize": 13})
        ax.set_title(f"(3) 出题使用的文件类型分布  (共 {sum(type_sizes_filtered)} 个)", fontsize=16)
        fig.tight_layout()
        fig.savefig(view_dir / "3_attachment_type_distribution.png", dpi=150)
        plt.close(fig)

    # ── (4) 出题要求的产出文件类型分布饼状图 ──
    deliverable_by_type = stats["deliverable_by_type"]
    d_types_sorted = sorted(deliverable_by_type.keys())
    d_sizes = [deliverable_by_type[t] for t in d_types_sorted]
    d_labels = [f"{t} ({s})" for t, s in zip(d_types_sorted, d_sizes) if s > 0]
    d_sizes_filtered = [s for s in d_sizes if s > 0]
    d_colors = TYPE_COLORS[:len(d_sizes_filtered)]

    if d_sizes_filtered:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(d_sizes_filtered, labels=d_labels, colors=d_colors,
               autopct="%1.1f%%", startangle=90, textprops={"fontsize": 13})
        ax.set_title(f"(4) 出题要求的产出文件类型分布  (共 {sum(d_sizes_filtered)} 个)", fontsize=16)
        fig.tight_layout()
        fig.savefig(view_dir / "4_deliverable_type_distribution.png", dpi=150)
        plt.close(fig)

    # ── (5) Rubric Category 分布饼状图 ──
    category_counter = stats["category_counter"]
    if category_counter:
        c_labels = [f"{k} ({v})" for k, v in sorted(category_counter.items(), key=lambda x: -x[1])]
        c_sizes = [v for v in sorted(category_counter.values(), reverse=True)]
        c_colors = TYPE_COLORS[:len(c_sizes)]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(c_sizes, labels=c_labels, colors=c_colors,
               autopct="%1.1f%%", startangle=90, textprops={"fontsize": 13})
        ax.set_title(f"(5) Rubric Category 分布  (共 {sum(c_sizes)} 条)", fontsize=16)
        fig.tight_layout()
        fig.savefig(view_dir / "5_rubric_category_distribution.png", dpi=150)
        plt.close(fig)

    # ── (6) Solver 工具调用分布饼状图 ──
    tool_counter = stats["tool_counter"]
    if tool_counter:
        t_items = sorted(tool_counter.items(), key=lambda x: -x[1])
        t_labels = [f"{k} ({v})" for k, v in t_items]
        t_sizes = [v for _, v in t_items]
        t_colors = TYPE_COLORS[:len(t_sizes)]
        fig, ax = plt.subplots(figsize=(10, 8))
        wedges, texts, autotexts = ax.pie(
            t_sizes, colors=t_colors, autopct="%1.1f%%", startangle=90,
            textprops={"fontsize": 11}, pctdistance=0.75,
        )
        ax.legend(wedges, t_labels, loc="lower center", bbox_to_anchor=(0.5, -0.15),
                  ncol=2, fontsize=10, frameon=False)
        ax.set_title(f"(6) Solver 工具调用分布  (共 {sum(t_sizes)} 次)", fontsize=16)
        fig.tight_layout()
        fig.savefig(view_dir / "6_solver_tool_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── (6b) Solver 每样本工具调用次数/步数区间分布柱状图 ──
    per_sample = stats.get("solver_tool_calls_per_sample", [])
    per_steps = stats.get("solver_tool_steps_per_sample", [])
    if per_sample:
        bucket_size = 5
        max_calls = max(per_sample)
        num_buckets = (max_calls // bucket_size) + 1
        buckets: dict[int, int] = defaultdict(int)
        for v in per_sample:
            buckets[v // bucket_size] += 1
        x_labels = [f"[{b*bucket_size},{(b+1)*bucket_size})" for b in range(num_buckets)]
        y_values = [buckets.get(b, 0) for b in range(num_buckets)]

        fig, ax = plt.subplots(figsize=(max(8, num_buckets * 0.8), 5))
        bars = ax.bar(x_labels, y_values, color="#2196F3", edgecolor="white")
        ax.set_xlabel("工具调用次数区间", fontsize=13)
        ax.set_ylabel("样本数", fontsize=13)
        ax.set_title(f"(6b) Solver 每样本工具调用次数分布  (共 {len(per_sample)} 个样本)", fontsize=15)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        for bar, val in zip(bars, y_values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                        str(val), ha="center", va="bottom", fontsize=10)
        fig.tight_layout()
        fig.savefig(view_dir / "6b_solver_tool_calls_per_sample_distribution.png", dpi=150)
        plt.close(fig)

    if per_steps:
        bucket_size = 5
        max_steps = max(per_steps)
        num_buckets = (max_steps // bucket_size) + 1
        step_buckets: dict[int, int] = defaultdict(int)
        for v in per_steps:
            step_buckets[v // bucket_size] += 1
        x_labels = [f"[{b*bucket_size},{(b+1)*bucket_size})" for b in range(num_buckets)]
        y_values = [step_buckets.get(b, 0) for b in range(num_buckets)]

        fig, ax = plt.subplots(figsize=(max(8, num_buckets * 0.8), 5))
        bars = ax.bar(x_labels, y_values, color="#9C27B0", edgecolor="white")
        ax.set_xlabel("工具调用步数区间", fontsize=13)
        ax.set_ylabel("样本数", fontsize=13)
        ax.set_title(f"(6c) Solver 每样本工具调用步数分布  (共 {len(per_steps)} 个样本)", fontsize=15)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        for bar, val in zip(bars, y_values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                        str(val), ha="center", va="bottom", fontsize=10)
        fig.tight_layout()
        fig.savefig(view_dir / "6c_solver_tool_steps_per_sample_distribution.png", dpi=150)
        plt.close(fig)

    print(f"饼状图已保存到: {view_dir}")


def parse_range(range_str: str) -> tuple[int, int]:
    """解析范围字符串，如 '0-10', '20-100'。"""
    parts = range_str.split("-")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"范围格式应为 'start-end'，如 '0-10'，收到: '{range_str}'")
    start, end = int(parts[0]), int(parts[1])
    if start < 0 or end <= start:
        raise argparse.ArgumentTypeError(f"范围无效: start={start}, end={end}，要求 start >= 0 且 end > start")
    return (start, end)


def main():
    parser = argparse.ArgumentParser(description="统计 search_agent 查找到的文件与 workflow 需要文件的相关性")
    parser.add_argument(
        "--range",
        type=parse_range,
        default=None,
        help="指定统计的样本范围，格式为 'start-end'，如 '0-10' 表示统计 sample_0 到 sample_9，"
             "'20-100' 表示统计 sample_20 到 sample_99。不指定则统计全部样本。",
    )
    args = parser.parse_args()
    sample_range = args.range

    output_dir = Path(__file__).resolve().parent / "output"

    if not output_dir.exists():
        print(f"输出目录不存在: {output_dir}")
        return

    if sample_range is not None:
        print(f"统计样本范围: sample_{sample_range[0]} 到 sample_{sample_range[1] - 1}")

    stats = collect_stats(output_dir, sample_range=sample_range)
    print_stats(stats)

    # 保存统计结果为 JSON
    stats_path = output_dir / "search_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"统计结果已保存到: {stats_path}")

    # 保存饼状图
    view_dir = Path(__file__).resolve().parent / "view1"
    save_pie_charts(stats, view_dir)

    # 总被统计的样本个数
    print(f"\n=== 总被统计的样本个数 ===")
    print(f"总样本数: {stats['sample_count']}")


if __name__ == "__main__":
    main()

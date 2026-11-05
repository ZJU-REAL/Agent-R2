import json
import os
from collections import defaultdict

import yaml


def _load_solver_max_steps(config_path: str) -> int:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg["solver"]["max_steps"]
    except Exception:
        return 50


def collect_stats(model_result_dir: str, max_steps: int = None):
    if max_steps is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        max_steps = _load_solver_max_steps(config_path)

    total_steps = 0
    total_tool_call_steps = 0  # 步数：一个 assistant 消息算一步（无论并行几个工具）
    tool_call_counts = defaultdict(int)
    total_samples = 0
    total_tool_calls = 0
    completed_samples = 0
    max_steps_reached_samples = 0
    traceback_counts = []
    accuracy_scores = []

    eval_completed_samples = sum(
        1 for sample_id in os.listdir(model_result_dir)
        if os.path.isfile(os.path.join(model_result_dir, sample_id, "eval_result.json"))
    )

    for sample_id in os.listdir(model_result_dir):
        sample_dir = os.path.join(model_result_dir, sample_id)
        if not os.path.isdir(sample_dir):
            continue

        conv_path = os.path.join(sample_dir, "solver_conversation.json")
        if not os.path.exists(conv_path):
            continue

        with open(conv_path, "r", encoding="utf-8") as f:
            conversation = json.load(f)

        traceback_count = 0
        sample_tool_calls = 0
        sample_tool_steps = 0
        for msg in conversation:
            if msg.get("role") == "tool" and isinstance(msg.get("content"), str):
                try:
                    result = json.loads(msg["content"])
                    if isinstance(result, dict) and "error" in result:
                        traceback_count += 1
                except (json.JSONDecodeError, ValueError):
                    pass
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                sample_tool_steps += 1
                for tc in msg["tool_calls"]:
                    tool_call_counts[tc["function"]["name"]] += 1
                    sample_tool_calls += 1
        traceback_counts.append(traceback_count)

        eval_result_path = os.path.join(sample_dir, "eval_result.json")
        solver_steps = sample_tool_steps
        if os.path.exists(eval_result_path):
            with open(eval_result_path, "r", encoding="utf-8") as f:
                eval_result = json.load(f)
            solver_steps = eval_result.get("solver_steps", sample_tool_steps)
            # completed = solver produced a non-empty predicted answer
            if eval_result.get("predicted"):
                completed_samples += 1
            accuracy_scores.append(eval_result.get("accuracy", 0.0))

        if solver_steps >= max_steps:
            max_steps_reached_samples += 1

        total_steps += sample_tool_steps
        total_tool_call_steps += sample_tool_steps
        total_tool_calls += sample_tool_calls
        total_samples += 1

    if total_samples == 0:
        return None

    traceback_distribution = defaultdict(int)
    for cnt in traceback_counts:
        if cnt == 0:
            continue
        bucket = ((cnt - 1) // 5) * 5
        traceback_distribution[bucket] += 1
    traceback_distribution = dict(sorted(traceback_distribution.items()))

    tool_usage_ratio = dict(sorted(
        {tool: count / total_tool_calls for tool, count in tool_call_counts.items()}.items(),
        key=lambda x: x[1], reverse=True,
    ))

    return {
        "avg_steps": total_steps / total_samples,
        "avg_tool_calls": total_tool_calls / total_samples,
        "tool_usage_ratio": tool_usage_ratio,
        "total_samples": total_samples,
        "total_tool_calls": total_tool_calls,
        "total_tool_call_steps": total_tool_call_steps,
        "tool_call_counts": dict(tool_call_counts),
        "completed_samples": completed_samples,
        "completion_rate": completed_samples / total_samples,
        "eval_completed_samples": eval_completed_samples,
        "max_steps_reached_samples": max_steps_reached_samples,
        "max_steps_reached_rate": max_steps_reached_samples / total_samples,
        "traceback_distribution": traceback_distribution,
        "avg_tool_errors": sum(traceback_counts) / total_samples,
        "accuracy": sum(accuracy_scores) / total_samples,
    }


def collect_stats_multi_run(base_name: str, results_root: str = None, max_steps: int = None):
    """
    自动扫描 results_root 下所有以 base_name 开头的目录，取平均并计算 pass@k。

    pass@k：对于每个样本，只要在所有运行中有至少一次 accuracy > 0，则该样本通过。
    """
    if results_root is None:
        results_root = os.path.join(os.path.dirname(__file__), "results")

    # 扫描所有以 base_name 开头的目录（含 base_name 本身）
    run_dirs = sorted([
        os.path.join(results_root, d)
        for d in os.listdir(results_root)
        if d.startswith(base_name)
        and os.path.isdir(os.path.join(results_root, d))
    ])

    run_stats_list = []
    for run_dir in run_dirs:
        stats = collect_stats(run_dir, max_steps=max_steps)
        if stats is not None:
            stats["_run_dir"] = run_dir
            run_stats_list.append(stats)

    if not run_stats_list:
        return None, None

    # --- 数值字段取平均 ---
    avg_keys = [
        "avg_steps", "avg_tool_calls", "total_samples", "total_tool_calls",
        "total_tool_call_steps", "completed_samples", "completion_rate",
        "eval_completed_samples", "max_steps_reached_samples",
        "max_steps_reached_rate", "avg_tool_errors", "accuracy",
    ]
    averaged = {}
    for key in avg_keys:
        vals = [s[key] for s in run_stats_list if key in s]
        averaged[key] = sum(vals) / len(vals) if vals else 0.0

    # tool_call_counts / tool_usage_ratio 按均值合并
    all_tools = set()
    for s in run_stats_list:
        all_tools.update(s.get("tool_call_counts", {}).keys())
    avg_tool_call_counts = {}
    for tool in all_tools:
        vals = [s["tool_call_counts"].get(tool, 0) for s in run_stats_list]
        avg_tool_call_counts[tool] = sum(vals) / len(vals)
    total_avg_tool_calls = sum(avg_tool_call_counts.values())
    tool_usage_ratio = dict(sorted(
        {tool: (count / total_avg_tool_calls if total_avg_tool_calls > 0 else 0)
         for tool, count in avg_tool_call_counts.items()}.items(),
        key=lambda x: x[1], reverse=True,
    ))

    # traceback_distribution 按均值合并
    all_buckets = set()
    for s in run_stats_list:
        all_buckets.update(s.get("traceback_distribution", {}).keys())
    avg_traceback_dist = {}
    for bucket in all_buckets:
        vals = [s["traceback_distribution"].get(bucket, 0) for s in run_stats_list]
        avg_traceback_dist[bucket] = sum(vals) / len(vals)
    avg_traceback_dist = dict(sorted(avg_traceback_dist.items()))

    averaged["tool_usage_ratio"] = tool_usage_ratio
    averaged["tool_call_counts"] = avg_tool_call_counts
    averaged["traceback_distribution"] = avg_traceback_dist

    # --- 计算 pass@k ---
    # 收集每个 sample_id 在各次运行中的 accuracy
    sample_accuracies = defaultdict(list)  # sample_id -> [acc_run1, acc_run2, ...]
    for s in run_stats_list:
        run_dir = s["_run_dir"]
        for sample_id in os.listdir(run_dir):
            sample_dir = os.path.join(run_dir, sample_id)
            if not os.path.isdir(sample_dir):
                continue
            eval_result_path = os.path.join(sample_dir, "eval_result.json")
            if os.path.exists(eval_result_path):
                with open(eval_result_path, "r", encoding="utf-8") as f:
                    eval_result = json.load(f)
                sample_accuracies[sample_id].append(eval_result.get("accuracy", 0.0))

    pass_at_k_samples = sum(1 for accs in sample_accuracies.values() if any(a > 0 for a in accs))
    total_eval_samples = len(sample_accuracies)
    pass_at_k = pass_at_k_samples / total_eval_samples if total_eval_samples > 0 else 0.0

    k = len(run_stats_list)
    pass_at_k_stats = {
        "pass_at_k": pass_at_k,
        "pass_at_k_samples": pass_at_k_samples,
        "total_eval_samples": total_eval_samples,
        "k": k,
        "actual_runs": k,
        "run_dirs": [s["_run_dir"] for s in run_stats_list],
    }

    return averaged, pass_at_k_stats


def print_stats(stats: dict, model_name: str = "", pass_at_k_stats: dict = None):
    header = f"=== 统计结果: {model_name} ===" if model_name else "=== 统计结果 ==="
    print(header)
    print(f"样本总数:           {stats['total_samples']:.1f}")
    print(f"完成样本数:         {stats['completed_samples']:.1f}")
    print(f"完成率:             {stats['completion_rate']*100:.2f}%")
    print(f"Eval完成样本数:     {stats['eval_completed_samples']:.1f}")
    print(f"Accuracy:           {stats['accuracy']*100:.2f}%")
    if pass_at_k_stats:
        k = pass_at_k_stats['k']
        actual = pass_at_k_stats['actual_runs']
        print(f"Pass@{k}:            {pass_at_k_stats['pass_at_k']*100:.2f}%  "
              f"({pass_at_k_stats['pass_at_k_samples']}/{pass_at_k_stats['total_eval_samples']} 样本通过, "
              f"基于{actual}次运行)")
    print(f"达工具上限样本数:   {stats['max_steps_reached_samples']:.1f}")
    print(f"达工具上限比例:     {stats['max_steps_reached_rate']*100:.2f}%")
    print(f"工具调用总步数:     {stats['total_tool_call_steps']:.1f}")
    print(f"工具调用总次数:     {stats['total_tool_calls']:.1f}")
    print(f"平均每样本工具步数: {stats['avg_steps']:.2f}")
    print(f"平均每样本工具次数: {stats['avg_tool_calls']:.2f}")
    print()
    print("工具调用报错（error）次数分布:")
    print(f"  平均每样本报错次数: {stats['avg_tool_errors']:.2f}")
    for bucket, count in stats["traceback_distribution"].items():
        print(f"  ({bucket}, {bucket+5}]{'':6} {count:.1f} 个样本")
    print()
    print("工具使用占比（按调用次数降序）:")
    for tool, ratio in stats["tool_usage_ratio"].items():
        count = stats["tool_call_counts"][tool]
        print(f"  {tool or '<empty>':<20} {ratio*100:6.2f}%  ({count:.1f} 次)")


if __name__ == "__main__":
    import sys

    results_root = os.path.join(os.path.dirname(__file__), "results")

    # model_name = "Qwen3-4B-Thinking-2507-rl80_3"
    model_name = "qwen3-30b-a3b-thinking-2507"
    # model_name = "qwen3-235b-a22b-thinking-2507"

    # 自动检测是否存在以 model_name 开头的多次运行目录（含 model_name 本身）
    matched_dirs = sorted([
        d for d in os.listdir(results_root)
        if d.startswith(model_name)
        and os.path.isdir(os.path.join(results_root, d))
    ])
    if len(matched_dirs) > 1:
        print(f"检测到多次运行模式，base={model_name}，匹配目录: {matched_dirs}")
        averaged, pass_at_k_stats = collect_stats_multi_run(model_name, results_root=results_root)
        if averaged:
            print_stats(averaged, model_name=f"{model_name} (avg of {pass_at_k_stats['actual_runs']} runs)",
                        pass_at_k_stats=pass_at_k_stats)
        else:
            print("未找到有效样本数据。")
    else:
        # 单次运行（matched_dirs 为空或只有 model_name 本身）
        model_result_dir = os.path.join(results_root, model_name)
        stats = collect_stats(model_result_dir)
        if stats:
            print_stats(stats, model_name=model_name)
        else:
            print("未找到有效样本数据。")

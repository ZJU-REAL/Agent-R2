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
        return 30


def collect_stats(model_result_dir: str, max_steps: int = None):
    """
    统计模型评估结果的工具使用数据。

    Args:
        model_result_dir: 模型结果目录，每个子目录对应一个样本
        max_steps: 工具调用上限次数，为 None 时自动从同级 config.yaml 的 solver.max_steps 读取

    Returns:
        dict: 包含以下统计数据：
            - avg_steps: 平均每个样本的工具执行步数
            - tool_usage_ratio: 每个工具的使用占比（按调用次数）
            - total_samples: 样本总数
            - total_tool_calls: 工具调用总次数
            - tool_call_counts: 每个工具的绝对调用次数
            - completed_samples: 完成的样本数
            - completion_rate: 完成率
            - max_steps_reached_samples: 达到工具调用上限的样本数
            - max_steps_reached_rate: 达到工具调用上限的比例
    """
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
    traceback_counts = []  # 每个样本中 Traceback 出现次数

    # 独立统计：有 eval_result.json 即算 eval 完成，不依赖 solver 结果
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

        # 检查完成率：output 目录存在且包含文件
        output_dir = os.path.join(sample_dir, "output")
        if os.path.isdir(output_dir) and any(os.path.isfile(os.path.join(output_dir, f)) for f in os.listdir(output_dir)):
            completed_samples += 1

        with open(conv_path, "r", encoding="utf-8") as f:
            conversation = json.load(f)

        # 统计工具调用报错次数（tool role 的 content JSON 中有 error 键）
        traceback_count = 0
        sample_tool_calls = 0
        sample_tool_steps = 0  # 步数：一个 assistant 消息算一步（无论并行几个工具）
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
                    tool_name = tc["function"]["name"]
                    tool_call_counts[tool_name] += 1
                    sample_tool_calls += 1
        traceback_counts.append(traceback_count)

        # 优先从 eval_result.json 读取 solver_steps，否则用对话中统计的步数
        eval_result_path = os.path.join(sample_dir, "eval_result.json")
        if os.path.exists(eval_result_path):
            with open(eval_result_path, "r", encoding="utf-8") as f:
                eval_result = json.load(f)
            solver_steps = eval_result.get("solver_steps", sample_tool_steps)
        else:
            solver_steps = sample_tool_steps
        if solver_steps >= max_steps:
            max_steps_reached_samples += 1

        total_steps += sample_tool_steps
        total_tool_call_steps += sample_tool_steps
        total_tool_calls += sample_tool_calls
        total_samples += 1

    if total_samples == 0:
        return None

    avg_steps = total_steps / total_samples
    avg_tool_calls = total_tool_calls / total_samples

    # 统计 traceback 次数区间分布，间隔为 5，跳过 0 次，左开右闭 (n, n+5]
    traceback_distribution = defaultdict(int)
    for cnt in traceback_counts:
        if cnt == 0:
            continue
        bucket = ((cnt - 1) // 5) * 5  # (0,5]->0, (5,10]->5, ...
        traceback_distribution[bucket] += 1
    traceback_distribution = dict(sorted(traceback_distribution.items()))

    tool_usage_ratio = {
        tool: count / total_tool_calls
        for tool, count in tool_call_counts.items()
    }
    # 按占比从高到低排序
    tool_usage_ratio = dict(
        sorted(tool_usage_ratio.items(), key=lambda x: x[1], reverse=True)
    )

    completion_rate = completed_samples / total_samples if total_samples > 0 else 0.0
    max_steps_reached_rate = max_steps_reached_samples / (total_samples - completed_samples) if (total_samples - completed_samples) > 0 else 0.0

    return {
        "avg_steps": avg_steps,
        "avg_tool_calls": avg_tool_calls,
        "tool_usage_ratio": tool_usage_ratio,
        "total_samples": total_samples,
        "total_tool_calls": total_tool_calls,
        "total_tool_call_steps": total_tool_call_steps,
        "tool_call_counts": dict(tool_call_counts),
        "completed_samples": completed_samples,
        "completion_rate": completion_rate,
        "eval_completed_samples": eval_completed_samples,
        "max_steps_reached_samples": max_steps_reached_samples,
        "max_steps_reached_rate": max_steps_reached_rate,
        "traceback_distribution": traceback_distribution,
        "avg_tool_errors": sum(traceback_counts) / total_samples,
    }


def print_stats(stats: dict, model_name: str = ""):
    header = f"=== 统计结果: {model_name} ===" if model_name else "=== 统计结果 ==="
    print(header)
    print(f"样本总数:           {stats['total_samples']}")
    print(f"完成样本数:         {stats['completed_samples']}")
    print(f"完成率:             {stats['completion_rate']*100:.2f}%")
    print(f"Eval完成样本数:     {stats['eval_completed_samples']}")
    print(f"达工具上限样本数:   {stats['max_steps_reached_samples']}")
    print(f"达工具上限比例:     {stats['max_steps_reached_rate']*100:.2f}%")
    print(f"工具调用总步数:     {stats['total_tool_call_steps']}")
    print(f"工具调用总次数:     {stats['total_tool_calls']}")
    print(f"平均每样本工具步数: {stats['avg_steps']:.2f}")
    print(f"平均每样本工具次数: {stats['avg_tool_calls']:.2f}")
    print()
    print("工具调用报错（error）次数分布:")
    print(f"  平均每样本报错次数: {stats['avg_tool_errors']:.2f}")
    for bucket, count in stats["traceback_distribution"].items():
        label = f"({bucket}, {bucket + 5}]"
        print(f"  {label:<12} {count} 个样本")
    print()
    print("工具使用占比（按调用次数降序）:")
    for tool, ratio in stats["tool_usage_ratio"].items():
        count = stats["tool_call_counts"][tool]
        tool_display = tool if tool else "<empty>"
        print(f"  {tool_display:<20} {ratio*100:6.2f}%  ({count} 次)")


if __name__ == "__main__":
    # name="claude-opus-4-5-20251101-kiro"
    # name="gemini-2.5-pro"
    # name="gpt-4.1"
    # name="deepseek-v3.1"
    # name="glm-4.7"
    # name="Qwen3-4B-Thinking-2507-rl80_3"
    # name="Qwen3-4B-Thinking-2507"
    # name="Qwen3-4B-Thinking-2507-claude-ckpt2000"
    # name="Qwen3-4B-Thinking-2507-rl80_3"
    # name="Qwen3-8B"
    # name="qwen3-30b-a3b-thinking-2507"
    # name="qwen3-vl-30b-a3b-thinking"
    # name="qwen3-235b-a22b-thinking-2507"
    # name="qwen3-vl-235b-a22b-thinking"
    # name="Qwen3.5-9B"
    # name="Qwen3-4B-Thinking-2507-rl78"
    # name="Qwen3-4B-Thinking-2507-claude-ckpt598"
    name="Qwen3-4B-Thinking-2507-claude-ckpt1194"

    model_result_dir = os.path.join(
        os.path.dirname(__file__), "results", name
    )
    stats = collect_stats(model_result_dir)
    if stats:
        print_stats(stats, model_name=name)
    else:
        print("未找到有效样本数据。")

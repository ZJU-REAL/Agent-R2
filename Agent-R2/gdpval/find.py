import json
import os
from pathlib import Path

# 配置：过滤文件类型（如 'pptx', 'docx', 'xlsx'），留空则包含所有类型
FILE_TYPE_FILTER = "pptx"

# 定义路径
base_path = Path(__file__).resolve().parent / "results"
models = {
    "claude": base_path / "claude-opus-4-7",
    "before_rl": base_path / "Qwen3-4B-Thinking-2507",
    "after_rl": base_path / "Qwen3-4B-Thinking-2507-rl110"
}

# 获取所有样本ID（使用claude目录作为参考）
sample_ids = [d.name for d in models["claude"].iterdir() if d.is_dir()]
print(f"找到 {len(sample_ids)} 个样本")
if FILE_TYPE_FILTER:
    print(f"过滤文件类型: {FILE_TYPE_FILTER}\n")

# 存储每个样本的数据
samples = []

for sample_id in sample_ids:
    sample_data = {"sample_id": sample_id}
    has_all_scores = True

    # 检查文件类型过滤：三个模型对应的样本都必须有该类型文件
    if FILE_TYPE_FILTER:
        has_target_file = True
        for model_name, model_path in models.items():
            output_dir = model_path / sample_id / "output"
            found = False
            if output_dir.exists():
                for f in output_dir.iterdir():
                    if f.suffix.lower() == f".{FILE_TYPE_FILTER.lower()}":
                        found = True
                        break
            if not found:
                has_target_file = False
                break
        if not has_target_file:
            continue

    for model_name, model_path in models.items():
        eval_file = model_path / sample_id / "eval_result.json"
        if eval_file.exists():
            with open(eval_file) as f:
                data = json.load(f)
                sample_data[f"{model_name}_score"] = data.get("score")
                sample_data[f"{model_name}_max_score"] = data.get("max_score")
        else:
            has_all_scores = False
            break

    if has_all_scores:
        # 检查分数是否有效
        rl_score = sample_data.get("after_rl_score")
        claude_score = sample_data.get("claude_score")
        before_rl_score = sample_data.get("before_rl_score")
        rl_max = sample_data.get("after_rl_max_score")
        claude_max = sample_data.get("claude_max_score")
        before_rl_max = sample_data.get("before_rl_max_score")

        if rl_score is not None and claude_score is not None and before_rl_score is not None:
            # 排除RL前0分的样本
            if before_rl_score > 0:
                # 计算百分比
                rl_pct = rl_score / rl_max * 100 if rl_max > 0 else 0
                claude_pct = claude_score / claude_max * 100 if claude_max > 0 else 0
                before_rl_pct = before_rl_score / before_rl_max * 100 if before_rl_max > 0 else 0

                sample_data["rl_pct"] = rl_pct
                sample_data["claude_pct"] = claude_pct
                sample_data["before_rl_pct"] = before_rl_pct

                improvement = rl_pct - before_rl_pct
                closeness = abs(rl_pct - claude_pct)

                sample_data["improvement"] = improvement
                sample_data["closeness"] = closeness
                sample_data["good_score"] = improvement - closeness * 0.5

                samples.append(sample_data)

print(f"有效样本数: {len(samples)}\n")

# 排序标准：RL 后分数高于 RL 前，且尽量接近 Claude
# good_score = (rl_score - before_rl_score) - 0.5 * |rl_score - claude_score|

for sample in samples:
    rl_score = sample["after_rl_score"]
    claude_score = sample["claude_score"]
    before_rl_score = sample["before_rl_score"]

    improvement = rl_score - before_rl_score
    closeness = abs(rl_score - claude_score)

    # 综合分数：提升量减去一个 closeness 惩罚
    # lambda 控制对closeness的惩罚程度
    # 这里我们给closeness一个较大的权重，因为它同等重要
    sample["improvement"] = improvement
    sample["closeness"] = closeness
    sample["good_score"] = improvement - closeness * 0.5

# 按good_score从高到低排序
samples.sort(key=lambda x: x["good_score"], reverse=True)

# 打印结果
print("=" * 100)
print(f"{'Sample ID':<40} {'RL后%':>7} {'RL前%':>7} {'Claude%':>7} {'提升':>7} {'差距':>6} {'好分数':>8}")
print("=" * 100)

for i, sample in enumerate(samples, 1):
    print(f"{sample['sample_id']:<40} "
          f"{sample['rl_pct']:>7.1f} "
          f"{sample['before_rl_pct']:>7.1f} "
          f"{sample['claude_pct']:>7.1f} "
          f"{sample['improvement']:>+7.1f} "
          f"{sample['closeness']:>6.1f} "
          f"{sample['good_score']:>8.2f}")

print("=" * 100)
print(f"\n排序说明：")
print(f"- '提升' = RL后分数 - RL前分数（正值表示RL后更好）")
print(f"- '差距' = |RL后分数 - Claude分数|（越小表示越接近Claude）")
print(f"- '好分数' = 提升 - 0.5 * 差距（越高表示越好）")
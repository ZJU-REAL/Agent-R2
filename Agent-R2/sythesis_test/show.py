"""
show.py - 展示满足各类条件的 case 及数量
用法: python show.py checkpoint [--type 1 2 3 ...] [--count_only]
不指定 --type 则展示所有类别

类别说明:
- 类别 1: 只有 solver_conversation.json，没有 solver_result.json
- 类别 2: 有 solver_conversation.json + solver_result.json 但 output/ 为空
- 类别 3: 有 evaluator_conversation.json + solver_*.json 但没有 eval_result.json
- 类别 4: 有 eval_result.json 和 solver_*.json，且 output/ 不为空，但没有 evaluator_conversation.json
- 类别 5: eval_result.json 中存在 reasoning 等于 'Failed to parse evaluation output'
"""
import os
import sys
import json
import argparse


def get_all_cases(checkpoint_dir: str):
    return sorted(
        os.path.join(checkpoint_dir, case_id)
        for case_id in os.listdir(checkpoint_dir)
        if os.path.isdir(os.path.join(checkpoint_dir, case_id))
    )


def check_type1(case_dir):
    """只有 solver_conversation.json，没有 solver_result.json"""
    files = set(os.listdir(case_dir))
    return "solver_conversation.json" in files and "solver_result.json" not in files


def check_type2(case_dir):
    """有 eval_result.json 和 solver_*.json，且 output/ 不为空，但没有 evaluator_conversation.json"""
    files = set(os.listdir(case_dir))
    if not (
        "eval_result.json" in files
        and "solver_conversation.json" in files
        and "solver_result.json" in files
        and "evaluator_conversation.json" not in files
    ):
        return False
    output_dir = os.path.join(case_dir, "output")
    return os.path.isdir(output_dir) and len(os.listdir(output_dir)) > 0


def check_type3(case_dir):
    """有 evaluator_conversation.json + solver_*.json 但没有 eval_result.json"""
    files = set(os.listdir(case_dir))
    return (
        "evaluator_conversation.json" in files
        and "solver_conversation.json" in files
        and "solver_result.json" in files
        and "eval_result.json" not in files
    )


def check_type4(case_dir):
    """有 solver_conversation.json + solver_result.json 但 output/ 为空"""
    files = set(os.listdir(case_dir))
    if "solver_conversation.json" not in files or "solver_result.json" not in files:
        return False
    output_dir = os.path.join(case_dir, "output")
    return os.path.isdir(output_dir) and len(os.listdir(output_dir)) == 0


def check_type5(case_dir):
    """eval_result.json 中存在 reasoning 等于 'Failed to parse evaluation output'"""
    eval_result = os.path.join(case_dir, "eval_result.json")
    if not os.path.isfile(eval_result):
        return False
    try:
        with open(eval_result) as f:
            data = json.load(f)
    except Exception:
        return False
    return any(
        e.get("reasoning") == "Failed to parse evaluation output"
        for e in data.get("evaluations", [])
    )


CHECKS = {
    1: ("只有 solver_conversation.json，没有 solver_result.json", check_type1),
    2: ("有 solver_conversation.json + solver_result.json 但 output/ 为空", check_type4),
    3: ("有 evaluator_conversation.json + solver_*.json 但没有 eval_result.json", check_type3),
    4: ("有 eval_result.json 和 solver_*.json，且 output/ 不为空，但没有 evaluator_conversation.json", check_type2),
    5: ("eval_result.json 中存在 reasoning 等于 'Failed to parse evaluation output'", check_type5),
}


def main():
    parser = argparse.ArgumentParser(description="展示各类待清理 case")
    parser.add_argument("checkpoint", help="checkpoint 目录名，例如 checkpoint-3096")
    parser.add_argument(
        "--results-dir",
        default=os.path.join(os.path.dirname(__file__), "results"),
        help="results 根目录路径（默认为脚本同级的 results/）",
    )
    parser.add_argument("--type", nargs="*", type=int, help="指定类别编号（不指定则展示全部）")
    parser.add_argument("--count_only", action="store_true", help="只打印数量，不打印每个 case 名")
    args = parser.parse_args()

    checkpoint_dir = os.path.join(args.results_dir, args.checkpoint)
    if not os.path.isdir(checkpoint_dir):
        print(f"Error: directory not found: {checkpoint_dir}")
        sys.exit(1)

    selected = args.type if args.type else list(CHECKS.keys())
    cases = get_all_cases(checkpoint_dir)

    for t in selected:
        if t not in CHECKS:
            print(f"[警告] 未知类别 {t}，跳过")
            continue
        desc, fn = CHECKS[t]
        hits = [os.path.basename(c) for c in cases if fn(c)]
        print(f"\n=== 类别 {t}: {desc} ===")
        print(f"数量: {len(hits)}")
        if not args.count_only:
            for name in hits:
                print(f"  {name}")


if __name__ == "__main__":
    main()

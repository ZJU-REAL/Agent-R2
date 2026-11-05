"""
clean.py - 删除满足各类条件的 case 对应文件
用法: python clean.py checkpoint [--type 1 2 3 ...]
不指定 --type 则删除所有类别

删除规则:
- 类别 1: 删除整个 case 目录（只有 solver_conversation.json，无 solver_result.json）
- 类别 2: 删除整个 case 目录（output/ 为空）
- 类别 3: 只删除 evaluator_conversation.json 和 eval_result.json
- 类别 4: 只删除 evaluator_conversation.json 和 eval_result.json
- 类别 5: 只删除 evaluator_conversation.json 和 eval_result.json
"""
import os
import sys
import shutil
import argparse
from show import CHECKS, get_all_cases


# 每个类别对应的删除操作: "dir"=删整个目录, "evaluator_files"=只删 evaluator_conversation.json + eval_result.json
CATEGORY_DELETE_ACTION = {
    1: "dir",
    2: "dir",
    3: "evaluator_files",
    4: "evaluator_files",
    5: "evaluator_files",
}


def delete_case(checkpoint_dir: str, case_id: str, action: str):
    case_dir = os.path.join(checkpoint_dir, case_id)
    if action == "dir":
        if os.path.isdir(case_dir):
            shutil.rmtree(case_dir)
            return 1
    elif action == "evaluator_files":
        removed = 0
        for fname in ("evaluator_conversation.json", "eval_result.json"):
            f = os.path.join(case_dir, fname)
            if os.path.isfile(f):
                os.remove(f)
                removed += 1
        return removed
    return 0


def main():
    parser = argparse.ArgumentParser(description="清理各类待删除 case 文件")
    parser.add_argument("checkpoint", help="checkpoint 目录名，例如 checkpoint-3096")
    parser.add_argument(
        "--results-dir",
        default=os.path.join(os.path.dirname(__file__), "results"),
        help="results 根目录路径（默认为脚本同级的 results/）",
    )
    parser.add_argument("--type", nargs="*", type=int, help="指定类别编号（不指定则处理全部）")
    args = parser.parse_args()

    checkpoint_dir = os.path.join(args.results_dir, args.checkpoint)
    if not os.path.isdir(checkpoint_dir):
        print(f"Error: directory not found: {checkpoint_dir}")
        sys.exit(1)

    selected = args.type if args.type else list(CHECKS.keys())
    cases = get_all_cases(checkpoint_dir)

    total_cases = 0
    total_deleted = 0

    for t in selected:
        if t not in CHECKS:
            print(f"[警告] 未知类别 {t}，跳过")
            continue
        desc, fn = CHECKS[t]
        hits = [os.path.basename(c) for c in cases if fn(c)]
        action = CATEGORY_DELETE_ACTION.get(t, "dir")

        print(f"\n=== 类别 {t}: {desc} ===")
        print(f"匹配数量: {len(hits)}")

        count_deleted = 0
        for case_id in hits:
            n = delete_case(checkpoint_dir, case_id, action)
            if n > 0:
                print(f"  已清理: {case_id}")
            count_deleted += n

        total_cases += len(hits)
        total_deleted += count_deleted
        print(f"本类别共清理: {count_deleted}")

    print(f"\n=== 汇总 ===")
    print(f"处理 case 数: {total_cases}")
    print(f"清理文件/目录数: {total_deleted}")


if __name__ == "__main__":
    main()

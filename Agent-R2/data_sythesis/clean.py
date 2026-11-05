"""
clean.py - 删除满足各类条件的 sample 对应文件
用法: python clean.py [--type 1 2 3 ...]
不指定 --type 则删除所有类别

删除规则:
- solver 类条件: 删除 solver 三件套
    conversation/solver_agent.json
    solver_result.json
    solver_tmp/  (整个目录)
- questioner 类条件: 删除 questioner + solver 六件套
    conversation/question_agent.json
    question_result.json
    question_tmp/
    + 上面 solver 三件套（如果存在）
"""
import os
import shutil
import argparse
from show import CHECKS, get_all_samples


def question_files(sample_dir):
    return {
        "agent": os.path.join(sample_dir, "conversation", "question_agent.json"),
        "result": os.path.join(sample_dir, "question_result.json"),
        "tmp": os.path.join(sample_dir, "question_tmp"),
    }


def remove_if_exists(path):
    if os.path.isfile(path):
        os.remove(path)
        return 1
    if os.path.isdir(path):
        shutil.rmtree(path)
        return 1
    return 0


def delete_solver(sample_dir):
    removed = 0
    removed += remove_if_exists(os.path.join(sample_dir, "conversation", "qwen_solver_agent.json"))
    removed += remove_if_exists(os.path.join(sample_dir, "qwen_solver_result.json"))
    removed += remove_if_exists(os.path.join(sample_dir, "qwen_solver_tmp"))
    return removed


def delete_questioner_and_solver(sample_dir):
    removed = 0
    q = question_files(sample_dir)
    removed += remove_if_exists(q["agent"])
    removed += remove_if_exists(q["result"])
    removed += remove_if_exists(q["tmp"])
    removed += delete_solver(sample_dir)
    return removed


def delete_sample(sample_dir):
    shutil.rmtree(sample_dir)
    return 1


# 每个类别对应的删除操作: "sample"=删整个目录, True=删questioner+solver, False=只删solver
CATEGORY_DELETE_QUESTIONER = {
    0: False,
    1: False,
    2: "sample",
    3: True,
    4: "sample",
    5: False,
}


def main():
    parser = argparse.ArgumentParser(description="清理各类待删除 sample 文件")
    parser.add_argument("--type", nargs="*", type=int, help="指定类别编号（不指定则处理全部）")
    args = parser.parse_args()

    selected = args.type if args.type else list(CHECKS.keys())
    samples = get_all_samples()

    total_samples = 0
    total_files = 0

    for t in selected:
        if t not in CHECKS:
            print(f"[警告] 未知类别 {t}，跳过")
            continue
        desc, fn = CHECKS[t]
        hits = [s for s in samples if fn(s)]
        delete_q = CATEGORY_DELETE_QUESTIONER.get(t, False)

        print(f"\n=== 类别 {t}: {desc} ===")
        print(f"匹配数量: {len(hits)}")

        count_files = 0
        for sample_dir in hits:
            if delete_q == "sample":
                n = delete_sample(sample_dir)
            elif delete_q:
                n = delete_questioner_and_solver(sample_dir)
            else:
                n = delete_solver(sample_dir)
            if n > 0:
                print(f"  已清理: {os.path.basename(sample_dir)} ({n} 项)")
            count_files += n

        total_samples += len(hits)
        total_files += count_files
        print(f"本类别共清理文件/目录: {count_files}")

    print(f"\n=== 汇总 ===")
    print(f"处理 sample 数: {total_samples}")
    print(f"清理文件/目录数: {total_files}")


if __name__ == "__main__":
    main()

"""
show.py - 展示满足各类条件的 sample 及数量
用法: python show.py [--type 1 2 3 ...]
"""
import os
import json
import glob
import argparse
import yaml
from urllib.parse import unquote

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
VALID_TOOLS = {"terminal", "web_search", "view_image_text", "web_fetch"}


def _load_solver_prefix() -> str:
    """从 config.yaml 读取 solver_agent family，返回文件前缀。"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        family = cfg.get("agents", {}).get("solver_agent", {}).get(
            "family", cfg.get("defaults", {}).get("family", "")
        )
        return family
    except Exception:
        return ""


SOLVER_PREFIX = _load_solver_prefix()


def get_all_samples():
    return sorted(glob.glob(os.path.join(BASE, "sample_*")))


def solver_files(sample_dir):
    result_name = f"{SOLVER_PREFIX}_solver_result.json" if SOLVER_PREFIX else "solver_result.json"
    tmp_name = f"{SOLVER_PREFIX}_solver_tmp" if SOLVER_PREFIX else "solver_tmp"
    agent_name = f"{SOLVER_PREFIX}_solver_agent.json" if SOLVER_PREFIX else "solver_agent.json"
    return {
        "agent": os.path.join(sample_dir, "conversation", agent_name),
        "result": os.path.join(sample_dir, result_name),
        "tmp_output": os.path.join(sample_dir, tmp_name, "output"),
    }


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# ── 各类检测函数 ────────────────────────────────────────────────────────────────

def check_type0(sample_dir):
    """qwen_solver_result.json 不存在（但 qwen_solver_agent.json 存在）"""
    agent = os.path.join(sample_dir, "conversation", "qwen_solver_agent.json")
    result = os.path.join(sample_dir, "qwen_solver_result.json")
    return os.path.exists(agent) and not os.path.exists(result)


def check_type1(sample_dir):
    """qwen_solver_result.json 存在，但 qwen_solver_tmp/output 目录为空"""
    result = os.path.join(sample_dir, "qwen_solver_result.json")
    if not os.path.exists(result):
        return False
    tmp_out = os.path.join(sample_dir, "qwen_solver_tmp", "output")
    if not os.path.isdir(tmp_out):
        return False
    return len(os.listdir(tmp_out)) == 0


def check_type2(sample_dir):
    """有 question_result.json，但 question_tmp/input 文件与 attachments 不一一对应"""
    qresult = os.path.join(sample_dir, "question_result.json")
    if not os.path.exists(qresult):
        return False
    data = load_json(qresult)
    if data is None:
        return False
    attachments = data.get("question", {}).get("attachments", [])
    expected = {a["filename"] for a in attachments}
    input_dir = os.path.join(sample_dir, "question_tmp", "input")
    if not os.path.isdir(input_dir):
        return bool(expected)
    actual = {unquote(f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))}
    return expected != actual


def check_type3(sample_dir):
    """没有 question_result.json，删除 question"""
    qresult = os.path.join(sample_dir, "question_result.json")
    qagent = os.path.join(sample_dir, "conversation", "question_agent.json")
    qtmp = os.path.join(sample_dir, "question_tmp")
    if os.path.exists(qresult):
        return False
    return os.path.exists(qagent) or os.path.isdir(qtmp)


def check_type4(sample_dir):
    """solver_agent 最后一条 assistant prompt_tokens+completion_tokens > 65536"""
    agent = solver_files(sample_dir)["agent"]
    if not os.path.exists(agent):
        return False
    data = load_json(agent)
    if not isinstance(data, list):
        return False
    for item in reversed(data):
        if item.get("role") == "assistant":
            usage = item.get("usage", {})
            if "prompt_tokens" in usage:
                return usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0) > 65536
    return False


def check_type5(sample_dir):
    """solver_agent 调用 web_search 超过 5 次，删除 solver"""
    agent = solver_files(sample_dir)["agent"]
    if not os.path.exists(agent):
        return False
    data = load_json(agent)
    if not isinstance(data, list):
        return False
    count = sum(
        1 for item in data
        if item.get("role") == "tool" and "search_results" in str(item.get("content", ""))
    )
    return count > 5


# ── 注册所有类别 ─────────────────────────────────────────────────────────────────

CHECKS = {
    0: ("qwen_solver_result.json 不存在（qwen_solver_agent.json 存在），删除 solver", check_type0),
    1: ("qwen_solver_result.json 存在但 qwen_solver_tmp/output 为空，删除 solver", check_type1),
    2: ("question_result.json 存在但 question_tmp/input 文件与 attachments 不一一对应，删除整个 sample", check_type2),
    3: ("没有 question_result.json，删除 questioner", check_type3),
    4: ("solver_agent token 超过 65536，删除整个 sample", check_type4),
    5: ("solver_agent 调用 web_search 超过 5 次，删除 solver", check_type5),
}


def main():
    parser = argparse.ArgumentParser(description="展示各类待清理 sample")
    parser.add_argument("--type", nargs="*", type=int, help="指定类别编号（不指定则展示全部）")
    parser.add_argument("--count_only", action="store_true", help="只打印数量，不打印每个 sample 名")
    args = parser.parse_args()

    selected = args.type if args.type else list(CHECKS.keys())
    samples = get_all_samples()

    for t in selected:
        if t not in CHECKS:
            print(f"[警告] 未知类别 {t}，跳过")
            continue
        desc, fn = CHECKS[t]
        hits = [os.path.basename(s) for s in samples if fn(s)]
        print(f"\n=== 类别 {t}: {desc} ===")
        print(f"数量: {len(hits)}")
        if not args.count_only:
            for name in hits:
                print(f"  {name}")


if __name__ == "__main__":
    main()

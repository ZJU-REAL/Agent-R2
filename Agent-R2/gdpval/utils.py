"""gdpval utils.py -- 工具函数：数据加载、结果保存、对话记录。"""
from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


def parse_json_response(content: str) -> Optional[Dict[str, Any]]:
    """从模型响应中解析JSON。"""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    patterns = [
        r'```json\s*\n(.*?)\n```',
        r'```\s*\n(.*?)\n```',
        r'\{[\s\S]*\}'
    ]
    for pattern in patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    try:
        start = content.find('{')
        if start != -1:
            depth = 0
            for i, char in enumerate(content[start:], start):
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        return json.loads(content[start:i+1])
    except json.JSONDecodeError:
        pass

    return None


# 项目根目录
GDPVAL_ROOT = Path(__file__).resolve().parent

# 线程锁
_lock = threading.Lock()


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """加载 YAML 配置文件。"""
    path = Path(config_path) if config_path else GDPVAL_ROOT / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_benchmark_data(
    data_path: Optional[str] = None,
    base_dir: Optional[str] = None,
    limit: int = -1,
) -> List[Dict[str, Any]]:
    """加载 benchmark JSON 数据，并将相对路径解析为绝对路径。

    Args:
        data_path: 数据文件路径，默认 data/train-00000-of-00001.json
        base_dir: reference_files / deliverable_files 的根目录
        limit: 最大加载数量，-1 表示全部

    Returns:
        任务列表，每个任务的 reference_files / deliverable_files 已转为本地绝对路径
    """
    if data_path is None:
        data_path = str(GDPVAL_ROOT / "data" / "train-00000-of-00001.json")
    if base_dir is None:
        base_dir = str(GDPVAL_ROOT)

    base = Path(base_dir)

    with open(data_path, "r", encoding="utf-8") as f:
        raw: List[Dict[str, Any]] = json.load(f)

    items: List[Dict[str, Any]] = []
    for i, item in enumerate(raw):
        if limit > 0 and i >= limit:
            break

        # 解析 reference_files 为本地绝对路径
        ref_paths = []
        for rel in item.get("reference_files", []):
            p = base / rel
            ref_paths.append(str(p))
        item["reference_files_local"] = ref_paths

        # 解析 deliverable_files 为本地绝对路径
        deliv_paths = []
        for rel in item.get("deliverable_files", []):
            p = base / rel
            deliv_paths.append(str(p))
        item["deliverable_files_local"] = deliv_paths

        # 解析 rubric_json（可能是字符串）
        rubric = item.get("rubric_json", "[]")
        if isinstance(rubric, str):
            rubric = json.loads(rubric)
        item["rubric"] = rubric

        items.append(item)

    return items


def save_result(result: Dict[str, Any], output_dir: str, task_id: str) -> Path:
    """保存单个任务的评估结果。"""
    d = Path(output_dir) / task_id
    d.mkdir(parents=True, exist_ok=True)
    path = d / "eval_result.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return path


def save_summary(results: List[Dict[str, Any]], output_dir: str) -> Path:
    """保存所有任务的汇总结果。"""
    d = Path(output_dir)
    d.mkdir(parents=True, exist_ok=True)

    num_tasks = len(results)

    # 原始总分（含负分项直接求和）
    total_score_raw = 0
    total_max_raw = 0
    sample_scores_raw = []
    for r in results:
        evals = r.get("evaluations", [])
        if not evals:
            s = r.get("score", 0)
            m = r.get("max_score", 0)
            total_score_raw += s
            total_max_raw += m
            sample_scores_raw.append(s / m if m > 0 else 0.0)
            continue
        s_raw = sum(e.get("score", 0) for e in evals)
        m_raw = sum(e.get("max_score", 0) for e in evals)
        total_score_raw += s_raw
        total_max_raw += m_raw
        sample_scores_raw.append(s_raw / m_raw if m_raw > 0 else 0.0)

    # 修正后总分（负分项转正）
    total_score = 0
    total_max = 0
    sample_scores = []
    for r in results:
        evals = r.get("evaluations", [])
        if not evals:
            s = r.get("score", 0)
            m = r.get("max_score", 0)
            total_score += s
            total_max += m
            sample_scores.append(s / m if m > 0 else 0.0)
            continue
        s_fixed, m_fixed = 0, 0
        for e in evals:
            ms = e.get("max_score", 0)
            sc = e.get("score", 0)
            if ms > 0:
                s_fixed += sc
                m_fixed += ms
            elif ms < 0:
                s_fixed += sc - ms  # score=0,max=-10 => +10; score=-10,max=-10 => +0
                m_fixed += -ms
        total_score += s_fixed
        total_max += m_fixed
        sample_scores.append(s_fixed / m_fixed if m_fixed > 0 else 0.0)

    # 方式1: 旧accuracy（原始负分项直接参与，样本等权）
    accuracy_raw = round(sum(sample_scores_raw) / len(sample_scores_raw), 4) if sample_scores_raw else 0.0
    # 方式2: 新accuracy（负分项转正后，样本等权）
    accuracy = round(sum(sample_scores) / len(sample_scores), 4) if sample_scores else 0.0
    # 方式3: total_score / total_max_score（修正后分数加权）
    accuracy_total = round(total_score / total_max, 4) if total_max > 0 else 0.0

    summary = {
        "num_tasks": num_tasks,
        "total_score": total_score,
        "total_max_score": total_max,
        "accuracy_raw": accuracy_raw,
        "accuracy": accuracy,
        "accuracy_total": accuracy_total,
        "results": results,
    }

    path = d / "summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return path


def save_conversation(
    messages: List[Dict[str, Any]],
    output_dir: str,
    task_id: str,
    phase: str,
) -> Path:
    """保存对话记录。phase = solver / evaluator。"""
    d = Path(output_dir) / task_id
    d.mkdir(parents=True, exist_ok=True)

    simplified = _simplify_messages(messages)
    path = d / f"{phase}_conversation.json"
    with _lock:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(simplified, f, ensure_ascii=False, indent=2)
    return path


def append_messages(
    conversation_dir: Path,
    phase: str,
    messages: List[Dict[str, Any]],
) -> None:
    """实时写入对话记录（覆盖模式）。

    Args:
        conversation_dir: conversation 文件夹路径
        phase: 阶段名称，如 solver / evaluator
        messages: 当前完整消息列表
    """
    conversation_dir = Path(conversation_dir)
    conversation_dir.mkdir(parents=True, exist_ok=True)
    simplified = _simplify_messages(messages)
    path = conversation_dir / f"{phase}_conversation.json"
    with _lock:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(simplified, f, ensure_ascii=False, indent=2)


def _simplify_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """简化消息，只保留 role 和 content，过滤 image_url 中的 base64。"""
    out = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            filtered = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    continue
                filtered.append(part)
            if len(filtered) == 1 and isinstance(filtered[0], dict) and filtered[0].get("type") == "text":
                content = filtered[0].get("text", "")
            else:
                content = filtered
        elif isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False)

        record: Dict[str, Any] = {"role": msg.get("role", ""), "content": content}
        if msg.get("reasoning_content"):
            record["reasoning_content"] = msg["reasoning_content"]
        if msg.get("tool_calls"):
            record["tool_calls"] = msg["tool_calls"]
        if msg.get("usage"):
            record["usage"] = msg["usage"]
        out.append(record)
    return out

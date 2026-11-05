"""sythesis_test utils.py -- 工具函数：数据加载、结果保存、对话记录。"""
from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
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


SYTHESIS_ROOT = Path(__file__).resolve().parent

# data_sythesis output 目录的本地路径，用于将其他机器上生成的 input_dir 映射到本机
_DATA_SYTHESIS_OUTPUT = Path(__file__).resolve().parent.parent / "data_sythesis" / "output"

_lock = threading.Lock()


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else SYTHESIS_ROOT / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_input_dir(input_dir_str: str) -> Path:
    """将 extra_info.input_dir 解析为本地路径。

    input_dir 由 prepare_data.py 生成，记录的是生成时的机器上的绝对路径。
    若路径在本机有效直接返回；否则提取 sample_xxx 之后的相对部分，
    拼接到本机的 _DATA_SYTHESIS_OUTPUT 下。
    """
    p = Path(input_dir_str)
    if p.exists():
        return p

    # 找到路径中 sample_xxx 段，取其后的相对路径拼接到本地 output 目录
    parts = p.parts
    for i, part in enumerate(parts):
        if part.startswith("sample_"):
            rel = Path(*parts[i:])
            return _DATA_SYTHESIS_OUTPUT / rel

    return p


def load_benchmark_data(
    data_path: Optional[str] = None,
    limit: int = -1,
) -> List[Dict[str, Any]]:
    """从 parquet 文件加载测试数据，转换为与 gdpval 兼容的任务列表格式。

    与 gdpval.utils.load_benchmark_data 的主要区别：
    - 数据源是 parquet 而非 JSON
    - prompt 已在 parquet 中预构建（system + user）
    - reference_files 来自 extra_info.input_dir（本地目录）
    - rubric 的 key 是 criterion/category/points/source（而非 rubric_item_id/score）
    """
    if data_path is None:
        data_path = str(SYTHESIS_ROOT / "data" / "test.parquet")

    df = pd.read_parquet(data_path)
    if limit > 0:
        df = df.iloc[:limit]

    items: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        prompt = row["prompt"]
        extra_info = row["extra_info"]
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)

        task_id = extra_info.get("task_id", "unknown")

        # 解析 rubric（JSON 字符串或列表）
        rubric_raw = extra_info.get("rubric", "[]")
        if isinstance(rubric_raw, str):
            rubric_raw = json.loads(rubric_raw)

        # 统一 rubric 格式：points -> score，补充 rubric_item_id
        # 部分行的 rubric 元素可能是字符串（数据生成时的序列化问题），尝试二次 parse
        rubric = []
        for i, item in enumerate(rubric_raw):
            if isinstance(item, str):
                try:
                    item = json.loads(item)
                except Exception:
                    item = {"criterion": item, "points": 0, "category": "", "source": ""}
            rubric.append({
                "rubric_item_id": f"{task_id}_r{i}",
                "criterion": item.get("criterion", ""),
                "score": item.get("points", 0),
                "category": item.get("category", ""),
                "source": item.get("source", ""),
            })

        # 解析 question
        question_raw = extra_info.get("question", "{}")
        if isinstance(question_raw, str):
            question_raw = json.loads(question_raw)

        # 解析 reference_files：从 input_dir 目录下读取所有文件
        input_dir_str = extra_info.get("input_dir", "")
        reference_files_local: List[str] = []
        if input_dir_str:
            local_input_dir = _resolve_input_dir(input_dir_str)
            if local_input_dir.is_dir():
                for f in sorted(local_input_dir.iterdir()):
                    if f.is_file():
                        reference_files_local.append(str(f))
            else:
                logger.warning(f"[{task_id}] input_dir 不存在: {local_input_dir}")

        # prompt 已预构建，numpy array -> list
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        prompt_list = list(prompt) if not isinstance(prompt, list) else prompt

        items.append({
            "task_id": task_id,
            "prompt": prompt_list,
            "rubric": rubric,
            "question": question_raw,
            "reference_files_local": reference_files_local,
            "deliverable_files_local": [],  # sythesis_test 无 ground truth 交付物
            "sector": extra_info.get("sector", ""),
            "occupation": extra_info.get("occupation", ""),
        })

    return items


def save_result(result: Dict[str, Any], output_dir: str, task_id: str) -> Path:
    d = Path(output_dir) / task_id
    d.mkdir(parents=True, exist_ok=True)
    path = d / "eval_result.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return path


def save_summary(results: List[Dict[str, Any]], output_dir: str) -> Path:
    d = Path(output_dir)
    d.mkdir(parents=True, exist_ok=True)

    total_score = sum(r.get("score", 0) for r in results)
    total_max = sum(r.get("max_score", 0) for r in results)

    sample_scores = [
        r.get("score", 0) / r.get("max_score", 1) if r.get("max_score", 0) > 0 else 0.0
        for r in results
    ]
    accuracy = round(sum(sample_scores) / len(sample_scores), 4) if sample_scores else 0.0

    summary = {
        "num_tasks": len(results),
        "total_score": total_score,
        "total_max_score": total_max,
        "accuracy": accuracy,
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
    conversation_dir = Path(conversation_dir)
    conversation_dir.mkdir(parents=True, exist_ok=True)
    simplified = _simplify_messages(messages)
    path = conversation_dir / f"{phase}_conversation.json"
    with _lock:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(simplified, f, ensure_ascii=False, indent=2)


def _simplify_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
        if msg.get("finish_reason"):
            record["finish_reason"] = msg["finish_reason"]
        if msg.get("_vllm_parser_fallback"):
            record["_vllm_parser_fallback"] = True
        out.append(record)
    return out

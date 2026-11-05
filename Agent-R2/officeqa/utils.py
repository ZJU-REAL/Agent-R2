"""officeqa utils.py -- 工具函数：数据加载、结果保存、对话记录。"""
from __future__ import annotations

import csv
import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

OFFICEQA_ROOT = Path(__file__).resolve().parent

_lock = threading.Lock()


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else OFFICEQA_ROOT / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_benchmark_data(
    data_path: str,
    docs_base: str,
    start: int = 0,
    end: int = -1,
) -> List[Dict[str, Any]]:
    """加载 officeqa_full.csv，解析文档文件路径。

    CSV columns: uid, question, answer, source_docs, source_files, difficulty
    source_files may be comma-separated (multiple files).
    Relative data_path / docs_base are resolved against the officeqa/ directory.
    """
    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = OFFICEQA_ROOT / data_path
    base = Path(docs_base)
    if not base.is_absolute():
        base = OFFICEQA_ROOT / base
    items: List[Dict[str, Any]] = []

    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    rows = rows[start:] if end < 0 else rows[start:end]

    for row in rows:
        uid = row["uid"]
        source_files_raw = row.get("source_files", "")
        # 支持逗号分隔多文件
        doc_names = [s.strip() for part in source_files_raw.splitlines() for s in part.split(",") if s.strip()]
        doc_local_paths = [str(base / name) for name in doc_names]

        items.append({
            "task_id": uid,
            "uid": uid,
            "question": row["question"],
            "answer": row["answer"],
            "source_docs": row.get("source_docs", ""),
            "source_files": doc_names,
            "doc_local_paths": doc_local_paths,
            "difficulty": row.get("difficulty", ""),
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

    num_tasks = len(results)
    acc_total = sum(r.get("accuracy", 0) for r in results)

    summary = {
        "num_tasks": num_tasks,
        "accuracy": round(acc_total / num_tasks, 4) if num_tasks > 0 else 0,
        "results": results,
    }

    path = d / "summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
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
            filtered = [p for p in content if not (isinstance(p, dict) and p.get("type") == "image_url")]
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

"""data-sythesis utils.py -- 工具函数。"""
from __future__ import annotations
import copy
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import threading

logger = logging.getLogger(__name__)


def parse_json_response(content: str) -> Optional[Dict[str, Any]]:
    """从模型响应中解析JSON。

    Args:
        content: 模型返回的内容

    Returns:
        解析后的字典，失败返回None
    """
    # 尝试直接解析
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 尝试提取JSON块
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

    # 最后尝试找到第一个完整的JSON对象
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
                        json_str = content[start:i+1]
                        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    return None


import pandas as pd

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

# 数据文件路径
SEED_TASK_DIR = PROJECT_ROOT / "seed_task"
SAMPLE_JSONL_PATH = SEED_TASK_DIR / "sample.jsonl"
TASK_STATEMENTS_PATH = SEED_TASK_DIR / "Task Statements.xlsx"
DISCIPLINE_TO_TITLE_PATH = SEED_TASK_DIR / "discipline_to_title.json"

# 线程锁，确保并发写入安全
_conversation_lock = threading.Lock()


def _get_content_hash(content: Any) -> str:
    """计算内容的哈希值，用于去重。"""
    import hashlib
    content_str = json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.md5(content_str.encode("utf-8")).hexdigest()


def append_conversation(
    conversation_dir: Path,
    phase: str,
    role: str,
    content: Any,
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """追加写入对话记录到 conversation 文件夹。

    Args:
        conversation_dir: conversation 文件夹路径
        phase: 阶段名称 (scenario_generation, file_search, question_design)
        role: 角色 (system, user, assistant, tool)
        content: 内容
        extra: 额外字段 (如 tool_calls, tool_call_id 等)
    """
    conversation_dir.mkdir(parents=True, exist_ok=True)
    conversation_file = conversation_dir / f"{phase}.json"

    # 构建记录：只保存 role 和 content
    record = {
        "role": role,
        "content": content,
    }

    with _conversation_lock:
        # 读取现有数据
        existing_records = []
        existing_hashes = set()
        if conversation_file.exists():
            try:
                with open(conversation_file, "r", encoding="utf-8") as f:
                    existing_records = json.load(f)
                    # 计算现有记录的哈希
                    for r in existing_records:
                        h = _get_content_hash(r)
                        existing_hashes.add(h)
            except (json.JSONDecodeError, Exception):
                existing_records = []

        # 检查是否已存在
        new_hash = _get_content_hash(record)
        if new_hash not in existing_hashes:
            existing_records.append(record)
            # 保存为 JSON，indent=4
            with open(conversation_file, "w", encoding="utf-8") as f:
                json.dump(existing_records, f, ensure_ascii=False, indent=4)


def append_messages(
    conversation_dir: Path,
    phase: str,
    messages: List[Dict[str, Any]]
) -> None:
    """写入对话记录，直接覆盖模式。

    简化消息：只提取 role 和 content。
    对于 view_image 工具，content 中的 image_url 不保存。

    Args:
        conversation_dir: conversation 文件夹路径
        phase: 阶段名称
        messages: 消息列表
    """
    conversation_dir.mkdir(parents=True, exist_ok=True)
    conversation_file = conversation_dir / f"{phase}.json"

    # 简化 messages：只提取 role 和 content
    simplified_messages = []
    for msg in messages:
        content = msg.get("content", "")

        # 如果 content 是列表（可能是 view_image 返回的），处理 image_url
        image_source_paths = []
        if isinstance(content, list):
            # 过滤掉 image_url 类型的元素，只保留 text，但保存 image_source_path
            filtered_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    src = part.get("image_source_path")
                    if src:
                        image_source_paths.append(src)
                    continue
                filtered_parts.append(part)
            # 如果过滤后只有一个 text 元素，直接转为字符串
            if len(filtered_parts) == 1 and isinstance(filtered_parts[0], dict) and filtered_parts[0].get("type") == "text":
                content = filtered_parts[0].get("text", "")
            else:
                content = filtered_parts
        elif isinstance(content, dict):
            # 如果是 dict，转为字符串
            content = json.dumps(content, ensure_ascii=False)

        record = {
            "role": msg.get("role", ""),
            "content": content,
        }
        # 如果有 reasoning_content 也保存
        if msg.get("reasoning_content"):
            record["reasoning_content"] = msg["reasoning_content"]
        # 如果有 tool_calls 也保存
        if msg.get("tool_calls"):
            record["tool_calls"] = msg["tool_calls"]
        # 如果有 image_source_paths 也保存（从被过滤的 image_url 中提取）
        if image_source_paths:
            record["image_source_paths"] = image_source_paths
        # 如果有 usage 也保存（兼容 Claude 和 Gemini 格式）
        if msg.get("usage"):
            u = msg["usage"]
            gemini_keys = ("promptTokenCount", "candidatesTokenCount", "totalTokenCount")
            claude_keys = ("prompt_tokens", "completion_tokens", "total_tokens")
            saved = {k: u[k] for k in (*gemini_keys, *claude_keys) if k in u}
            if saved:
                record["usage"] = saved
        simplified_messages.append(record)

    # 直接覆盖写入
    with open(conversation_file, "w", encoding="utf-8") as f:
        json.dump(simplified_messages, f, ensure_ascii=False, indent=4)


def load_sample_data(begin: Optional[int] = None, end: Optional[int] = None) -> List[Dict[str, Any]]:
    """加载 sample.jsonl 数据。

    Args:
        begin: 起始索引（包含），None 表示从0开始
        end: 结束索引（不包含），None 表示加载全部

    Returns:
        数据列表，每项包含 discipline 和 original_document 字段
    """
    data = []
    with open(SAMPLE_JSONL_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if begin is not None and i < begin:
                continue
            if end is not None and i >= end:
                break
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_task_statements() -> pd.DataFrame:
    """加载 Task Statements.xlsx 数据。

    Returns:
        DataFrame，包含 Title 和 Task 字段
    """
    return pd.read_excel(TASK_STATEMENTS_PATH)


def load_discipline_to_title() -> Dict[str, str]:
    """加载 discipline 到 Title 的映射。

    Returns:
        映射字典 {discipline: title}
    """
    with open(DISCIPLINE_TO_TITLE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_tasks_for_title(df: pd.DataFrame, title: str) -> List[str]:
    """获取指定 Title 的所有任务描述。

    Args:
        df: Task Statements DataFrame
        title: 职业名称

    Returns:
        任务描述列表
    """
    filtered = df[df["Title"] == title]
    return filtered["Task"].tolist()


def build_title_to_tasks_mapping(df: pd.DataFrame) -> Dict[str, List[str]]:
    """构建 Title 到任务列表的映射。

    Args:
        df: Task Statements DataFrame

    Returns:
        映射字典 {title: [task1, task2, ...]}
    """
    mapping = {}
    for title, group in df.groupby("Title"):
        mapping[title] = group["Task"].tolist()
    return mapping


def get_discipline_for_sample(sample: Dict[str, Any]) -> str:
    """获取样本的 discipline 字段。"""
    return sample.get("discipline", "")


def get_original_document(sample: Dict[str, Any]) -> str:
    """获取样本的 original_document 字段。"""
    return sample.get("original_document", "")


def map_discipline_to_title(discipline: str, mapping: Dict[str, str]) -> Optional[str]:
    """将 discipline 映射到对应的 Title。

    Args:
        discipline: 学科名称
        mapping: discipline 到 Title 的映射

    Returns:
        对应的 Title，如果不存在则返回 None
    """
    return mapping.get(discipline)


def save_output(data: Dict[str, Any], output_path: Path) -> None:
    """保存输出数据到 JSON 文件。

    Args:
        data: 要保存的数据
        output_path: 输出文件路径
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_output(output_path: Path) -> Optional[Dict[str, Any]]:
    """加载输出数据。

    Args:
        output_path: 输出文件路径

    Returns:
        加载的数据，如果文件不存在则返回 None
    """
    if not output_path.exists():
        return None
    with open(output_path, "r", encoding="utf-8") as f:
        return json.load(f)
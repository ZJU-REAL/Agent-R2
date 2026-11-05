"""data-sythesis prompts/retry_prompt.py -- 重试提示词。"""
from __future__ import annotations
from typing import List


def build_json_retry_message() -> str:
    """构建JSON解析失败时的重试提示。

    Returns:
        重试提示字符串
    """
    return """无法解析你返回的JSON格式,检查有没有转义错误,重新输出有效的JSON"""


def build_file_not_found_message(missing_files: List[str], available_files: List[str]) -> str:
    """构建文件不存在时的重试提示。

    Args:
        missing_files: 缺失的文件名列表
        available_files: 可用的文件名列表

    Returns:
        重试提示字符串
    """
    missing_str = ", ".join(missing_files) if missing_files else "无"
    return f"""你返回的JSON中以下文件不存在：
{missing_str},请重新输出"""
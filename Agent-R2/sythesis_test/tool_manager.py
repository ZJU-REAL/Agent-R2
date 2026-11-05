"""sythesis_test tool_manager.py -- 工具注册、执行分发、消息构建。

Solver 和 Evaluator 共用同一套工具（terminal、view_image_text、web_fetch、web_search）。
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# 添加项目根目录到路径
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tools.terminal import TERMINAL_TOOL, execute_terminal
from tools.view_image_text import VIEW_IMAGE_TEXT_TOOL, view_image_text
from tools.web_fetch import WEB_FETCH_TOOL, web_fetch
from tools.web_search import WEB_SEARCH_TOOL, web_search

logger = logging.getLogger(__name__)

_RUN_LOG = Path(__file__).resolve().parent / "run.log"


def _log_tool_error(tool_name: str, error_msg: str) -> None:
    """将工具调用错误以 - ERROR - 前缀追加写入 run.log。"""
    line = f"- ERROR - tool={tool_name} {error_msg}\n"
    try:
        with open(_RUN_LOG, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass
    logger.error("- ERROR - tool=%s %s", tool_name, error_msg)


def get_solver_tools() -> List[Dict[str, Any]]:
    """获取 Solver Agent 可用的工具定义。"""
    return [
        TERMINAL_TOOL,
        VIEW_IMAGE_TEXT_TOOL,
        WEB_FETCH_TOOL,
        WEB_SEARCH_TOOL,
    ]


def get_evaluator_tools() -> List[Dict[str, Any]]:
    """获取 Evaluator Agent 可用的工具定义。"""
    return [
        TERMINAL_TOOL,
        VIEW_IMAGE_TEXT_TOOL,
    ]


async def execute_tool(
    tool_name: str,
    tool_args: Dict[str, Any],
    *,
    output_tmp_dir: Optional[str] = None,
    session_id: Optional[str] = None,
    env: Any = None,
) -> Dict[str, Any]:
    """执行指定工具并返回结果。"""
    try:
        if not tool_args:
            msg = "tool_args is empty (JSON parse failed or model passed no arguments)."
            _log_tool_error(tool_name, msg)
            return {"- ERROR -": msg}

        if tool_name == "terminal":
            result = await execute_terminal(tool_args, session_id=session_id, env=env)
            if result.get("error"):
                _log_tool_error(tool_name, result['error'])
            return result

        if tool_name == "view_image_text":
            image_path = tool_args.get("image_path", "")
            query = tool_args.get("query", "")
            if not image_path or not query:
                msg = "Both 'query' and 'image_path' are required."
                _log_tool_error(tool_name, msg)
                return {"- ERROR -": msg}
            from sythesis_test.sandbox.file_management import download_file_from_sandbox
            try:
                filepath = await download_file_from_sandbox(env, session_id, image_path, output_tmp_dir)
            except Exception as e:
                err_str = str(e)
                import re as _re
                m = _re.search(r'"message\\?":\s*\\"([^"\\]+)', err_str) or _re.search(r'"message":\s*"([^"]+)"', err_str)
                msg = f"Failed to download image: {m.group(1) if m else err_str}"
                _log_tool_error(tool_name, msg)
                return {"- ERROR -": msg}
            result = await view_image_text(query=query, image_path=str(filepath))
            if "error" in result:
                _log_tool_error(tool_name, result["error"])
            return result

        if tool_name == "web_fetch":
            result = await web_fetch(url=tool_args.get("url", ""))
            if result.get("error"):
                _log_tool_error(tool_name, result["error"])
            return result

        if tool_name == "web_search":
            result = await web_search(query=tool_args.get("query", ""))
            if result.get("error"):
                _log_tool_error(tool_name, result["error"])
            return result

        msg = f"Unknown tool: {tool_name}"
        _log_tool_error(tool_name, msg)
        return {"- ERROR -": msg}

    except Exception as e:
        err_str = str(e)
        _log_tool_error(tool_name, err_str)
        return {"- ERROR -": f"Tool {tool_name} failed: {err_str}"}


def build_tool_message_content(tool_name: str, tool_result: Any, max_length: int = 0) -> Union[str, List[Dict[str, Any]]]:
    """构造 tool message content。view_image 返回多模态列表，其他返回字符串。"""
    if tool_name == "view_image" and isinstance(tool_result, dict) and tool_result.get("type") == "image":
        images = tool_result.get("images", [])
        errors = tool_result.get("errors", [])
        loaded = [img["filename"] for img in images]
        text = f"Image(s) loaded: {', '.join(loaded)}" if loaded else "No images."
        if errors:
            text += f"\nErrors: {'; '.join(errors)}"
        resized = sum(1 for img in images if img.get("resized"))
        if resized:
            text += f"\n[{resized} image(s) resized to 1800x1800 max]"
        parts: List[Dict[str, Any]] = [{"type": "text", "text": text}]
        for img in images:
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{img['mime_type']};base64,{img['base64_data']}"},
                "image_source_path": img["filepath"]
            })
        return parts if len(parts) > 1 else text

    if isinstance(tool_result, (dict, list)):
        text = json.dumps(tool_result, ensure_ascii=False, indent=2)
    else:
        text = str(tool_result)
    if max_length and len(text) > max_length:
        length = max_length // 2
        text = text[:length] + "...(truncated)..." + text[-length:]
    return text
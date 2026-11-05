"""gdpval tool_manager.py -- 工具注册、执行分发、消息构建。

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
from tools.view_image_multimodal import VIEW_IMAGE_TOOL as VIEW_IMAGE_MULTIMODAL_TOOL, view_image as view_image_multimodal
from tools.view_image_text import VIEW_IMAGE_TEXT_TOOL, view_image_text
from tools.web_fetch import WEB_FETCH_TOOL, web_fetch
from tools.web_search import WEB_SEARCH_TOOL, web_search
from settings import load_settings

logger = logging.getLogger(__name__)

_RUN_LOG = Path(__file__).resolve().parent / "run.log"
_GDPVAL_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def _load_gdpval_config() -> Dict[str, Any]:
    import yaml
    try:
        with open(_GDPVAL_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _log_tool_error(tool_name: str, error_msg: str) -> None:
    """将工具调用错误以 - ERROR - 前缀追加写入 run.log。"""
    line = f"- ERROR - tool={tool_name} {error_msg}\n"
    try:
        with open(_RUN_LOG, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass
    logger.error("- Tool_call_error - tool=%s %s", tool_name, error_msg)


def _is_solver_external_vision() -> bool:
    cfg = _load_gdpval_config()
    return cfg.get("solver", {}).get("external_vision", False)


def _is_evaluator_external_vision() -> bool:
    cfg = _load_gdpval_config()
    return cfg.get("evaluator", {}).get("external_vision", False)


def get_solver_tools() -> List[Dict[str, Any]]:
    """获取 Solver Agent 可用的工具定义。

    external_vision=false: view_image_multimodal（直接返回图片 base64 给有多模态能力的模型）
    external_vision=true:  view_image_text（调用 vision_model 将图片转为文字描述）
    """
    if _is_solver_external_vision():
        return [
            TERMINAL_TOOL,
            VIEW_IMAGE_TEXT_TOOL,
            WEB_FETCH_TOOL,
            WEB_SEARCH_TOOL,
        ]
    return [
        TERMINAL_TOOL,
        VIEW_IMAGE_MULTIMODAL_TOOL,
        WEB_FETCH_TOOL,
        WEB_SEARCH_TOOL,
    ]


def get_evaluator_tools() -> List[Dict[str, Any]]:
    """获取 Evaluator Agent 可用的工具定义。"""
    if _is_evaluator_external_vision():
        return [
            TERMINAL_TOOL,
            VIEW_IMAGE_TEXT_TOOL,
        ]
    return [
        TERMINAL_TOOL,
        VIEW_IMAGE_MULTIMODAL_TOOL,
    ]


def _get_vision_api_url() -> str:
    """根据 vision_model.ports 拼接多端点 URL，为空则 fallback 到 settings 的 llm_api_url。"""
    cfg = _load_gdpval_config()
    vision = cfg.get("vision_model", {})
    ports = vision.get("ports") or []
    if ports:
        return ",".join(f"http://localhost:{p}/v1" for p in ports)
    return load_settings()["llm_api_url"]


def _get_vision_kwargs() -> Dict[str, Any]:
    cfg = _load_gdpval_config()
    vision = cfg.get("vision_model", {})
    return {
        "family": vision.get("family"),
        "model_id": vision.get("model_id"),
        "sk": load_settings()["llm_api_key"],
        "api_url": _get_vision_api_url(),
    }


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
            msg = f"Failed to parse arguments for tool {tool_name}. Please provide valid JSON arguments."
            _log_tool_error(tool_name, msg)
            return {"error": msg}

        if tool_name == "terminal":
            result = await execute_terminal(tool_args, session_id=session_id, env=env)
            if result.get("error"):
                _log_tool_error(tool_name, result['error'])
            return result

        if tool_name == "view_image":
            # view_image_multimodal：直接返回图片 base64，供有多模态能力的模型使用
            filename = tool_args.get("filename", "")
            result = await view_image_multimodal(
                filename,
                session_id=session_id,
                env=env,
                output_tmp_dir=output_tmp_dir,
            )
            if result.get("error"):
                _log_tool_error(tool_name, result["error"])
            return result

        if tool_name == "view_image_text":
            image_path = tool_args.get("image_path", "")
            query = tool_args.get("query", "")
            if not image_path.lower().endswith((".png", ".jpeg", ".jpg")):
                msg = f"Unsupported image format: '{image_path}'. Only .png, .jpeg, .jpg are supported."
                _log_tool_error(tool_name, msg)
                return {"error": msg}
            from gdpval.sandbox.file_management import download_file_from_sandbox
            try:
                check_result = await env.session_exec(
                    session_id=session_id,
                    command=f"find \"{image_path}\" -type f 2>/dev/null",
                    exec_dir="/",
                    combine_output=True,
                )
                found = check_result.get("output", "").strip()
                if not found:
                    msg = f"Image file not found in sandbox: {image_path}"
                    _log_tool_error(tool_name, msg)
                    return {"error": msg}
                filepath = await download_file_from_sandbox(env, session_id, image_path, output_tmp_dir)
            except Exception as e:
                err_str = str(e)
                import re as _re
                m = _re.search(r"'exceptions': \[\"([^\"]+)\"", err_str)
                msg = f"Failed to download image: {m.group(1) if m else err_str}"
                _log_tool_error(tool_name, msg)
                return {"error": msg}
            result = await view_image_text(
                query=query,
                image_path=str(filepath),
                **_get_vision_kwargs(),
            )
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
        return {"error": msg}

    except Exception as e:
        err_str = str(e)
        _log_tool_error(tool_name, err_str)
        return {"error": f"Tool {tool_name} failed: {err_str}"}


def build_tool_message_content(tool_name: str, tool_result: Any, max_length: int = 0) -> Union[str, List[Dict[str, Any]]]:
    """构造 tool message content。view_image 返回多模态列表，其他返回字符串。"""
    if tool_name == "view_image" and isinstance(tool_result, dict) and tool_result.get("type") == "image":
        text = f"Image loaded: {tool_result['filename']}"
        if tool_result.get("resized"):
            text += " [resized to 1800x1800 max]"
        parts: List[Dict[str, Any]] = [{"type": "text", "text": text}, {
            "type": "image_url",
            "image_url": {"url": f"data:{tool_result['mime_type']};base64,{tool_result['base64_data']}"},
            "image_source_path": tool_result["filepath"],
        }]
        return parts

    if isinstance(tool_result, (dict, list)):
        text = json.dumps(tool_result, ensure_ascii=False, indent=2)
    else:
        text = str(tool_result)
    if max_length and len(text) > max_length:
        length = max_length // 2
        text = text[:length] + "...(truncated)..." + text[-length:]
    return text
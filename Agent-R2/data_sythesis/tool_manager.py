"""data-sythesis tool_manager.py -- 工具注册、执行分发、消息构建。

提供文件搜索工具: web_search_pptx, web_search_xlsx, web_search_pdf, web_search_image
以及辅助工具: terminal, finish, view_image
问题设计工具: terminal, view_image, web_fetch, web_search
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
from tools.web_search_pptx import WEB_SEARCH_PPTX_TOOL, web_search_pptx
from tools.web_search_xlsx import WEB_SEARCH_XLSX_TOOL, web_search_xlsx
from tools.web_search_pdf import WEB_SEARCH_PDF_TOOL, web_search_pdf
from tools.web_search_docx import WEB_SEARCH_DOCX_TOOL, web_search_docx
from tools.web_search_image import WEB_SEARCH_IMAGE_TOOL, web_search_image
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
    logger.error("- Tool_call_error - tool=%s %s", tool_name, error_msg)



def get_search_tools() -> List[Dict[str, Any]]:
    """获取所有可用工具定义。"""
    return [
        TERMINAL_TOOL,
        VIEW_IMAGE_MULTIMODAL_TOOL,
        WEB_SEARCH_PPTX_TOOL,
        WEB_SEARCH_XLSX_TOOL,
        WEB_SEARCH_PDF_TOOL,
        WEB_SEARCH_DOCX_TOOL,
        WEB_SEARCH_IMAGE_TOOL,
    ]


def get_question_tools() -> List[Dict[str, Any]]:
    """获取问题设计Agent可用的工具定义。"""
    return [
        TERMINAL_TOOL,
        VIEW_IMAGE_MULTIMODAL_TOOL,
        WEB_FETCH_TOOL,
        WEB_SEARCH_TOOL,
    ]


def get_solver_tools() -> List[Dict[str, Any]]:
    """获取解题Agent可用的工具定义。使用 view_image_text 代替 view_image，
    因为微调目标是纯文本模型，轨迹中不应包含多模态内容。"""
    return [
        TERMINAL_TOOL,
        VIEW_IMAGE_TEXT_TOOL,
        WEB_FETCH_TOOL,
        WEB_SEARCH_TOOL,
    ]


async def execute_tool(
    tool_name: str,
    tool_args: Dict[str, Any],
    *,
    output_tmp_dir: Optional[str] = None,
    session_id: Optional[str] = None,
    env: Any = None,
    num: Optional[int] = None,
    search_mode: str = "async",
    size_limit: int = 1,
    page_limit: int = 40,
    top: int = 5
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
            from data_sythesis.sandbox.file_management import download_file_from_sandbox
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
            result = await view_image_text(query=query, image_path=str(filepath))
            if "error" in result:
                _log_tool_error(tool_name, result["error"])
            return result

        if tool_name == "web_search_pptx":
            return await web_search_pptx(
                query=tool_args.get("query", ""),
                sandbox_dir=tool_args.get("sandbox_dir", "/home/workspace/search_files").rstrip("/"),
                session_id=session_id,
                env=env,
                output_tmp_dir=output_tmp_dir,
                num=num,
                search_mode=search_mode,
                page_limit=page_limit,
                size_limit=size_limit,
                top=top,
            )

        if tool_name == "web_search_xlsx":
            return await web_search_xlsx(
                query=tool_args.get("query", ""),
                sandbox_dir=tool_args.get("sandbox_dir", "/home/workspace/search_files").rstrip("/"),
                session_id=session_id,
                env=env,
                output_tmp_dir=output_tmp_dir,
                num=num,
                search_mode=search_mode,
                size_limit=size_limit,
                top=top,
            )

        if tool_name == "web_search_pdf":
            return await web_search_pdf(
                query=tool_args.get("query", ""),
                sandbox_dir=tool_args.get("sandbox_dir", "/home/workspace/search_files").rstrip("/"),
                session_id=session_id,
                env=env,
                output_tmp_dir=output_tmp_dir,
                num=num,
                search_mode=search_mode,
                page_limit=page_limit,
                size_limit=size_limit,
                top=top,
            )

        if tool_name == "web_search_docx":
            return await web_search_docx(
                query=tool_args.get("query", ""),
                sandbox_dir=tool_args.get("sandbox_dir", "/home/workspace/search_files").rstrip("/"),
                session_id=session_id,
                env=env,
                output_tmp_dir=output_tmp_dir,
                num=num,
                search_mode=search_mode,
                page_limit=page_limit,
                size_limit=size_limit,
                top=top,
            )

        if tool_name == "web_search_image":
            return await web_search_image(
                query=tool_args.get("query", ""),
                sandbox_dir=tool_args.get("sandbox_dir", "/home/workspace/search_files").rstrip("/"),
                session_id=session_id,
                env=env,
                output_tmp_dir=output_tmp_dir,
                num=num,
                search_mode=search_mode,
                top=top,
            )

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
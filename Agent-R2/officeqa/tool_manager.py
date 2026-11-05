"""officeqa tool_manager.py -- 工具注册、执行分发、消息构建。"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tools.terminal import TERMINAL_TOOL, execute_terminal

logger = logging.getLogger(__name__)

_RUN_LOG = Path(__file__).resolve().parent / "run.log"


def _log_tool_error(tool_name: str, error_msg: str) -> None:
    line = f"- ERROR - tool={tool_name} {error_msg}\n"
    try:
        with open(_RUN_LOG, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass
    logger.error("- Tool_call_error - tool=%s %s", tool_name, error_msg)


def get_solver_tools() -> List[Dict[str, Any]]:
    return [TERMINAL_TOOL]


async def execute_tool(
    tool_name: str,
    tool_args: Dict[str, Any],
    *,
    session_id: str = None,
    env: Any = None,
) -> Dict[str, Any]:
    try:
        if not tool_args:
            msg = f"Failed to parse arguments for tool {tool_name}. Please provide valid JSON arguments."
            _log_tool_error(tool_name, msg)
            return {"error": msg}

        if tool_name == "terminal":
            try:
                result = await execute_terminal(tool_args, session_id=session_id, env=env)
            except Exception as e:
                err_str = str(e)
                if "Failed to exec command" in err_str or "33043" in err_str:
                    result = {
                        "error": (
                            "Command too long. Write your script to /home/workspace/solution.py "
                            "in small chunks (≤30 lines per call), then run it with "
                            "`python3 /home/workspace/solution.py`."
                        )
                    }
                else:
                    result = {"error": f"Terminal failed: {err_str}"}
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
    if isinstance(tool_result, (dict, list)):
        text = json.dumps(tool_result, ensure_ascii=False, indent=2)
    else:
        text = str(tool_result)
    if max_length and len(text) > max_length:
        length = max_length // 2
        text = text[:length] + "...(truncated)..." + text[-length:]
    return text

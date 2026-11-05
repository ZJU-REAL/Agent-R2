"""terminal 工具 -- 在沙盒内执行 shell 命令。"""
from __future__ import annotations
import asyncio
import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

TERMINAL_TOOL = {
    "type": "function",
    "function": {
        "name": "terminal",
        "description": "Executes a given bash command in a persistent shell session.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command to execute."},
            },
            "required": ["command"],
        },
    },
}


async def execute_terminal(
    tool_args: Dict[str, Any],
    *,
    session_id: Optional[str],
    env: Any,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    if session_id is None or env is None:
        return {"error": "terminal requires session_id and env."}
    command = tool_args.get("command", "")
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            result = await env.session_exec(
                session_id=session_id,
                command=command,
                combine_output=True,
            )
            last_exc = None
            if attempt > 0:
                logger.info("session_exec succeeded on attempt %d", attempt + 1)
            break
        except Exception as e:
            logger.warning("session_exec attempt %d/3 failed: %s", attempt + 1, e)
            last_exc = e
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
    if last_exc is not None:
        logger.error("session_exec all 3 attempts failed, last error: %s", last_exc)
        error_msg = str(last_exc)
        m = re.search(r"'exceptions': \[\"([^\"]+)\"", error_msg)
        if m:
            error_msg = m.group(1)
        return {"error": f"terminal exec failed: {error_msg}"}

    output = result.get("output", "") if isinstance(result, dict) else str(result)
    return_code = result.get("return_code", 0) if isinstance(result, dict) else 0

    if return_code != 0:
        return {"error": output}
    return {"status": "success", "output": output}

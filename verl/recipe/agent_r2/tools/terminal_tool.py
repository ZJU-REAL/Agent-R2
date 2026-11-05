"""terminal_tool.py -- verl BaseTool wrapper for the sandbox terminal tool."""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from ..sandbox.file_management import upload_files_to_sandbox

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

MAX_OUTPUT_LENGTH = 4096

_TERMINAL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "terminal",
        "description": (
            "Executes a given bash command in a persistent shell session inside the sandbox. "
            "Use this tool to run scripts, inspect files, install packages, etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute.",
                }
            },
            "required": ["command"],
        },
    },
}


def _truncate_output(text: str, max_len: int = MAX_OUTPUT_LENGTH) -> str:
    if len(text) <= max_len:
        return text
    half = max_len // 2
    return text[:half] + f"\n... [truncated {len(text) - max_len} chars] ...\n" + text[-half:]


class SandboxTerminalTool(BaseTool):
    """verl BaseTool that executes bash commands in a sandbox session.

    The session is created fresh per trajectory by RealAgentLoop and passed
    via create(session=...).  Session lifecycle (create/destroy) is managed
    entirely by RealAgentLoop — this tool only holds a reference.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._sessions: dict[str, Any] = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        session=None,
        input_dir: str = "",
        **kwargs,
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())

        if instance_id in self._sessions:
            return instance_id, ToolResponse()

        if session is None:
            raise RuntimeError("SandboxTerminalTool.create() requires a session parameter")

        self._sessions[instance_id] = session

        # Clean workspace at the start of every trajectory
        await session.exec(
            "rm -rf /mnt/user-data/uploads/* /mnt/user-data/outputs/* /home/workspace/* /tmp/* 2>/dev/null"
            "; mkdir -p /mnt/user-data/uploads /mnt/user-data/outputs /home/workspace"
        )

        # Upload input attachments if provided
        if input_dir and not Path(input_dir).exists():
            logger.warning(f"[SandboxTerminalTool] input_dir not found: {input_dir}")
        elif input_dir:
            input_path = Path(input_dir)
            local_files = [f for f in input_path.iterdir() if f.is_file()]
            logger.warning(f"[SandboxTerminalTool] upload start: {input_dir} -> /mnt/user-data/uploads, files={[f.name for f in local_files]}")
            if local_files:
                try:
                    remote_paths = await upload_files_to_sandbox(
                        env=session.env,
                        session_id=session.session_id,
                        local_files=local_files,
                        local_root=input_path,
                        remote_root="/mnt/user-data/uploads",
                    )
                    logger.warning(f"[SandboxTerminalTool] upload ok: {remote_paths}")
                except Exception as e:
                    logger.warning(f"[SandboxTerminalTool] upload failed: {e}")

        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        session = self._sessions.get(instance_id)
        if session is None:
            return ToolResponse(text=json.dumps({"error": "no sandbox session for this instance"})), 0.0, {}

        command = parameters.get("command", "")

        timeout = self.config.get("timeout", 120.0)
        result = await session.exec(command, timeout=timeout)

        return_code = result.get("return_code", 0)
        output = result.get("output", "")

        if result.get("error"):
            text = json.dumps({"error": _truncate_output(result["error"])}, ensure_ascii=False)
        elif return_code != 0 and output:
            text = json.dumps({"error": _truncate_output(output)}, ensure_ascii=False)
        else:
            text = json.dumps({"status": "success", "output": _truncate_output(output)}, ensure_ascii=False)

        return ToolResponse(text=text), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        session = self._sessions.get(instance_id)
        if session is None:
            return 0.0
        files = await session.list_output_files()
        return 1.0 if files else 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._sessions.pop(instance_id, None)
        # Session lifecycle is managed by RealAgentLoop; do not destroy here.

"""view_image_text_tool.py -- verl BaseTool wrapper for image understanding.

Downloads images from sandbox before passing them to the LLM for vision understanding.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_VIEW_IMAGE_TEXT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "view_image_text",
        "description": "Understand an image using AI vision. Pass a query and a sandbox image path. Returns a text description of the image.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What you want to understand about the image."},
                "image_path": {"type": "string", "description": "Image path in sandbox (e.g., /mnt/user-data/uploads/image.png)."},
            },
            "required": ["query", "image_path"],
        },
    },
}


def _normalize_sandbox_path(filename: str) -> str:
    p = str(filename).strip()
    if not p:
        return ""
    return p if p.startswith("/") else f"/mnt/user-data/uploads/{p}"


def _json_tool_response(data: dict) -> tuple[ToolResponse, float, dict]:
    """Match project/tools/view_image_text.py: success {\"result\"}, failure {\"error\"}."""
    return ToolResponse(text=json.dumps(data, ensure_ascii=False)), 0.0, {}


class ViewImageTextTool(BaseTool):
    """verl BaseTool for image understanding using LLM vision."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._sessions: dict[str, Any] = {}
        self._tmp_dirs: dict[str, Path] = {}  # instance_id → persistent download dir

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self,
        instance_id: Optional[str] = None,
        session=None,
        tmp_dir: str = "",
        **kwargs,
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        if session is not None:
            self._sessions[instance_id] = session
        if tmp_dir:
            p = Path(tmp_dir)
            p.mkdir(parents=True, exist_ok=True)
            self._tmp_dirs[instance_id] = p
        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        session = self._sessions.get(instance_id)
        if session is None:
            return _json_tool_response({"error": "no sandbox session for this instance"})

        query = parameters.get("query", "")
        image_path = parameters.get("image_path", "")

        if not query or not image_path:
            return _json_tool_response({"error": "query and image_path are required"})

        if not image_path.lower().endswith((".png", ".jpg", ".jpeg")):
            msg = f"Unsupported image format: '{image_path}'. Only .png, .jpg, .jpeg are supported."
            logger.warning(f"[view_image_text][{instance_id}] {msg}")
            return _json_tool_response({"error": msg})

        remote_path = _normalize_sandbox_path(image_path)
        logger.warning(f"[view_image_text][{instance_id}] image_path: {remote_path}, query: {query}")

        # Use persistent tmp_dir if configured, otherwise use a temp dir
        persistent_dir = self._tmp_dirs.get(instance_id)
        if persistent_dir is not None:
            download_dir = persistent_dir
            tmp_ctx = None
        else:
            tmp_ctx = tempfile.TemporaryDirectory()
            download_dir = Path(tmp_ctx.name)

        try:
            # Pre-check: verify the file exists in sandbox before attempting download
            check_result = await session.exec(f'find "{remote_path}" -type f 2>/dev/null', timeout=10.0)
            found = check_result.get("output", "").strip()
            if not found:
                msg = f"Image file not found in sandbox: {remote_path}"
                logger.warning(f"[view_image_text][{instance_id}] {msg}")
                return _json_tool_response({"error": msg})

            success = await session.download_file(remote_path, download_dir)
            if not success and remote_path != image_path:
                success = await session.download_file(image_path, download_dir)
            if not success:
                msg = f"Failed to download image from sandbox: {image_path}"
                logger.warning(f"[view_image_text][{instance_id}] {msg}")
                return _json_tool_response({"error": msg})

            downloaded_files = [p for p in download_dir.iterdir() if p.is_file()]
            if not downloaded_files:
                return _json_tool_response(
                    {"error": f"Image file not found after download: {image_path}"}
                )

            # Pick the most recently modified file (last download)
            local_image_path = str(
                sorted(downloaded_files, key=lambda p: p.stat().st_mtime)[-1]
            )

            try:
                from llm.image_utils import mime_and_base64_from_image_url
                mime, b64 = mime_and_base64_from_image_url(local_image_path)
            except (ValueError, FileNotFoundError, OSError) as e:
                return _json_tool_response({"error": f"Failed to read image: {e}"})

            data_url = f"data:{mime};base64,{b64}"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ]

            family = self.config.get("llm_family", "qwen")
            model_id = self.config.get("llm_model_id", "qwen3.6-plus")
            from recipe.agent_r2._settings import load_settings
            sk = load_settings()["llm_api_key"]
            api_url = load_settings()["llm_api_url"]

            if not sk:
                return _json_tool_response({"error": "llm_api_key not configured in settings.yaml"})

            try:
                from llm import call_model
                result = await call_model(
                    family=family,
                    model_id=model_id,
                    sk=sk,
                    messages=messages,
                    tools=[],
                    api_url=api_url,
                )
                if result.get("error"):
                    logger.warning(f"[view_image_text][{instance_id}] llm error: {result['error']}")
                    return _json_tool_response({"error": result["error"]})
                return _json_tool_response({"result": result.get("content", "")})
            except Exception as e:
                return _json_tool_response({"error": str(e)})
        finally:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._sessions.pop(instance_id, None)
        self._tmp_dirs.pop(instance_id, None)
        # Session lifecycle is managed by RealAgentLoop; do not destroy here.

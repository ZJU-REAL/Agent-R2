"""web_fetch_tool.py -- verl BaseTool wrapper for web fetch."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Optional
from uuid import uuid4

import aiohttp
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from recipe.agent_r2._settings import load_settings

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_SERPER_SCRAPE_URL = "https://scrape.serper.dev"
_MAX_LENGTH = 6000

_WEB_FETCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_fetch",
        "description": "Read a web page and return its title and content.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL of the web page to read."}
            },
            "required": ["url"],
        },
    },
}


def _json_tool_response(data: dict) -> tuple[ToolResponse, float, dict]:
    """Match project/tools/web_fetch.py: success {\"content\"}, failure {\"content\", \"error\"}."""
    return ToolResponse(text=json.dumps(data, ensure_ascii=False)), 0.0, {}


class WebFetchTool(BaseTool):
    """verl BaseTool for web fetch."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.api_key = load_settings()["serper_api_key"]
        self.proxy = load_settings().get("proxy") or None

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self,
        instance_id: Optional[str] = None,
        **kwargs,
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        url = parameters.get("url", "")
        if not url:
            return _json_tool_response({"error": "url is required"})

        normalized = url if url.startswith(("http://", "https://")) else f"https://{url}"

        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {"url": normalized, "includeMarkdown": True}
        proxy = (
            os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY") or
            os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY") or
            self.proxy
        )

        last_error = None
        for attempt in range(3):
            try:
                timeout = aiohttp.ClientTimeout(total=30)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        _SERPER_SCRAPE_URL, headers=headers, json=payload, proxy=proxy
                    ) as response:
                        response.raise_for_status()
                        result = await response.json()

                content = result.get("markdown") or result
                if isinstance(content, str) and len(content) > _MAX_LENGTH:
                    truncated_chars = len(content) - _MAX_LENGTH
                    content = f"{content[:_MAX_LENGTH]}\n\n... [truncated {truncated_chars} characters]"

                if not isinstance(content, str):
                    content = str(content)
                return _json_tool_response({"content": content})
            except Exception as e:
                last_error = e
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)

        logger.warning(f"[web_fetch][{instance_id}] error: {last_error}")
        return _json_tool_response({"content": "", "error": str(last_error)})

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        pass

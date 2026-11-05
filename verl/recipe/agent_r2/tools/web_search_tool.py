"""web_search_tool.py -- verl BaseTool wrapper for web search."""
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

_SERPER_URL = "https://google.serper.dev/search"

_WEB_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information. Returns text snippets from search results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."}
            },
            "required": ["query"],
        },
    },
}


async def _serper_search(query: str, api_key: str, default_proxy: Optional[str], max_retries: int = 3) -> dict:
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = json.dumps({"q": query, "autocorrect": False})

    proxy = (
        os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY") or
        os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY") or
        default_proxy
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    _SERPER_URL, headers=headers, data=payload, proxy=proxy
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    results = data.get("organic", [])[:1]
                    return {"search_results": results}
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
    return {"search_results": [], "error": str(last_error)}


class WebSearchTool(BaseTool):
    """verl BaseTool for web search."""

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
        query = parameters.get("query", "")
        if not query:
            return ToolResponse(text="Error: query is required"), 0.0, {}

        result = await _serper_search(query, self.api_key, self.proxy)
        if result.get("error"):
            logger.warning(f"[web_search][{instance_id}] error: {result['error']}")
        text = json.dumps(result, ensure_ascii=False)
        return ToolResponse(text=text), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        pass

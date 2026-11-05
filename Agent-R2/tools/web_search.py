"""web_search -- serper 文本搜索。"""
from __future__ import annotations
import asyncio
import json
import os
from typing import Any, Dict
import aiohttp
from settings import load_settings

_SERPER_API_KEY = load_settings()["serper_api_key"]
_SERPER_URL = "https://google.serper.dev/search"
_PROXY = load_settings().get("proxy") or None

WEB_SEARCH_TOOL = {
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


async def web_search(query: str, **kwargs) -> Dict[str, Any]:
    return await _serper_search(query, _SERPER_URL)


async def _serper_search(query: str, url: str, extra_params: dict = None, max_retries: int = 3) -> Dict[str, Any]:
    api_key = _SERPER_API_KEY
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = json.dumps({"q": query, "autocorrect": False, **(extra_params or {})})

    proxy = (
        os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY') or
        os.environ.get('http_proxy') or os.environ.get('HTTP_PROXY') or
        _PROXY
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, data=payload, proxy=proxy) as response:
                    response.raise_for_status()
                    data = await response.json()
                    results = data.get('organic', [])[:1]
                    return {"search_results": results}
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
    return {"search_results": [], "error": str(last_error)}


if __name__ == "__main__":
    import asyncio

    async def test():
        result = await web_search("The capital of the United States")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    asyncio.run(test())
"""web_fetch -- 读取网页内容。"""
from __future__ import annotations
import asyncio
import os
from typing import Any, Dict
import aiohttp
from settings import load_settings

_SERPER_SCRAPE_URL = "https://scrape.serper.dev"
_SERPER_API_KEY = load_settings()["serper_api_key"]
_MAX_LENGTH = 6000
_PROXY = load_settings().get("proxy") or None

WEB_FETCH_TOOL = {
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


async def web_fetch(url: str, **kwargs) -> Dict[str, Any]:
    normalized = url if url.startswith(("http://", "https://")) else f"https://{url}"

    headers = {
        "X-API-KEY": _SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {"url": normalized, "includeMarkdown": True}

    proxy = (
        os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY') or
        os.environ.get('http_proxy') or os.environ.get('HTTP_PROXY') or
        _PROXY
    )

    last_error = None
    for attempt in range(3):
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(_SERPER_SCRAPE_URL, headers=headers, json=payload, proxy=proxy) as response:
                    response.raise_for_status()
                    result = await response.json()

            content = result.get("markdown") or result

            if isinstance(content, str) and len(content) > _MAX_LENGTH:
                truncated_chars = len(content) - _MAX_LENGTH
                content = f"{content[:_MAX_LENGTH]}\n\n... [truncated {truncated_chars} characters]"

            return {"content": content}
        except Exception as e:
            last_error = e
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
    return {"content": "", "error": str(last_error)}


if __name__ == "__main__":
    import asyncio
    import json

    async def test():
        result = await web_fetch("https://ors.od.nih.gov/sr/dohs/ergonomics/computer-workstation-ergonomics-self-assessment-checklist")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    asyncio.run(test())

from __future__ import annotations
from functools import partial
from typing import Any, Dict, List, Optional

from llm.claude import call_claude
from llm.gemini import call_gemini
from llm.openrouter import call_openrouter
from llm.vllm import call_vllm

MODEL_CALLERS = {
    "gpt":        partial(call_openrouter, family="gpt"),
    "qwen":       partial(call_openrouter, family="qwen"),
    "glm":        partial(call_openrouter, family="glm"),
    "doubao":     partial(call_openrouter, family="doubao"),
    "deepseek":   partial(call_openrouter, family="deepseek"),
    "claude":     call_claude,
    "gemini":     call_gemini,
    "vllm":       call_vllm,
}


async def call_model(
    family: str,
    model_id: str,
    sk: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    *,
    api_url: Optional[str] = None,
    **extra: Any,
) -> Dict[str, Any]:
    caller = MODEL_CALLERS.get(family)
    if not caller:
        return {"error": f"Unknown model family: {family}", "content": ""}
    return await caller(model_id, sk, messages, tools, api_url=api_url, **extra)

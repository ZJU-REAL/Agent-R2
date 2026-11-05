"""OpenAI-compatible async LLM client (GPT, Qwen, Doubao, GLM, etc.)."""
import asyncio
from copy import deepcopy
from urllib.parse import urlsplit
from openai import AsyncOpenAI
from llm.image_utils import openai_image_part_with_data_url

_EXTRA_BODY: dict[str, dict] = {
    "qwen":   {"enable_thinking": True},
    "glm":    {"thinking": {"type": "enabled"}},
    "doubao": {"thinking": {"type": "enabled"}},
    "deepseek":   {"enable_thinking": True},
}


async def call_openrouter(
    model_id: str,
    sk: str,
    messages: list,
    tools: list,
    *,
    family: str = "gpt",
    api_url: str = None,
    reasoning_effort: str = None,
    max_retries: int = 10,
    retry_delay: float = 2.0,
    **kwargs,
) -> dict:
    base = (api_url or "").strip().rstrip("/")
    if base and urlsplit(base).path in ("", "/"):
        base += "/v1"
    client = AsyncOpenAI(base_url=base, api_key=sk)
    openai_messages = _convert_messages(deepcopy(messages))
    resp_tools = [t for t in (tools or []) if t.get("type") == "function"]

    extra_body = dict(_EXTRA_BODY.get(family, {}))
    if family == "doubao" and reasoning_effort:
        extra_body["reasoning_effort"] = reasoning_effort

    try:
        last_error = None
        for attempt in range(max_retries):
            try:
                kw: dict = {"model": model_id, "messages": openai_messages, "stream": True}
                if resp_tools:
                    kw["tools"] = resp_tools
                if extra_body:
                    kw["extra_body"] = extra_body

                stream = await client.chat.completions.create(**kw)
                content, reasoning_content = "", ""
                tool_calls: list = []
                finish_reason, usage = None, None

                async for chunk in stream:
                    if chunk.usage:
                        usage = {
                            "prompt_tokens": chunk.usage.prompt_tokens,
                            "completion_tokens": chunk.usage.completion_tokens,
                        }
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    finish_reason = choice.finish_reason or finish_reason
                    delta = choice.delta
                    if not delta:
                        continue
                    if getattr(delta, "reasoning_content", None):
                        reasoning_content += delta.reasoning_content
                    if delta.content:
                        content += delta.content
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            while len(tool_calls) <= tc.index:
                                tool_calls.append({"type": "function", "function": {"name": "", "arguments": ""}, "id": ""})
                            if tc.id:
                                tool_calls[tc.index]["id"] = tc.id
                            if tc.function and tc.function.name:
                                tool_calls[tc.index]["function"]["name"] += tc.function.name
                            if tc.function and tc.function.arguments:
                                tool_calls[tc.index]["function"]["arguments"] += tc.function.arguments

                if not content.strip() and not tool_calls:
                    raise ValueError(
                        f"Empty model response (finish_reason={finish_reason!r}, usage={usage!r})"
                    )
                return {
                    "role": "assistant",
                    "content": content.strip(),
                    "reasoning_content": reasoning_content.strip(),
                    "tool_calls": tool_calls or None,
                    "finish_reason": finish_reason,
                    "usage": usage,
                }
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * attempt)
                    continue
                raise Exception(f"{family} request failed: {e}")
        raise Exception(f"{family} request failed after {max_retries} retries: {last_error}")
    finally:
        await client.close()


def _convert_messages(messages: list) -> list:
    out = []
    for msg in messages:
        role, content = msg.get("role", ""), msg.get("content", "")
        if role == "system":
            out.append({"role": "system", "content": content})
        elif role == "user":
            if isinstance(content, list):
                items = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        items.append({"type": "text", "text": item.get("text", "")})
                    elif isinstance(item, dict) and item.get("type") == "image_url":
                        items.append(openai_image_part_with_data_url(item))
                    else:
                        items.append({"type": "text", "text": str(item) if not isinstance(item, dict) else ""})
                out.append({"role": "user", "content": items})
            else:
                out.append({"role": "user", "content": content})
        elif role == "assistant":
            item: dict = {"role": "assistant", "content": content or ""}
            if msg.get("tool_calls"):
                item["tool_calls"] = [
                    {
                        "id": tc.get("id", f"call_{tc['function']['name']}"),
                        "type": "function",
                        "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]},
                    }
                    for tc in msg["tool_calls"]
                ]
            out.append(item)
        elif role == "tool":
            tc = msg.get("content", "")
            if isinstance(tc, list):
                texts, images = [], []
                for item in tc:
                    if isinstance(item, dict) and item.get("type") == "text":
                        texts.append(item.get("text", ""))
                    elif isinstance(item, dict) and item.get("type") == "image_url":
                        images.append(openai_image_part_with_data_url(item))
                    else:
                        texts.append(str(item) if not isinstance(item, dict) else "")
                out.append({"role": "tool", "tool_call_id": msg.get("tool_call_id", ""), "content": "\n".join(texts)})
                if images:
                    out.append({"role": "user", "content": images})
            else:
                out.append({"role": "tool", "tool_call_id": msg.get("tool_call_id", ""), "content": tc})
    return out

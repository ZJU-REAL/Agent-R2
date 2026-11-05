"""vLLM async client (OpenAI-compatible). 支持多端点轮询负载均衡。"""
import asyncio
import json
from copy import deepcopy

import httpx
from openai import AsyncOpenAI
from llm.image_utils import openai_image_part_with_data_url

# 多端点轮询计数器
_vllm_endpoint_counter = 0


def _pick_endpoint(api_urls: str) -> str:
    """从逗号分隔的多端点中轮询选择一个。例如 'http://host:8000/v1,http://host:8001/v1'。"""
    global _vllm_endpoint_counter
    urls = [u.strip().rstrip("/") for u in api_urls.split(",") if u.strip()]
    if not urls:
        raise ValueError("vLLM api_url is empty")
    idx = _vllm_endpoint_counter % len(urls)
    _vllm_endpoint_counter += 1
    return urls[idx]


async def call_vllm(model_id: str, sk: str, messages: list, tools: list,
                    api_url: str = None,
                    chunk_idle_timeout: float = 120.0,
                    total_timeout: float = 120.0,
                    max_retries: int = 5, retry_delay: float = 2.0,
                    **kwargs) -> dict:
    """支持多端点: api_url 可以是逗号分隔的多个 URL。

    Args:
        chunk_idle_timeout: 每个 chunk 之间最长空闲等待秒数（防止流中途 hang 死）
        total_timeout: 整个流消费的总超时秒数（兜底保护）
    """
    base = _pick_endpoint(api_url or "")
    # 用 httpx.Timeout 精细控制：read 超时作用于每次 chunk 读取，防止流中途卡死
    client = AsyncOpenAI(
        base_url=base,
        api_key=sk,
        timeout=httpx.Timeout(connect=120.0, read=chunk_idle_timeout, write=30.0, pool=10.0),
    )
    openai_messages = _convert_messages(deepcopy(messages))
    resp_tools = [t for t in tools if t.get("type") == "function"]
    try:
        last_error = None
        for attempt in range(max_retries):
            try:
                kw = {"model": model_id, "messages": openai_messages, "stream": True, "temperature": 0.6}
                if resp_tools:
                    kw["tools"] = resp_tools
                stream = await client.chat.completions.create(**kw)
                content, reasoning_content = "", ""
                tool_calls = []
                finish_reason, usage = None, None

                async def _consume():
                    nonlocal content, reasoning_content, tool_calls, finish_reason, usage
                    async for chunk in stream:
                        if chunk.usage:
                            usage = {"prompt_tokens": chunk.usage.prompt_tokens, "completion_tokens": chunk.usage.completion_tokens}
                        if not chunk.choices:
                            continue
                        choice = chunk.choices[0]
                        finish_reason = choice.finish_reason or finish_reason
                        delta = choice.delta
                        if not delta:
                            continue
                        if getattr(delta, 'reasoning', None):
                            reasoning_content += delta.reasoning
                        if delta.content:
                            content += delta.content
                        if delta.tool_calls:
                            for tc in delta.tool_calls:
                                while len(tool_calls) <= tc.index:
                                    tool_calls.append({"type": "function", "function": {"name": "", "arguments": ""}, "id": ""})
                                if tc.id:
                                    tool_calls[tc.index]['id'] = tc.id
                                if tc.function and tc.function.name:
                                    tool_calls[tc.index]['function']['name'] += tc.function.name
                                if tc.function and tc.function.arguments:
                                    tool_calls[tc.index]['function']['arguments'] += tc.function.arguments

                try:
                    await asyncio.wait_for(_consume(), timeout=total_timeout)
                except asyncio.TimeoutError:
                    raise Exception(f"vLLM stream timed out after {total_timeout}s")

                return {"role": "assistant", "content": content.strip(), "reasoning_content": reasoning_content.strip(),
                        "tool_calls": tool_calls or None, "finish_reason": finish_reason, "usage": usage}
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                    continue
                raise Exception(f"vLLM request failed: {e}")
        raise Exception(f"vLLM request failed after {max_retries} retries: {last_error}")
    finally:
        await client.close()


def _convert_messages(messages):
    """与 qwen 相同的 OpenAI chat format 转换。"""
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
                    elif not isinstance(item, dict):
                        items.append({"type": "text", "text": str(item)})
                out.append({"role": "user", "content": items})
            else:
                out.append({"role": "user", "content": content})
        elif role == "assistant":
            item = {"role": "assistant", "content": content or ""}
            if msg.get("tool_calls"):
                tool_calls_out = []
                for tc in msg["tool_calls"]:
                    args = tc["function"]["arguments"]
                    if isinstance(args, str):
                        try:
                            args = json.dumps(json.loads(args), ensure_ascii=False)
                        except Exception:
                            args = json.dumps(args)
                    elif not isinstance(args, str):
                        args = json.dumps(args, ensure_ascii=False)
                    tool_calls_out.append({
                        "id": tc.get("id", f"call_{tc['function']['name']}"),
                        "type": "function",
                        "function": {"name": tc["function"]["name"], "arguments": args},
                    })
                item["tool_calls"] = tool_calls_out
            out.append(item)
        elif role == "tool":
            tc = msg.get("content", "")
            if isinstance(tc, list):
                items = []
                for item in tc:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        items.append(item.get('text', ''))
                    elif isinstance(item, dict) and item.get('type') == 'image_url':
                        items.append(openai_image_part_with_data_url(item))
                    elif not isinstance(item, dict):
                        items.append(str(item))
                out.append({"role": "tool", "tool_call_id": msg.get("tool_call_id", ""), "content": items})
            else:
                out.append({"role": "tool", "tool_call_id": msg.get("tool_call_id", ""), "content": tc})
    return out

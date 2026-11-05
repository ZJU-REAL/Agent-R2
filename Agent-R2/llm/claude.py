"""Claude async LLM client."""
import asyncio
import json
import re
import uuid
from copy import deepcopy
import aiohttp
from llm.image_utils import mime_and_base64_from_image_url


async def call_claude(model_id: str, sk: str, messages: list, tools: list,
                      api_url: str = None, reasoning_effort: str = "max",
                      max_retries: int = 10, retry_delay: float = 2.0, **kwargs) -> dict:
    base = (api_url or "").strip().rstrip("/")
    url = f"{base}/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {sk}",
        "anthropic-beta": "interleaved-thinking-2025-05-14",
        "anthropic-version": "2023-06-01",
    }
    messages = deepcopy(messages)
    system_prompt = next((m.pop('content') for m in messages if m['role'] == 'system'), None)
    messages = [m for m in messages if m['role'] != 'system']
    claude_tools = [
        {"name": t['function']['name'], "description": t['function'].get('description', ''),
         "input_schema": t['function'].get('parameters', {"type": "object", "properties": {}})}
        for t in tools
    ]
    claude_messages = _convert_messages(messages)

    if reasoning_effort == "disabled":
        thinking_cfg = {"type": "disabled"}
        max_tokens = 16384
        payload_extra = {}
    elif re.search(r"4[.\-]5", model_id):
        thinking_cfg = {"type": "enabled", "budget_tokens": 64000}
        max_tokens = 64000
        payload_extra = {}
    else:
        thinking_cfg = {"type": "adaptive"}
        max_tokens = 128000
        payload_extra = {"output_config": {"effort": reasoning_effort}}

    payload = {
        "model": model_id,
        "messages": claude_messages,
        "tools": claude_tools,
        "stream": True,
        "max_tokens": max_tokens,
        "thinking": thinking_cfg,
        "anthropic_beta": ["interleaved-thinking-2025-05-14"],
        **payload_extra,
    }
    if system_prompt:
        payload["system"] = [{"type": "text", "text": system_prompt}]

    last_error = None
    for attempt in range(max_retries):
        try:
            content = ""
            reasoning_content = ""
            thinking_signature = ""
            tool_calls = []
            finish_reason = None
            usage = None
            current_tool = None

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=10800)) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        raise Exception(f"HTTP {resp.status}: {body}")
                    async for raw_line in resp.content:
                        line = raw_line.decode("utf-8").strip()
                        if not line.startswith("data: "):
                            if "error" in line:
                                raise Exception(f"Stream error: {line}")
                            continue
                        try:
                            chunk = json.loads(line[len("data: "):])
                        except json.JSONDecodeError:
                            continue

                        ctype = chunk.get("type")
                        if ctype == "message_start":
                            u = chunk.get("message", {}).get("usage", {})
                            usage = {"prompt_tokens": u.get("input_tokens", 0),
                                     "completion_tokens": u.get("output_tokens", 0)}
                        elif ctype == "content_block_start":
                            cb = chunk.get("content_block", {})
                            if cb.get("type") == "tool_use":
                                current_tool = {"type": "function",
                                                "function": {"name": cb["name"], "arguments": ""},
                                                "id": cb.get("id") or f"claude-{uuid.uuid4()}"}
                        elif ctype == "content_block_delta":
                            delta = chunk.get("delta", {})
                            dtype = delta.get("type")
                            if dtype == "text_delta":
                                content += delta.get("text", "")
                            elif dtype == "thinking_delta":
                                reasoning_content += delta.get("thinking", "")
                            elif dtype == "signature_delta":
                                thinking_signature = delta.get("signature", "")
                            elif dtype == "input_json_delta" and current_tool:
                                current_tool["function"]["arguments"] += delta.get("partial_json", "")
                        elif ctype == "content_block_stop":
                            if current_tool and current_tool.get("function", {}).get("name"):
                                tool_calls.append(current_tool)
                                current_tool = None
                        elif ctype == "message_delta":
                            delta = chunk.get("delta", {})
                            if delta.get("stop_reason"):
                                finish_reason = {"end_turn": "stop", "tool_use": "tool_calls",
                                                 "max_tokens": "length"}.get(delta["stop_reason"], delta["stop_reason"])
                            if "usage" in chunk:
                                completion = chunk["usage"].get("output_tokens", 0)
                                prompt = (usage or {}).get("prompt_tokens", 0)
                                usage = {**(usage or {}), "completion_tokens": completion,
                                         "total_tokens": prompt + completion}

            for tc in tool_calls:
                if tc["function"]["arguments"].strip() == "":
                    tc["function"]["arguments"] = "{}"

            return {"role": "assistant", "content": content.strip(),
                    "reasoning_content": reasoning_content.strip(),
                    "thought_signature": thinking_signature or None,
                    "tool_calls": tool_calls or None, "finish_reason": finish_reason, "usage": usage}
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * attempt)
                continue
            raise Exception(f"Claude request failed: {e}")
    raise Exception(f"Claude request failed after {max_retries} retries: {last_error}")


def _convert_messages(messages: list) -> list:
    claude_msgs = []
    for msg in messages:
        role = msg.get('role')
        if role == 'user':
            content = _convert_user_content(msg.get('content', ''))
            if claude_msgs and claude_msgs[-1]['role'] == 'user':
                prev = claude_msgs[-1]['content']
                if isinstance(prev, list):
                    claude_msgs[-1]['content'] = prev + (content if isinstance(content, list) else [{"type": "text", "text": content}])
                else:
                    claude_msgs[-1]['content'] = prev + "\n" + content
            else:
                claude_msgs.append({"role": "user", "content": content})
        elif role == 'assistant':
            blocks = []
            if msg.get('reasoning_content') and msg.get('thought_signature'):
                blocks.append({"type": "thinking", "thinking": msg['reasoning_content'],
                                "signature": msg['thought_signature']})
            elif msg.get('reasoning_content'):
                blocks.append({"type": "redacted_thinking", "data": msg['reasoning_content']})
            if msg.get('content'):
                blocks.append({"type": "text", "text": msg['content']})
            for tc in msg.get('tool_calls') or []:
                args = tc['function']['arguments']
                if isinstance(args, str):
                    args = json.loads(args) if args else {}
                blocks.append({"type": "tool_use", "id": tc['id'],
                               "name": tc['function']['name'], "input": args})
            claude_msgs.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
        elif role == 'tool':
            tool_content = msg['content']
            if isinstance(tool_content, list):
                converted = []
                for item in tool_content:
                    if isinstance(item, dict) and item.get('type') == 'image_url':
                        url = (item.get('image_url') or {}).get('url', '')
                        try:
                            mime, data = mime_and_base64_from_image_url(url)
                            converted.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}})
                        except (ValueError, FileNotFoundError, OSError):
                            converted.append(item)
                    else:
                        converted.append(item)
                tool_content = converted
            tool_result = {"type": "tool_result", "tool_use_id": msg['tool_call_id'], "content": tool_content}
            if claude_msgs and claude_msgs[-1]['role'] == 'user':
                prev = claude_msgs[-1]['content']
                if isinstance(prev, list):
                    prev.append(tool_result)
                else:
                    claude_msgs[-1]['content'] = [{"type": "text", "text": prev}, tool_result]
            else:
                claude_msgs.append({"role": "user", "content": [tool_result]})
    return claude_msgs


def _convert_user_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                out.append({"type": "text", "text": item.get('text', '')})
            elif isinstance(item, dict) and item.get('type') == 'image_url':
                url = (item.get('image_url') or {}).get('url', '')
                try:
                    mime, data = mime_and_base64_from_image_url(url)
                    out.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}})
                except (ValueError, FileNotFoundError, OSError) as e:
                    out.append({"type": "text", "text": f"[image omitted: {url!s} ({e})]"})
        return out
    return content

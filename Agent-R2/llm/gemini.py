"""Gemini async LLM client."""
import asyncio
import json
import uuid
from copy import deepcopy
import aiohttp
from llm.image_utils import mime_and_base64_from_image_url


async def call_gemini(model_id: str, sk: str, messages: list, tools: list,
                      api_url: str = None, max_retries: int = 10, retry_delay: float = 2.0, **kwargs) -> dict:
    base = (api_url or "").strip().rstrip("/")
    url = f"{base}/v1/models/{model_id}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {sk}",
    }
    messages = deepcopy(messages)
    system_prompt = next((m["content"] for m in messages if m["role"] == "system"), None)
    messages = [m for m in messages if m["role"] != "system"]

    gemini_tools = [{"function_declarations": [t["function"] for t in tools]}] if tools else []
    contents = _convert_messages(messages)

    payload: dict = {
        "tools": gemini_tools,
        "contents": contents,
        "generationConfig": {
            "thinkingConfig": {
                "includeThoughts": True,
                "thinkingBudget": -1,
            }
        },
    }
    if system_prompt:
        payload["system_instruction"] = {"parts": [{"text": system_prompt}]}

    last_error = None
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload,
                                        timeout=aiohttp.ClientTimeout(total=10800)) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        raise Exception(f"HTTP {resp.status}: {body}")
                    response = await resp.json()

            if "candidates" not in response or "parts" not in response["candidates"][0]["content"]:
                raise Exception(f"Gemini unexpected response: {response}")

            candidate = response["candidates"][0]
            content, reasoning_content, thought_signature = "", "", ""
            tool_calls = []

            for part in candidate["content"]["parts"]:
                if "thoughtSignature" in part:
                    thought_signature = part["thoughtSignature"]
                if "text" in part:
                    if part.get("thought", False):
                        reasoning_content += part["text"]
                    else:
                        content += part["text"]
                if "functionCall" in part:
                    tool_calls.append({
                        "type": "function",
                        "function": {
                            "name": part["functionCall"]["name"],
                            "arguments": json.dumps(part["functionCall"]["args"], ensure_ascii=False),
                        },
                        "id": f"gemini-{uuid.uuid4()}",
                    })

            usage = response.get("usageMetadata")
            return {
                "role": "assistant",
                "content": content.strip(),
                "reasoning_content": reasoning_content.strip(),
                "thought_signature": thought_signature or None,
                "tool_calls": tool_calls or None,
                "finish_reason": candidate.get("finishReason"),
                "usage": usage,
            }
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * attempt)
                continue
            raise Exception(f"Gemini request failed: {e}")
    raise Exception(f"Gemini request failed after {max_retries} retries: {last_error}")


def _convert_messages(messages: list) -> list:
    contents = []
    tool_call_id2name: dict = {}

    for msg in messages:
        role = msg["role"]
        if role == "user":
            parts = _user_parts(msg.get("content", ""))
            if contents and contents[-1]["role"] == "user":
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": "user", "parts": parts})
        elif role == "tool":
            name = tool_call_id2name.get(msg["tool_call_id"], "unknown")
            part = {"functionResponse": {"name": name, "response": {"content": msg["content"]}}}
            if contents and contents[-1]["role"] == "user":
                contents[-1]["parts"].append(part)
            else:
                contents.append({"role": "user", "parts": [part]})
        elif role == "assistant":
            parts = []
            if msg.get("reasoning_content"):
                parts.append({"text": msg["reasoning_content"], "thought": True})
            if msg.get("content"):
                parts.append({"text": msg["content"]})
            for idx, tc in enumerate(msg.get("tool_calls") or []):
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)
                p: dict = {"functionCall": {"name": tc["function"]["name"], "args": args}}
                if idx == 0 and msg.get("thought_signature"):
                    p["thoughtSignature"] = msg["thought_signature"]
                tool_call_id2name[tc["id"]] = tc["function"]["name"]
                parts.append(p)
            contents.append({"role": "model", "parts": parts})

    return contents


def _user_parts(content) -> list:
    if isinstance(content, str):
        return [{"text": content}]
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append({"text": item["text"]})
        elif isinstance(item, dict) and item.get("type") == "image_url":
            try:
                mime, b64 = mime_and_base64_from_image_url((item.get("image_url") or {}).get("url", ""))
                parts.append({"inlineData": {"mimeType": mime, "data": b64}})
            except (ValueError, FileNotFoundError, OSError):
                pass
    return parts

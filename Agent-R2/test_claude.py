import re
import json
import requests
from copy import deepcopy


def claude(proxy_url, model_id, sk, messages, tools, reasoning_effort):
    messages = deepcopy(messages)
    system_setting = messages.pop(0)['content'] if messages[0]['role'] == 'system' else None

    for i, message in enumerate(messages):
        if message['role'] == 'assistant':
            content = []
            if message['reasoning_content']:
                content.append({
                    "type": "thinking",
                    "thinking": message['reasoning_content'],
                    "signature": message['signature']
                })
                
            if message['content']:
                content.append({
                    "type": "text",
                    "text": message['content']
                })
            
            for tool_call in message['tool_calls']:
                try:
                    arguments = json.loads(tool_call['function']['arguments'])
                except:
                    arguments = {}
                    
                tool_call = {
                    "type": "tool_use",
                    "id": tool_call['id'],
                    "name": tool_call['function']['name'],
                    "input": arguments
                }
                content.append(tool_call)

            messages[i] = {
                "role": "assistant",
                "content": content
            }
            
        if message['role'] == 'tool':
            messages[i] = {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message['tool_call_id'],
                        "content": message['content']
                    }
                ]
            }

        if message['role'] == 'user':
            messages[i] = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": message['content']
                    }
                ]
            }
    
    messages[-1]['content'][0]['cache_control'] = {"type": "ephemeral"}
        
    tools = deepcopy(tools)
    for i, tool in enumerate(tools):
        tools[i] = {
            'name': tool['function']['name'],
            'description': tool['function']['description'],
            'input_schema': tool['function']['parameters']
        }
    
    url = f"{proxy_url}/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "Authorization": sk,
        "anthropic-beta": "interleaved-thinking-2025-05-14",
        "anthropic-version": "2023-06-01"
    }
    payload = {
        "model": model_id,
        "messages": messages,
        "tools": tools,
        "stream": True,
        "anthropic_version": "bedrock-2023-05-31",
        "anthropic_beta": ["interleaved-thinking-2025-05-14"],
    }

    if re.search(r"4.5|4-5", model_id):
        payload["max_tokens"] = 64000
        payload["thinking"] = {"type": "enabled", "budget_tokens": 64000}
    else:
        payload["max_tokens"] = 128000
        payload["thinking"] = {"type": "adaptive"}
        payload["output_config"] = {"effort": reasoning_effort}

    if system_setting:
        payload["system"] = [
            {
                "type": "text",
                "text": system_setting
            }
        ]

    headers['Authorization'] = f"Bearer {sk}"

    reasoning_content = ""
    tool_calls = []
    content = ""
    signature = ""
    finish_reason = None
    usage = None

    with requests.post(url, headers=headers, json=payload, stream=True, timeout=10800) as response:
        for line in response.iter_lines():
            line = line.decode("utf-8")

            try:
                if line.startswith("data: "):
                    chunk = json.loads(line[len("data: "):])
                    
                    if chunk['type'] == "content_block_start":
                        content_block = chunk["content_block"]
                        if content_block["type"] == "tool_use":
                            tool_call = {
                                "id": content_block["id"],
                                "type": "function",
                                "function": {
                                    "name": content_block["name"],
                                    "arguments": ""
                                }
                            }
                            tool_calls.append(tool_call)

                    if chunk['type'] == "content_block_delta":
                        delta = chunk["delta"]
                        
                        if delta["type"] == "thinking_delta":
                            reasoning_content += delta["thinking"]
                        
                        if delta["type"] == "signature_delta":
                            signature += delta.get("signature", "")
                            
                        if delta["type"] == "input_json_delta":
                            tool_calls[-1]['function']['arguments'] += delta["partial_json"]
                            
                        if delta["type"] == "text_delta":
                            content += delta["text"]
                    
                    if chunk['type'] == "message_delta":
                        delta = chunk["delta"]
                        finish_reason = delta["stop_reason"]
                        if 'usage' in chunk:
                            usage = chunk['usage']
                        
                    if chunk['type'] == "message_stop":
                        if 'amazon-bedrock-invocationMetrics' in chunk:
                            usage = chunk["amazon-bedrock-invocationMetrics"]
                        
                else:
                    if 'error' in line:
                        raise Exception(f"请求失败: {line}")
            except:
                raise Exception(f"请求失败: {line}")
        
    for tool_call in tool_calls:
        if tool_call['function']['arguments'].strip() == "":
            tool_call['function']['arguments'] = "{}"

    message = {
        "role": "assistant",
        "reasoning_content": reasoning_content.strip(),
        "tool_calls": tool_calls,
        "content": content.strip(),
        "signature": signature,
        "finish_reason": finish_reason,
        "usage": usage
    }
    
    return message


if __name__ == "__main__":
    tools = [
        {
            "type": "function",
            "function":{
                "name": "python",
                "description": "Interpreter for executing python code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "description": "Code to execute",
                            "type": "string"
                        }
                    },
                    "required": [
                        "code"
                    ]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "这是一个高级联网工具。当根据你的知识库，无法准确地回答用户问题时，调用该工具搜索查询用户所需的外部信息。注意该工具无法提供图片、视频等非文本内容。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_list": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "description": "搜索问题，支持在问题末尾添加'site:xxx'以指定搜索站点"
                            },
                            "description": "基于原始问题改写的多个搜索问题"
                        }
                    },
                    "required": [
                        "query_list"
                    ]
                }
            }
        }
    ]
    messages = [
        {
            "role": "system",
            "content": "you are a math expert"
        },
        {
            "role": "user",
            "content": "hi",
        }
    ]
    # 网关地址和 key 读取 settings/settings.yaml
    from settings import load_settings
    proxy_url = load_settings()["llm_api_url"]
    sk = load_settings()["llm_api_key"]

    model_id = 'claude-opus-4-5-20251101'
    # model_id = 'claude-sonnet-4-6'
    reasoning_effort = 'high'
    print(claude(proxy_url, model_id, sk, messages, tools, reasoning_effort))
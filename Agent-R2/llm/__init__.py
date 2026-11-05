"""LLM 调用层：统一路由到 gpt / claude / gemini / qwen / vllm / glm。"""
from llm.router import call_model
__all__ = ["call_model"]

"""data-sythesis agents/scenario_agent.py -- 场景生成Agent。

负责根据网络语料内容，挖掘潜在的工作流场景。
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_sythesis.prompts.scenario_prompt import build_scenario_system_prompt, build_scenario_user_prompt
from data_sythesis.prompts.retry_prompt import build_json_retry_message
from data_sythesis.utils import append_messages, parse_json_response

from llm.router import call_model

logger = logging.getLogger(__name__)


@dataclass
class ScenarioAgentConfig:
    """场景生成Agent的配置。"""
    family: str
    model_id: str
    max_retry: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "model_id": self.model_id,
            "max_retry": self.max_retry,
        }


async def run_scenario_agent(
    *,
    original_document: str,
    discipline: str,
    title: str,
    candidate_tasks: List[str],
    config: ScenarioAgentConfig,
    sk: str,
    api_url: str,
    conversation_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """生成场景。

    Args:
        original_document: 原始文档内容
        discipline: 学科领域
        title: 职业名称
        candidate_tasks: 候选任务列表
        config: Agent配置
        sk: API密钥
        api_url: API地址
        conversation_dir: conversation 文件夹路径，用于记录对话

    Returns:
        生成的场景数据，失败返回None
    """
    system_prompt = build_scenario_system_prompt()
    user_prompt = build_scenario_user_prompt(
        original_document=original_document,
        title=title,
        candidate_tasks=candidate_tasks
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    max_retries = config.max_retry
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            resp = await call_model(
                family=config.family,
                model_id=config.model_id,
                sk=sk,
                messages=messages,
                tools=[],  # 场景生成不需要工具
                api_url=api_url
            )
            elapsed = time.time() - t0
            logger.info(f"[scenario_agent] model call took {elapsed:.1f}s")

            # 将模型输出添加到 messages
            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": resp.get("content", "")}
            if resp.get("reasoning_content"):
                assistant_msg["reasoning_content"] = resp["reasoning_content"]
            if resp.get("usage"):
                assistant_msg["usage"] = resp["usage"]
            messages.append(assistant_msg)

            # 实时追加写入对话记录
            if conversation_dir:
                append_messages(
                    conversation_dir=conversation_dir,
                    phase="scenario_agent",
                    messages=messages
                )

            content = resp.get("content", "")
            if not content:
                logger.error("模型返回空内容")
                return None

            # 尝试解析JSON
            result = parse_json_response(content)
            if result:
                # 添加原始信息
                result["discipline"] = discipline
                result["title"] = title
                result["original_document_length"] = len(original_document)

                # 校验必须字段
                if "summary" not in result:
                    logger.error("模型返回结果缺少 summary 字段")
                    return None
                if "workflow" not in result:
                    logger.error("模型返回结果缺少 workflow 字段")
                    return None

                # 保存最终的对话记录
                if conversation_dir:
                    append_messages(
                        conversation_dir=conversation_dir,
                        phase="scenario_agent",
                        messages=messages
                    )
                return result

            # JSON解析失败，追加重试消息
            logger.warning(f"[scenario_agent] JSON解析失败，第 {attempt + 1} 次重试")
            if attempt < max_retries - 1:
                messages.append({
                    "role": "user",
                    "content": build_json_retry_message()
                })
                continue
            else:
                logger.error("无法解析模型返回的JSON")
                return None

        except Exception as e:
            logger.error(f"场景生成失败: {e}")
            if conversation_dir:
                append_messages(
                    conversation_dir=conversation_dir,
                    phase="scenario_agent",
                    messages=messages
                )
            return None

    # 保存最终的对话记录
    if conversation_dir:
        append_messages(
            conversation_dir=conversation_dir,
            phase="scenario_agent",
            messages=messages
        )
    return None
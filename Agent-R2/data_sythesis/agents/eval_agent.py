"""data-sythesis agents/eval_agent.py -- 评分Agent。

这个Agent接收question_agent生成的rubric和solver_agent的对话轨迹，
按rubric逐项打分，输出结构化评估结果。
不需要沙盒环境，纯LLM推理。
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_sythesis.prompts.eval_prompt import (
    build_eval_system_prompt,
    build_eval_user_prompt,
)
from data_sythesis.utils import append_messages

from llm.router import call_model

logger = logging.getLogger(__name__)


@dataclass
class EvalAgentConfig:
    """评分Agent的配置。"""
    family: str
    model_id: str
    max_steps: int = 10  # 评分通常不需要多轮，给足够的思考步数即可

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "model_id": self.model_id,
            "max_steps": self.max_steps,
        }


def _parse_evaluation_json(text: str) -> Optional[Dict[str, Any]]:
    """从模型回复中提取 JSON 评估结果。"""
    # 尝试提取 ```json ... ``` 代码块
    json_block = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_block:
        try:
            return json.loads(json_block.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试提取最外层 { ... }
    brace_match = re.search(r'\{[\s\S]*\}', text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _load_solver_trajectory(solver_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从 solver_result 中提取对话轨迹。

    solver_result 可能直接包含 messages，也可能指向 conversation_dir。
    """
    # 优先从 solver_result 中直接取 messages
    messages = solver_result.get("messages", [])
    if messages:
        return messages

    # 尝试从 conversation_dir 加载
    conversation_dir = solver_result.get("conversation_dir", "")
    if conversation_dir:
        conv_dir = Path(conversation_dir)
        # 匹配 {family}_solver_agent.json（如 claude_solver_agent.json）
        candidates = list(conv_dir.glob("*_solver_agent.json"))
        conv_path = candidates[0] if candidates else conv_dir / "solver_agent.json"
        if conv_path.exists():
            try:
                with open(conv_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"[eval_agent] 加载对话轨迹失败: {e}")

    return []


def _build_eval_output(
    eval_result: Dict[str, Any],
    rubric: List[Dict[str, Any]],
    steps: int,
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """构建评估输出，从rubric补充criterion和max_score。"""
    evaluations_raw = eval_result.get("evaluations", [])

    # rubric_item_id -> rubric_item 映射
    rubric_map = {
        item.get("rubric_item_id", item.get("id", "")): item
        for item in rubric
    }

    evaluations = []
    total_score = 0
    for eval_item in evaluations_raw:
        rubric_item_id = eval_item.get("rubric_item_id", "")
        rubric_item = rubric_map.get(rubric_item_id, {})

        criterion = rubric_item.get("criterion", eval_item.get("criterion", ""))
        max_score = rubric_item.get("score", rubric_item.get("max_score", eval_item.get("max_score", 0)))
        score = min(eval_item.get("score", 0), max_score)  # 不超过上限
        reasoning = eval_item.get("reasoning", "")

        evaluations.append({
            "rubric_item_id": rubric_item_id,
            "criterion": criterion,
            "score": score,
            "max_score": max_score,
            "reasoning": reasoning,
        })
        total_score += score

    total_max_score = sum(
        item.get("score", item.get("max_score", 0)) for item in rubric
    )

    return {
        "evaluations": evaluations,
        "total_score": total_score,
        "total_max_score": total_max_score,
        "steps": steps,
        "status": "completed",
        "messages": messages,
    }


async def run_eval_agent(
    *,
    question_data: Dict[str, Any],
    solver_result: Dict[str, Any],
    config: EvalAgentConfig,
    sk: str,
    api_url: str,
    conversation_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """运行评分Agent。

    基于question_agent生成的rubric和solver_agent的对话轨迹，逐项打分。
    不需要沙盒环境，纯LLM推理，通常1轮即可完成。

    Args:
        question_data: question_agent的产出，包含 question、rubric 等
        solver_result: solver_agent的产出，包含 messages 或 conversation_dir
        config: Agent配置
        sk: API密钥
        api_url: API地址
        conversation_dir: 对话记录保存目录

    Returns:
        评估结果，包含:
        - evaluations: 逐项评分列表
        - total_score: 总得分
        - total_max_score: 总满分
        - steps: 实际执行轮数
        - status: 完成状态
        - messages: 对话消息列表
    """
    rubric: List[Dict[str, Any]] = question_data.get("rubric", [])
    question: Dict[str, Any] = question_data.get("question", {})

    # 无 rubric 直接返回空结果
    if not rubric:
        logger.warning("[eval_agent] 没有 rubric，跳过评分")
        return {
            "evaluations": [],
            "total_score": 0,
            "total_max_score": 0,
            "steps": 0,
            "status": "no_rubric",
            "messages": [],
        }

    # 提取 solver 对话轨迹
    solver_trajectory = _load_solver_trajectory(solver_result)
    if not solver_trajectory:
        logger.warning("[eval_agent] solver 轨迹为空，所有项目得0分")
        evaluations = [
            {
                "rubric_item_id": item.get("rubric_item_id", item.get("id", "")),
                "criterion": item.get("criterion", ""),
                "score": 0,
                "max_score": item.get("score", item.get("max_score", 0)),
                "reasoning": "Solver produced no trajectory.",
            }
            for item in rubric
        ]
        return {
            "evaluations": evaluations,
            "total_score": 0,
            "total_max_score": sum(item.get("score", item.get("max_score", 0)) for item in rubric),
            "steps": 0,
            "status": "no_trajectory",
            "messages": [],
        }

    # 构建 prompt
    system_prompt = build_eval_system_prompt(max_steps=config.max_steps)
    user_prompt = build_eval_user_prompt(
        rubric=rubric,
        solver_trajectory=solver_trajectory,
        question=question if question else None,
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    steps = 0
    max_steps = config.max_steps

    while steps < max_steps:
        steps += 1
        t0 = time.time()

        # 实时写入对话记录
        if conversation_dir:
            append_messages(
                conversation_dir=conversation_dir,
                phase="eval_agent",
                messages=messages,
            )

        resp = await call_model(
            family=config.family,
            model_id=config.model_id,
            sk=sk,
            messages=messages,
            tools=None,
            api_url=api_url,
        )

        elapsed = time.time() - t0
        logger.info(f"[eval_agent step {steps}] model call took {elapsed:.1f}s")

        if resp.get("error"):
            messages.append({"role": "assistant", "content": f"Model error: {resp['error']}"})
            break

        assistant_msg: Dict[str, Any] = {"role": "assistant", "content": resp.get("content", "")}
        if resp.get("reasoning_content"):
            assistant_msg["reasoning_content"] = resp["reasoning_content"]
        if resp.get("thought_signature"):
            assistant_msg["thought_signature"] = resp["thought_signature"]
        if resp.get("usage"):
            assistant_msg["usage"] = resp["usage"]
        messages.append(assistant_msg)

        content = resp.get("content", "")
        eval_result = _parse_evaluation_json(content)
        if eval_result and "evaluations" in eval_result:
            logger.info(f"[eval_agent] 评分完成，step {steps}")
            if conversation_dir:
                append_messages(
                    conversation_dir=conversation_dir,
                    phase="eval_agent",
                    messages=messages,
                )
            return _build_eval_output(eval_result, rubric, steps, messages)

        # JSON 解析失败，提示模型重新输出
        logger.warning(f"[eval_agent step {steps}] JSON 解析失败，提示重试")
        messages.append({
            "role": "user",
            "content": (
                "Your response did not contain valid JSON with an 'evaluations' key. "
                "Please output ONLY the JSON code block as specified."
            ),
        })

    # 保存最终对话记录
    if conversation_dir:
        append_messages(
            conversation_dir=conversation_dir,
            phase="eval_agent",
            messages=messages,
        )

    logger.warning(f"[eval_agent] max_steps ({max_steps}) reached, evaluation failed")
    evaluations = [
        {
            "rubric_item_id": item.get("rubric_item_id", item.get("id", "")),
            "criterion": item.get("criterion", ""),
            "score": 0,
            "max_score": item.get("score", item.get("max_score", 0)),
            "reasoning": "Failed to parse evaluation output.",
        }
        for item in rubric
    ]
    return {
        "evaluations": evaluations,
        "total_score": 0,
        "total_max_score": sum(item.get("score", item.get("max_score", 0)) for item in rubric),
        "steps": steps,
        "status": "parse_failed",
        "messages": messages,
    }
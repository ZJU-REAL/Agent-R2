"""officeqa solver.py -- Solver Agent: 上传文档，多轮对话，提取 \\boxed{} 答案。"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from llm.router import call_model
from officeqa.prompt import build_solver_system_prompt, build_solver_user_prompt
from officeqa.tool_manager import get_solver_tools, execute_tool, build_tool_message_content
from officeqa.sandbox.file_management import upload_files_to_sandbox

from officeqa.utils import append_messages

logger = logging.getLogger(__name__)


@dataclass
class SolverConfig:
    family: str = "qwen"
    model_id: str = "qwen3.6-plus"
    max_steps: int = 40
    ports: List[int] = field(default_factory=list)
    max_tool_response_length: int = 4096

    def get_url(self, default_url: str = "") -> str:
        if self.ports:
            return ",".join(f"http://localhost:{p}/v1" for p in self.ports)
        return default_url


async def run_solver(
    *,
    task_data: Dict[str, Any],
    session_id: str,
    env: Any,
    config: SolverConfig,
    sk: str,
    api_url: str,
    output_tmp_dir: str,
    assigned_url: str = None,
    conversation_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """上传文档 → 多轮对话 → 提取最终 \\boxed{} 答案。"""
    run_dir = Path(output_tmp_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    doc_local_paths: List[str] = task_data.get("doc_local_paths", [])

    # Step 1: 上传文档文件到沙盒
    sandbox_doc_paths: List[str] = []
    if doc_local_paths:
        local_files = [p for p in doc_local_paths if Path(p).is_file()]
        if local_files:
            local_root = str(Path(local_files[0]).parent)
            try:
                remote_paths = await upload_files_to_sandbox(
                    env=env,
                    session_id=session_id,
                    local_files=local_files,
                    local_root=local_root,
                    remote_root="/mnt/user-data/uploads",
                )
                sandbox_doc_paths = remote_paths
            except Exception as e:
                logger.warning(f"[solver] 上传文档失败: {e}")
                sandbox_doc_paths = [f"/mnt/user-data/uploads/{Path(p).name}" for p in local_files]
        else:
            logger.warning(f"[solver] 文档文件不存在: {doc_local_paths}")
            sandbox_doc_paths = [f"/mnt/user-data/uploads/{Path(p).name}" for p in doc_local_paths]

    # Step 2: 构建 prompt
    system_prompt = build_solver_system_prompt(max_steps=config.max_steps)
    user_prompt = build_solver_user_prompt(
        question=task_data.get("question", ""),
        doc_paths=sandbox_doc_paths,
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    tools = get_solver_tools()
    steps = 0
    completed_early = False
    final_answer = None

    # Step 3: 多轮对话
    while steps < config.max_steps:
        steps += 1
        t0 = time.time()

        if conversation_dir:
            append_messages(conversation_dir=conversation_dir, phase="solver", messages=messages)

        resp = await call_model(
                family=config.family,
                model_id=config.model_id,
                sk=sk,
                messages=messages,
                tools=tools,
                api_url=assigned_url if assigned_url else config.get_url(api_url),
            )

        logger.info(f"[solver step {steps}] model call took {time.time() - t0:.1f}s")

        if resp.get("error"):
            messages.append({"role": "assistant", "content": f"Model error: {resp['error']}"})
            break

        assistant_msg: Dict[str, Any] = {"role": "assistant", "content": resp.get("content", "")}
        if resp.get("tool_calls"):
            assistant_msg["tool_calls"] = resp["tool_calls"]
        if resp.get("reasoning_content"):
            assistant_msg["reasoning_content"] = resp["reasoning_content"]
        if resp.get("thought_signature"):
            assistant_msg["thought_signature"] = resp["thought_signature"]
        if resp.get("usage"):
            assistant_msg["usage"] = resp["usage"]
        messages.append(assistant_msg)
        if conversation_dir:
            append_messages(conversation_dir=conversation_dir, phase="solver", messages=messages)

        tool_calls = resp.get("tool_calls") or []
        if not tool_calls:
            completed_early = True
            logger.info(f"[solver step {steps}] 模型未调用工具，任务完成")
            final_answer = _extract_boxed_from_messages(messages)
            break

        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            call_id = tc["id"]
            raw_args = tc["function"].get("arguments", "{}")
            try:
                tool_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except Exception:
                tool_args = {}

            result = await execute_tool(
                tool_name=tool_name,
                tool_args=tool_args,
                session_id=session_id,
                env=env,
            )

            content = build_tool_message_content(tool_name, result, config.max_tool_response_length)
            messages.append({"role": "tool", "tool_call_id": call_id, "content": content})

    if conversation_dir:
        append_messages(conversation_dir=conversation_dir, phase="solver", messages=messages)

    # Step 4: 若循环结束仍未提取到答案
    if final_answer is None:
        final_answer = _extract_boxed_from_messages(messages)

    return {
        "final_answer": final_answer,
        "sandbox_doc_paths": sandbox_doc_paths,
        "steps": steps,
        "status": "completed" if completed_early else "max_steps_reached",
        "messages": messages,
    }


def _extract_boxed_from_messages(messages: List[Dict[str, Any]]) -> Optional[str]:
    """从最后一条 assistant 消息中提取 <FINAL_ANSWER> 标签内容，未找到则返回原始 content。"""
    from officeqa.reward import extract_final_answer_from_xml

    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "") or ""
            if not content:
                content = msg.get("reasoning_content", "") or ""
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            answer, reasoning = extract_final_answer_from_xml(content)
            # reasoning is None only when no tag was found and the full text was returned
            if reasoning is not None or (answer and answer != content):
                return answer.strip() if answer else None
            return content or None
    return None

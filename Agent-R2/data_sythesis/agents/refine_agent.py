"""data-sythesis agents/refine_agent.py -- 核验与修订Agent。

solver 产出参考交付物后，对照 rubric 检查：不合理的未满足项修订 rubric，
合理的未满足项在沙盒中修订一次交付物。
"""
from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_sythesis.prompts.refine_prompt import (
    build_refine_system_prompt,
    build_refine_user_prompt,
)
from data_sythesis.prompts.retry_prompt import build_json_retry_message
from data_sythesis.utils import append_messages, parse_json_response
from data_sythesis.tool_manager import get_solver_tools, execute_tool, build_tool_message_content
from data_sythesis.sandbox.file_management import (
    download_files_from_sandbox,
    upload_files_to_sandbox,
)

from llm.router import call_model

logger = logging.getLogger(__name__)


@dataclass
class RefineAgentConfig:
    """核验与修订Agent的配置。"""
    family: str
    model_id: str
    max_steps: int = 20
    max_retry: int = 3
    username: str = ""
    reasoning_effort: str = "high"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "model_id": self.model_id,
            "max_steps": self.max_steps,
            "max_retry": self.max_retry,
            "username": self.username,
            "reasoning_effort": self.reasoning_effort,
        }


async def _list_output_files(env: Any, session_id: str) -> List[str]:
    """列出 /mnt/user-data/outputs 目录下的所有文件。"""
    try:
        result = await env.session_exec(
            session_id=session_id,
            command="find /mnt/user-data/outputs -type f",
            exec_dir="/mnt/user-data/outputs",
            combine_output=True,
        )
        output = result.get("output", "")
        files = [f.strip() for f in output.strip().split("\n") if f.strip()]
        return files
    except Exception as e:
        logger.warning(f"[refine_agent] 列出输出文件失败: {e}")
        return []


def _normalize_rubric(rubric: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep rubric fields consistent with question_agent output."""
    normalized: List[Dict[str, Any]] = []
    for item in rubric:
        if not isinstance(item, dict):
            continue
        criterion = item.get("criterion", "")
        if not criterion:
            continue
        normalized.append({
            "criterion": criterion,
            "category": item.get("category", "content"),
            "points": item.get("points", item.get("score", item.get("max_score", 0))),
            "source": item.get("source", ""),
        })
    return normalized


async def run_refine_agent(
    *,
    question_data: Dict[str, Any],
    solver_result: Dict[str, Any],
    local_attachments: List[str],
    session_id: str,
    env: Any,
    output_tmp_dir: str,
    config: RefineAgentConfig,
    sk: str,
    api_url: str,
    conversation_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """运行核验与修订Agent。

    对照 rubric 检查 solver 交付物：不合理的未满足项修订 rubric，
    合理的未满足项在沙盒中修订一次交付物。

    Args:
        question_data: question_agent 产出，包含 question、rubric
        solver_result: solver_agent 产出，包含 output_files
        local_attachments: 输入附件的本地路径列表
        session_id: sandbox 会话 ID
        env: Environment 对象
        output_tmp_dir: 本地临时目录
        config: Agent 配置
        sk: API 密钥
        api_url: API 地址
        conversation_dir: conversation 文件夹路径

    Returns:
        核验结果，包含:
        - rubric: 修订后的 rubric
        - unmet_items: 未满足项及处理方式
        - deliverable_revised: 是否修订了交付物
        - output_files: 最终交付物路径
        - steps / status
    """
    run_dir = Path(output_tmp_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = run_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    question = question_data.get("question", {})
    original_rubric = question_data.get("rubric", [])

    # Step 1: 上传输入附件
    logger.info(f"[refine_agent] 上传 {len(local_attachments)} 个输入附件到沙盒...")
    attachment_paths: List[Dict[str, str]] = []
    if local_attachments:
        local_files: List[Path] = []
        for local_path_str in local_attachments:
            local_path = Path(local_path_str)
            if local_path.is_file():
                local_files.append(local_path)
            elif local_path.is_dir():
                for f in local_path.rglob("*"):
                    if f.is_file():
                        local_files.append(f)

        if local_files:
            if len(local_files) == 1:
                local_root = local_files[0].parent
            else:
                local_root = Path(local_files[0])
                for f in local_files[1:]:
                    while not str(f).startswith(str(local_root)):
                        local_root = local_root.parent
            try:
                remote_paths = await upload_files_to_sandbox(
                    env=env,
                    session_id=session_id,
                    local_files=local_files,
                    local_root=local_root,
                    remote_root="/mnt/user-data/uploads",
                )
                for local_f, remote_f in zip(local_files, remote_paths):
                    attachment_paths.append({
                        "sandbox_path": remote_f,
                        "description": f"输入附件: {local_f.name}",
                    })
                logger.info(f"[refine_agent] 上传附件完成: {len(remote_paths)} 个文件")
            except Exception as e:
                logger.warning(f"[refine_agent] 上传附件失败: {e}")

    # Step 2: 上传 solver 交付物到 outputs
    solver_output = solver_result.get("output_files", solver_result.get("output_file", {}))
    local_deliverables: List[str] = list(solver_output.get("local_paths", []) or [])
    if not local_deliverables:
        local_dir = solver_output.get("local_dir", "")
        if local_dir:
            d = Path(local_dir)
            if d.is_dir():
                local_deliverables = [str(f) for f in d.rglob("*") if f.is_file()]

    logger.info(f"[refine_agent] 上传 {len(local_deliverables)} 个交付物到沙盒 outputs...")
    deliverable_sandbox_paths: List[str] = []
    if local_deliverables:
        deliv_files = [Path(p) for p in local_deliverables if Path(p).is_file()]
        if deliv_files:
            if len(deliv_files) == 1:
                deliv_root = deliv_files[0].parent
            else:
                deliv_root = Path(deliv_files[0])
                for f in deliv_files[1:]:
                    while not str(f).startswith(str(deliv_root)):
                        deliv_root = deliv_root.parent
            try:
                deliverable_sandbox_paths = await upload_files_to_sandbox(
                    env=env,
                    session_id=session_id,
                    local_files=deliv_files,
                    local_root=deliv_root,
                    remote_root="/mnt/user-data/outputs",
                )
                logger.info(f"[refine_agent] 上传交付物完成: {len(deliverable_sandbox_paths)} 个文件")
            except Exception as e:
                logger.warning(f"[refine_agent] 上传交付物失败: {e}")
                deliverable_sandbox_paths = await _list_output_files(env, session_id)
    else:
        logger.warning("[refine_agent] solver 没有本地交付物")

    if not deliverable_sandbox_paths:
        deliverable_sandbox_paths = await _list_output_files(env, session_id)

    # Step 3: 构建初始 prompt
    system_prompt = build_refine_system_prompt(max_steps=config.max_steps)
    user_prompt = build_refine_user_prompt(
        question=question,
        rubric=original_rubric,
        attachment_paths=attachment_paths,
        deliverable_paths=deliverable_sandbox_paths,
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    tools = get_solver_tools()
    steps = 0
    max_steps = config.max_steps
    max_json_retries = config.max_retry
    final_result: Optional[Dict[str, Any]] = None
    completed_early = False

    # Step 4: 多轮对话
    while steps < max_steps:
        steps += 1
        t0 = time.time()

        if conversation_dir:
            append_messages(
                conversation_dir=conversation_dir,
                phase="refine_agent",
                messages=messages,
            )

        resp = await call_model(
            family=config.family,
            model_id=config.model_id,
            sk=sk,
            messages=messages,
            tools=tools,
            api_url=api_url,
            username=config.username,
            reasoning_effort=config.reasoning_effort,
        )

        elapsed = time.time() - t0
        logger.info(f"[refine_agent step {steps}] model call took {elapsed:.1f}s")

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

        tool_calls = resp.get("tool_calls") or []
        if not tool_calls:
            logger.info(f"[refine_agent step {steps}] 模型未调用工具，跳出主循环")
            completed_early = True
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
                output_tmp_dir=str(tmp_dir),
                session_id=session_id,
                env=env,
            )
            content = build_tool_message_content(tool_name, result)
            messages.append({"role": "tool", "tool_call_id": call_id, "content": content})

    if not completed_early:
        logger.warning(f"[refine_agent] max_steps ({max_steps}) reached; requesting final JSON without tools")
        messages.append({"role": "user", "content": (
            "The tool-analysis budget is exhausted. Using only the evidence already collected, "
            "return the final refine JSON in the required schema."
        )})
        resp = await call_model(
            family=config.family,
            model_id=config.model_id,
            sk=sk,
            messages=messages,
            tools=None,
            api_url=api_url,
            username=config.username,
            reasoning_effort=config.reasoning_effort,
        )
        assistant_msg_final: Dict[str, Any] = {"role": "assistant", "content": resp.get("content", "")}
        if resp.get("reasoning_content"):
            assistant_msg_final["reasoning_content"] = resp["reasoning_content"]
        if resp.get("usage"):
            assistant_msg_final["usage"] = resp["usage"]
        messages.append(assistant_msg_final)
        completed_early = True

    if completed_early:
        json_retry_count = 0
        while json_retry_count < max_json_retries:
            if conversation_dir:
                append_messages(
                    conversation_dir=conversation_dir,
                    phase="refine_agent",
                    messages=messages,
                )

            content = messages[-1].get("content", "") if messages[-1].get("role") == "assistant" else ""
            parsed = parse_json_response(content)
            if parsed and isinstance(parsed.get("rubric"), list):
                final_result = parsed
                logger.info("[refine_agent] JSON解析成功")
                break

            json_retry_count += 1
            if json_retry_count >= max_json_retries:
                logger.warning(f"[refine_agent] JSON解析失败达到{max_json_retries}次，停止")
                break
            logger.warning(f"[refine_agent] JSON解析失败，第{json_retry_count}次重试")
            messages.append({"role": "user", "content": build_json_retry_message()})

            resp = await call_model(
                family=config.family,
                model_id=config.model_id,
                sk=sk,
                messages=messages,
                tools=None,
                api_url=api_url,
                username=config.username,
                reasoning_effort=config.reasoning_effort,
            )
            if resp.get("error"):
                break
            assistant_msg_retry: Dict[str, Any] = {"role": "assistant", "content": resp.get("content", "")}
            if resp.get("reasoning_content"):
                assistant_msg_retry["reasoning_content"] = resp["reasoning_content"]
            if resp.get("thought_signature"):
                assistant_msg_retry["thought_signature"] = resp["thought_signature"]
            if resp.get("usage"):
                assistant_msg_retry["usage"] = resp["usage"]
            messages.append(assistant_msg_retry)

    if conversation_dir:
        append_messages(
            conversation_dir=conversation_dir,
            phase="refine_agent",
            messages=messages,
        )

    if not final_result:
        logger.error("[refine_agent] 无法获取有效结果")
        return None

    # Step 5: 下载最终交付物
    output_files = await _list_output_files(env, session_id)
    downloaded_output: List[str] = []
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_files:
        logger.info(f"[refine_agent] 下载产出文件: {len(output_files)} 个文件")
        try:
            downloaded_output = await download_files_from_sandbox(
                env=env,
                session_id=session_id,
                remote_paths=output_files,
                save_dir=output_dir,
            )
            logger.info(f"[refine_agent] 下载完成: {len(downloaded_output)} 个文件")
        except Exception as e:
            logger.warning(f"[refine_agent] 下载产出文件失败: {e}")
    else:
        logger.warning("[refine_agent] /mnt/user-data/outputs 目录下没有找到产出文件")

    refined_rubric = _normalize_rubric(final_result.get("rubric", []))
    if not refined_rubric:
        refined_rubric = _normalize_rubric(original_rubric)

    return {
        "rubric": refined_rubric,
        "unmet_items": final_result.get("unmet_items", []),
        "deliverable_revised": bool(final_result.get("deliverable_revised", False)),
        "output_files": {
            "sandbox_paths": output_files,
            "local_paths": downloaded_output,
            "local_dir": str(output_dir),
        },
        "question_title": question.get("title", ""),
        "steps": steps,
        "status": "completed",
    }

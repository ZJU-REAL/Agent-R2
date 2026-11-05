"""data-sythesis agents/solver_agent.py -- 解题Agent。

这个Agent接收question_agent设计的问题，作为"解题者"完成生产力任务。
需要把skills和问题的输入附件上传到sandbox中，让模型来解题。
对话轨迹将用于构建微调数据集。
多轮对话，使用工具分析文件、创建产出文件，达到最大轮数后结束。
结束后下载产出文件到本地。
"""
from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_sythesis.prompts.solver_prompt import (
    build_solver_system_prompt,
    build_solver_user_prompt,
)
from data_sythesis.utils import append_messages
from data_sythesis.tool_manager import get_solver_tools, execute_tool, build_tool_message_content
from data_sythesis.sandbox.file_management import (
    download_files_from_sandbox,
    upload_files_to_sandbox,
    upload_tgz_and_extract,
    bundle_dir_to_tgz,
)

from llm.router import call_model

logger = logging.getLogger(__name__)


# Skills 目录路径
SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


@dataclass
class SolverAgentConfig:
    """解题Agent的配置。"""
    family: str
    model_id: str
    max_steps: int = 30  # 最多执行30轮
    upload_skills: bool = True  # 是否上传skills到沙盒
    username: str = ""
    reasoning_effort: str = "high"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "model_id": self.model_id,
            "max_steps": self.max_steps,
            "upload_skills": self.upload_skills,
            "username": self.username,
            "reasoning_effort": self.reasoning_effort,
        }


async def _list_output_files(env: Any, session_id: str) -> List[str]:
    """列出 /mnt/user-data/outputs 目录下的所有文件。

    Args:
        env: Environment对象
        session_id: sandbox会话ID

    Returns:
        文件路径列表
    """
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
        logger.warning(f"[solver_agent] 列出输出文件失败: {e}")
        return []


async def run_solver_agent(
    *,
    question_data: Dict[str, Any],
    local_attachments: List[str],
    session_id: str,
    env: Any,
    output_tmp_dir: str,
    config: SolverAgentConfig,
    sk: str,
    api_url: str,
    conversation_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """运行解题Agent。

    多轮对话，使用工具分析文件、创建产出文件。
    最多执行max_steps轮，达到最大轮数后结束。
    对话轨迹将用于构建微调数据集。

    Args:
        question_data: question_agent产出的问题数据
            包含 question, rubric, reference_file, attachments 等
        local_attachments: 输入附件的本地路径列表
        session_id: sandbox会话ID
        env: Environment对象
        output_tmp_dir: 本地临时目录
        config: Agent配置
        sk: API密钥
        api_url: API地址
        conversation_dir: conversation文件夹路径

    Returns:
        解题结果，包含:
        - output_file: 产出文件信息
        - steps: 实际执行的轮数
        - status: 完成状态
        - conversation_dir: 对话记录目录
    """
    run_dir = Path(output_tmp_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = run_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    question = question_data.get("question", {})
    deliverable = question.get("deliverable", {})

    # Step 2: 上传 skills 到沙盒（如果配置允许）
    if config.upload_skills:
        logger.info("[solver_agent] 上传 skills 到沙盒...")
        try:
            skills_tgz = bundle_dir_to_tgz(
                SKILLS_DIR,
                tmp_dir / "skills.tar.gz",
                arcname="skills"
            )
            await upload_tgz_and_extract(
                env=env,
                session_id=session_id,
                local_tgz=skills_tgz,
                remote_tgz="/home/workspace/skills.tar.gz",
                extract_dir="/home/workspace"
            )
            logger.info("[solver_agent] skills 上传完成")
        except Exception as e:
            logger.warning(f"[solver_agent] skills 上传失败: {e}")
    else:
        logger.info("[solver_agent] 跳过 skills 上传 (upload_skills=False)")

    # Step 3: 上传问题输入附件到沙盒
    logger.info(f"[solver_agent] 上传 {len(local_attachments)} 个输入附件到沙盒...")
    attachment_paths = []
    if local_attachments:
        # 收集本地文件路径
        local_files = []
        for local_path_str in local_attachments:
            local_path = Path(local_path_str)
            if local_path.is_file():
                local_files.append(local_path)
            elif local_path.is_dir():
                # 如果是目录，收集目录下的所有文件
                for f in local_path.rglob("*"):
                    if f.is_file():
                        local_files.append(f)

        if local_files:
            # 确定上传根目录（取共同前缀）
            # 简单处理：使用第一个文件的父目录的父目录作为根
            if len(local_files) == 1:
                local_root = local_files[0].parent
            else:
                # 找到共同的父目录
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
                    remote_root="/mnt/user-data/uploads"
                )
                # 构建附件路径信息
                for local_f, remote_f in zip(local_files, remote_paths):
                    attachment_paths.append({
                        "sandbox_path": remote_f,
                        "description": f"输入附件: {local_f.name}",
                    })
                logger.info(f"[solver_agent] 上传附件完成: {len(remote_paths)} 个文件")
            except Exception as e:
                logger.warning(f"[solver_agent] 上传附件失败: {e}")
                # fallback: 使用question中的attachments信息
                for att in question.get("attachments", []):
                    attachment_paths.append({
                        "sandbox_path": att.get("sandbox_path", ""),
                        "description": att.get("description", ""),
                    })
    else:
        # 没有本地附件文件，使用question中的attachments信息
        for att in question.get("attachments", []):
            attachment_paths.append({
                "sandbox_path": att.get("sandbox_path", ""),
                "description": att.get("description", ""),
            })

    # Step 4: 构建初始prompt
    system_prompt = build_solver_system_prompt(max_steps=config.max_steps, upload_skills=config.upload_skills)
    user_prompt = build_solver_user_prompt(
        question=question,
        attachment_paths=attachment_paths,
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    tools = get_solver_tools()
    steps = 0
    max_steps = config.max_steps
    completed_early = False

    # Step 5: 多轮对话
    while steps < max_steps:
        steps += 1
        t0 = time.time()

        # 实时写入对话记录
        if conversation_dir:
            append_messages(
                conversation_dir=conversation_dir,
                phase=f"{config.family}_solver_agent",
                messages=messages
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
        logger.info(f"[solver_agent step {steps}] model call took {elapsed:.1f}s")

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
            # 没有工具调用，模型已完成任务，结束对话
            completed_early = True
            logger.info(f"[solver_agent step {steps}] 模型未调用工具，任务完成，提前结束")
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

    # 保存最终的对话记录
    if conversation_dir:
        append_messages(
            conversation_dir=conversation_dir,
            phase=f"{config.family}_solver_agent",
            messages=messages
        )

    # Step 6: 下载产出文件到本地
    output_files = await _list_output_files(env, session_id)
    downloaded_output: List[str] = []
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_files:
        logger.info(f"[solver_agent] 下载产出文件: {len(output_files)} 个文件")
        try:
            downloaded_output = await download_files_from_sandbox(
                env=env,
                session_id=session_id,
                remote_paths=output_files,
                save_dir=output_dir
            )
            logger.info(f"[solver_agent] 下载完成: {len(downloaded_output)} 个文件")
        except Exception as e:
            logger.warning(f"[solver_agent] 下载产出文件失败: {e}")
    else:
        logger.warning("[solver_agent] /mnt/user-data/outputs 目录下没有找到产出文件")

    return {
        "output_files": {
            "sandbox_paths": output_files,
            "local_paths": downloaded_output,
            "local_dir": str(output_dir),
        },
        "question_title": question.get("title", ""),
        "deliverable": deliverable,
        "steps": steps,
        "status": "completed" if completed_early else "max_steps_reached",
        "conversation_dir": str(conversation_dir) if conversation_dir else "",
    }
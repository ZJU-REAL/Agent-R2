"""data-sythesis agents/question_agent.py -- 问题设计Agent。

这个Agent根据search_agent提交的符合要求的文件和workflow，设计生产力问题。
需要用到沙盒环境和skills。
多轮对话，使用工具分析文件，达到最大轮数后返回JSON结果。
问题必须有附件，结束后需要把问题输入的附件下载下来。
"""
from __future__ import annotations
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_sythesis.prompts.question_prompt import (
    build_question_system_prompt,
    build_question_user_prompt,
)
from data_sythesis.prompts.retry_prompt import build_json_retry_message, build_file_not_found_message
from data_sythesis.utils import append_messages, parse_json_response
from data_sythesis.tool_manager import get_question_tools, execute_tool, build_tool_message_content
from data_sythesis.sandbox.file_management import (
    download_files_from_sandbox,
    upload_files_to_sandbox,
)
from data_sythesis.preprocess import pptx as _preprocess_pptx
from data_sythesis.preprocess import docx as _preprocess_docx
from data_sythesis.preprocess import xlsx as _preprocess_xlsx
from data_sythesis.preprocess import pdf as _preprocess_pdf

from llm.router import call_model

logger = logging.getLogger(__name__)


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
        logger.warning(f"[question_agent] 列出输出文件失败: {e}")
        return []


@dataclass
class QuestionAgentConfig:
    """问题设计Agent的配置。"""
    family: str
    model_id: str
    max_steps: int = 20
    max_retry: int = 3
    only_raw_files: bool = False
    soffice: bool = True
    preprocess_timeout: int = 120

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "model_id": self.model_id,
            "max_steps": self.max_steps,
            "max_retry": self.max_retry,
            "only_raw_files": self.only_raw_files,
            "soffice": self.soffice,
            "preprocess_timeout": self.preprocess_timeout,
        }


async def _preprocess_relevant_files(
    relevant_files_dir: Path,
    relevant_files: List[Dict[str, Any]],
    use_soffice: bool = True,
    timeout: int = 120,
) -> None:
    """对 relevant_files_dir 中每个支持的文件运行预处理。

    只对 highly_relevant 的文件生成预处理内容（content.md + 图片），
    partially_relevant 文件不做预处理，无论是否上传。

    为每个文件在同目录下建立同名子文件夹，将文本内容保存为 content.md，
    图片由 preprocess 模块直接写入该子文件夹。
    上传时 rglob 会自动包含这些子文件夹内容。

    每个文件的预处理有 timeout 秒超时限制，超时则跳过该文件。
    """
    import asyncio

    # 构建高度相关文件名集合，用于过滤
    highly_relevant_names = {
        Path(f.get("sandbox_path", "")).name
        for f in relevant_files
        if f.get("relevance_type") == "highly_relevant"
    }
    _PREPROCESS_MAP = {
        ".pptx": _preprocess_pptx,
        ".docx": _preprocess_docx,
        ".xlsx": _preprocess_xlsx,
        ".pdf": _preprocess_pdf,
    }

    for file_path in sorted(relevant_files_dir.iterdir()):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        module = _PREPROCESS_MAP.get(ext)
        if module is None:
            continue

        # 只预处理高度相关的文件
        if file_path.name not in highly_relevant_names:
            logger.info(f"[question_agent] skip preprocess (not highly_relevant): {file_path.name}")
            continue

        out_dir = relevant_files_dir / file_path.stem
        out_dir.mkdir(exist_ok=True)

        def _do_preprocess():
            if ext == ".xlsx":
                result = module.preprocess(str(file_path))
                parts = [
                    f"## Sheet: {name}\n\n{sheet.get('markdown', '')}"
                    for name, sheet in result.get("sheets", {}).items()
                    if sheet.get("markdown")
                ]
                return "\n\n".join(parts)
            else:
                result = module.preprocess(str(file_path), output_dir=str(out_dir), use_soffice=use_soffice)
                return result.get("text", "")

        try:
            text = await asyncio.wait_for(
                asyncio.to_thread(_do_preprocess),
                timeout=timeout,
            )
            if text:
                (out_dir / "content.md").write_text(text, encoding="utf-8")
            logger.info(f"[question_agent] preprocessed {file_path.name} -> {out_dir.name}/")
        except asyncio.TimeoutError:
            logger.warning(f"[question_agent] preprocess timeout ({timeout}s) for {file_path.name}, skipping")
        except Exception as e:
            logger.warning(f"[question_agent] preprocess failed for {file_path.name}: {e}")


async def run_question_agent(
    *,
    scenario: Dict[str, Any],
    workflow: Dict[str, Any],
    relevant_files: List[Dict[str, Any]],
    relevant_files_dir: str,
    search_info: List[Dict[str, Any]],
    discipline: str,
    title: str,
    session_id: str,
    env: Any,
    output_tmp_dir: str,
    config: QuestionAgentConfig,
    sk: str,
    api_url: str,
    conversation_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """运行问题设计Agent。

    多轮对话，使用工具分析文件。
    最多执行max_steps轮，达到最大轮数后直接返回JSON格式的结果。

    Args:
        scenario: 场景描述（包含title, background等）
        workflow: 工作流描述（包含steps等）
        matched_files: 符合要求的文件列表（包含高度相关和部分相关，带relevance_type）
        relevant_files_dir: search_agent下载的relevant files本地目录路径
        discipline: 学科领域
        title: 职业名称
        session_id: sandbox会话ID
        env: Environment对象
        output_tmp_dir: 本地临时目录
        config: Agent配置
        sk: API密钥
        api_url: API地址
        conversation_dir: conversation文件夹路径

    Returns:
        设计的问题数据，包含:
        - question: 问题描述
        - rubric: 评分标准
        - attachments: 问题的附件（输入文件）
    """
    run_dir = Path(output_tmp_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = run_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: 预处理 relevant files，提取文本和图片存入同名子文件夹
    if relevant_files_dir:
        local_relevant_dir = Path(relevant_files_dir)
        if local_relevant_dir.exists():
            logger.info("[question_agent] 预处理 relevant files...")
            await _preprocess_relevant_files(local_relevant_dir, relevant_files=relevant_files, use_soffice=config.soffice, timeout=config.preprocess_timeout)
        else:
            logger.warning(f"[question_agent] relevant_files_dir 不存在: {relevant_files_dir}")

    # Step 2: 上传 search_agent 下载的 relevant files（含预处理子文件夹）到沙盒
    # only_raw_files=True 时仅上传源文件，跳过预处理子文件夹
    only_raw = config.only_raw_files
    if only_raw:
        logger.info("[question_agent] only_raw_files=True，仅上传源文件，跳过预处理内容")

    relevant_files_upload_dir = "/mnt/user-data/uploads"
    logger.info(f"[question_agent] 上传 relevant files 到沙盒 {relevant_files_upload_dir}...")
    uploaded_relevant_paths: List[Dict[str, str]] = []
    uploaded_file_names: set = set()
    if relevant_files_dir:
        local_relevant_dir = Path(relevant_files_dir)
        if local_relevant_dir.exists():
            # 确定需要上传的顶层文件名集合
            upload_file_names: set = set()
            for f in relevant_files:
                sandbox_path = f.get("sandbox_path", "")
                if sandbox_path:
                    upload_file_names.add(Path(sandbox_path).name)

            # 收集本地文件：源文件受过滤，预处理子文件夹跟随源文件（only_raw=True 时跳过）
            all_local_files = [f for f in local_relevant_dir.rglob("*") if f.is_file()]
            local_files: List[Path] = []
            for lf in all_local_files:
                if lf.parent == local_relevant_dir:
                    # 顶层源文件：按名称过滤
                    if lf.name in upload_file_names:
                        local_files.append(lf)
                elif not only_raw:
                    # 预处理子文件夹内容：父文件夹名对应源文件名（不含扩展名）
                    parent_stem = lf.parent.name
                    if any(Path(n).stem == parent_stem for n in upload_file_names):
                        local_files.append(lf)

            if local_files:
                try:
                    remote_paths = await upload_files_to_sandbox(
                        env=env,
                        session_id=session_id,
                        local_files=local_files,
                        local_root=local_relevant_dir,
                        remote_root=relevant_files_upload_dir,
                    )
                    # 构建上传后的sandbox路径映射
                    for local_f, remote_f in zip(local_files, remote_paths):
                        uploaded_relevant_paths.append({
                            "local_path": str(local_f),
                            "sandbox_path": remote_f,
                        })
                    logger.info(f"[question_agent] relevant files 上传完成: {len(remote_paths)} 个文件")
                except Exception as e:
                    logger.warning(f"[question_agent] relevant files 上传失败: {e}")
        else:
            logger.warning(f"[question_agent] relevant_files_dir 不存在: {relevant_files_dir}")
    else:
        logger.warning("[question_agent] 未提供 relevant_files_dir，跳过上传")

    # 更新 relevant_files 中的 sandbox_path 为实际上传路径
    if uploaded_relevant_paths:
        # 建立文件名到新sandbox_path的映射
        name_to_path = {}
        for item in uploaded_relevant_paths:
            name_to_path[Path(item["local_path"]).name] = item["sandbox_path"]
        for f in relevant_files:
            old_path = f.get("sandbox_path", "")
            old_name = Path(old_path).name if old_path else ""
            if old_name and old_name in name_to_path:
                f["sandbox_path"] = name_to_path[old_name]

    # 记录已上传的文件名（用于验证JSON中的文件是否存在）
    for item in uploaded_relevant_paths:
        uploaded_file_names.add(Path(item["local_path"]).name)

    # 构建初始prompt
    system_prompt = build_question_system_prompt(max_steps=config.max_steps)
    user_prompt = build_question_user_prompt(
        scenario=scenario,
        workflow=workflow,
        matched_files=relevant_files,
        title=title,
        search_info=search_info,
        only_raw_files=config.only_raw_files,
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    tools = get_question_tools()
    steps = 0
    max_steps = config.max_steps
    max_json_retries = config.max_retry
    final_result = None  # 如果模型提前完成，直接存储解析结果
    completed_early = False

    seen_tool_requests = {}
    while steps < max_steps:
        steps += 1
        t0 = time.time()

        # 实时写入对话记录
        if conversation_dir:
            append_messages(
                conversation_dir=conversation_dir,
                phase="question_agent",
                messages=messages
            )

        resp = await call_model(
            family=config.family,
            model_id=config.model_id,
            sk=sk,
            messages=messages,
            tools=tools,
            api_url=api_url
        )

        elapsed = time.time() - t0
        logger.info(f"[question_agent step {steps}] model call took {elapsed:.1f}s")

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
            logger.info(f"[question_agent step {steps}] 模型未调用工具，跳出主循环")
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

        repeated = False
        for tc in tool_calls:
            signature = json.dumps(tc.get("function", {}), sort_keys=True, ensure_ascii=False)
            seen_tool_requests[signature] = seen_tool_requests.get(signature, 0) + 1
            repeated = repeated or seen_tool_requests[signature] >= 3
        if repeated:
            messages.append({
                "role": "user",
                "content": (
                    "You have repeated an identical tool request at least three times. "
                    "Use the results already obtained; do not repeat successful analysis or file copies. "
                    "If the attachments have been inspected and copied, return the final question "
                    "and source-backed rubric JSON in the required schema now. "
                    "Do not invent missing evidence."
                ),
            })

    if not completed_early:
        logger.warning(f"[question_agent] max_steps ({max_steps}) reached; requesting final JSON without tools")
        messages.append({"role": "user", "content": (
            "The tool-analysis budget is exhausted. Using only the attachment evidence already "
            "collected, return the final question and rubric JSON in the required schema. "
            "Do not claim to have inspected or copied files that you have not inspected or copied."
        )})
        resp = await call_model(
            family=config.family, model_id=config.model_id, sk=sk,
            messages=messages, tools=None, api_url=api_url)
        messages.append({"role": "assistant", "content": resp.get("content", "")})
        completed_early = True

    # 统一 JSON 解析 + retry（仅 completed_early 时执行，达到 max_steps 直接 fail）
    if completed_early:
        json_retry_count = 0
        while json_retry_count < max_json_retries:
            if conversation_dir:
                append_messages(
                    conversation_dir=conversation_dir,
                    phase="question_agent",
                    messages=messages
                )

            content = messages[-1].get("content", "") if messages[-1].get("role") == "assistant" else ""
            early_result = parse_json_response(content)

            if not early_result:
                json_retry_count += 1
                if json_retry_count >= max_json_retries:
                    logger.warning(f"[question_agent] JSON解析失败达到{max_json_retries}次，停止")
                    break
                logger.warning(f"[question_agent] JSON解析失败，第{json_retry_count}次重试")
                messages.append({"role": "user", "content": build_json_retry_message()})
            else:
                question_data = early_result.get("question", {})
                attachments = question_data.get("attachments", [])
                json_file_names = {Path(a.get("sandbox_path", "")).name for a in attachments if a.get("sandbox_path")}
                missing = [n for n in json_file_names if n not in uploaded_file_names]
                if missing:
                    json_retry_count += 1
                    if json_retry_count >= max_json_retries:
                        logger.warning(f"[question_agent] 文件不存在达到上限: {missing}")
                        break
                    logger.warning(f"[question_agent] JSON中文件不存在: {missing}，第{json_retry_count}次重试")
                    messages.append({"role": "user", "content": build_file_not_found_message(missing, list(uploaded_file_names))})
                else:
                    final_result = early_result
                    logger.info(f"[question_agent] JSON解析成功")
                    break

            resp = await call_model(
                family=config.family,
                model_id=config.model_id,
                sk=sk,
                messages=messages,
                tools=None,
                api_url=api_url
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

    # 保存最终的对话记录
    if conversation_dir:
        append_messages(
            conversation_dir=conversation_dir,
            phase="question_agent",
            messages=messages
        )

    if not final_result:
        logger.error("[question_agent] 无法获取有效结果")
        return None

    # 下载附件：从 /mnt/user-data/outputs 下载模型已复制的附件到本地 input/
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_files = await _list_output_files(env, session_id)
    downloaded_input: List[str] = []

    if output_files:
        logger.info(f"[question_agent] downloading {len(output_files)} input files from outputs...")
        try:
            downloaded_input = await download_files_from_sandbox(
                env=env,
                session_id=session_id,
                remote_paths=output_files,
                save_dir=input_dir,
            )
            logger.info(f"[question_agent] downloaded {len(downloaded_input)} input files to {input_dir}")
        except Exception as e:
            logger.warning(f"[question_agent] failed to download input files: {e}")
    else:
        logger.warning("[question_agent] /mnt/user-data/outputs 目录下没有找到附件文件")

    return {
        "question": final_result.get("question", {}),
        "rubric": final_result.get("rubric", []),
        "reference_file": final_result.get("reference_file", {}),
        "downloaded_input": downloaded_input,
        "input_dir": str(input_dir),
        "status": "completed" if final_result else "failed",
        "steps": steps,
    }

"""data-sythesis agents/search_agent.py -- 文件搜索Agent。

这个Agent负责多轮搜集附件，每轮搜集后查看内容是否符合场景需要。
最多执行20轮搜索，达到最大轮数后直接返回JSON格式的结果。
"""
from __future__ import annotations
import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

from data_sythesis.prompts.search_prompt import build_search_system_prompt, build_search_user_prompt, build_search_final_result_message
from data_sythesis.prompts.retry_prompt import build_json_retry_message, build_file_not_found_message
from data_sythesis.utils import append_messages, parse_json_response
from data_sythesis.tool_manager import get_search_tools, execute_tool, build_tool_message_content


from llm.router import call_model

logger = logging.getLogger(__name__)


@dataclass
class SearchAgentConfig:
    """搜索Agent的配置。"""
    family: str
    model_id: str
    max_steps: int = 20
    max_retry: int = 3
    sandbox_base_dir: str = "/home/workspace/search_files"
    num: int = 2
    search_mode: str = "async"
    size_limit: int = 2
    page_limit: int = 40
    top: int = 2
    only_high_relevant_files: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "model_id": self.model_id,
            "max_steps": self.max_steps,
            "max_retry": self.max_retry,
            "sandbox_base_dir": self.sandbox_base_dir,
            "num": self.num,
            "search_mode": self.search_mode,
            "size_limit": self.size_limit,
            "page_limit": self.page_limit,
            "top": self.top,
            "only_high_relevant_files": self.only_high_relevant_files,
        }


async def run_search_agent(
    *,
    scenario: Dict[str, Any],
    workflow: Dict[str, Any],
    required_files: List[Any],
    session_id: str,
    env: Any,
    output_tmp_dir: str,
    config: SearchAgentConfig,
    sk: str,
    api_url: str,
    conversation_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """运行文件搜索Agent。

    多轮搜集附件，每轮搜集后查看内容是否符合场景需要。
    最多执行max_steps轮，达到最大轮数后直接返回JSON格式的结果。

    Args:
        scenario: 场景描述（包含title, background等）
        workflow: 工作流描述（包含steps等）
        required_files: 需要的文件列表
        session_id: sandbox会话ID
        env: Environment对象
        output_tmp_dir: 本地临时目录
        config: Agent配置
        sk: API密钥
        api_url: API地址
        conversation_dir: conversation文件夹路径

    Returns:
        搜索结果，包含:
        - highly_relevant_files: 高度相关的文件
        - partially_relevant_files: 部分相关的文件
        - irrelevant_files: 不相关的文件
        - status: completed / max_steps_reached
        - steps: 实际执行的轮数
    """
    run_dir = Path(output_tmp_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = run_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 构建初始prompt
    system_prompt = build_search_system_prompt(max_steps=config.max_steps)
    user_prompt = build_search_user_prompt(
        scenario=scenario,
        workflow=workflow,
        required_files=required_files,
        sandbox_base_dir=config.sandbox_base_dir,
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    tools = get_search_tools()
    steps = 0
    max_steps = config.max_steps
    max_json_retries = config.max_retry

    # 收集结果
    highly_relevant_files: List[Dict[str, Any]] = []
    partially_relevant_files: List[Dict[str, Any]] = []
    irrelevant_files: List[Dict[str, Any]] = []
    completed_early = False

    # 收集搜索元信息（title + snippet）
    search_info: List[Dict[str, Any]] = []

    # 已搜索到的文件名集合（用于验证JSON中的文件是否存在）
    # 从tool执行结果中收集，每次tool返回结果后更新
    searched_file_names: set[str] = set()

    while steps < max_steps:
        steps += 1
        t0 = time.time()

        # 实时写入对话记录
        if conversation_dir:
            append_messages(
                conversation_dir=conversation_dir,
                phase="search_agent",
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
        logger.info(f"[search_agent step {steps}] model call took {elapsed:.1f}s")

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
            # 模型停止了工具调用，跳出循环，后续统一做 JSON 解析
            logger.info(f"[search_agent step {steps}] 模型未调用工具，跳出主循环")
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
                num=config.num,
                search_mode=config.search_mode,
                size_limit=config.size_limit,
                page_limit=config.page_limit,
                top=config.top,
            )

            # 收集搜索元信息
            if isinstance(result, dict) and result.get("status") == "success":
                titles = result.get("title", [])
                snippets = result.get("snippet", [])
                file_paths = result.get("file_path", [])
                urls = result.get("url", [])
                # 基于 file_paths 迭代，安全获取对应的 title、snippet 和 url
                # 某些工具（如 web_search_image）可能没有 snippet
                for i, fp in enumerate(file_paths):
                    if not fp:
                        continue
                    t = titles[i] if i < len(titles) else ""
                    s = snippets[i] if i < len(snippets) else ""
                    u = urls[i] if i < len(urls) else ""
                    search_info.append({
                        "query": tool_args.get("query", ""),
                        "tool": tool_name,
                        "title": t,
                        "snippet": s,
                        "file_path": fp,
                        "url": u,
                    })
                    # 记录已搜索到的文件名
                    searched_file_names.add(Path(fp).name)

            content = build_tool_message_content(tool_name, result)

            messages.append({"role": "tool", "tool_call_id": call_id, "content": content})

    if not completed_early:
        # 达到最大步数，先让模型生成 JSON
        logger.warning(f"[search_agent] max_steps ({max_steps}) reached, requesting final result...")
        messages.append({
            "role": "user",
            "content": build_search_final_result_message()
        })
        resp = await call_model(
            family=config.family,
            model_id=config.model_id,
            sk=sk,
            messages=messages,
            tools=None,
            api_url=api_url
        )
        if not resp.get("error"):
            assistant_msg_final: Dict[str, Any] = {"role": "assistant", "content": resp.get("content", "")}
            if resp.get("reasoning_content"):
                assistant_msg_final["reasoning_content"] = resp["reasoning_content"]
            if resp.get("thought_signature"):
                assistant_msg_final["thought_signature"] = resp["thought_signature"]
            if resp.get("usage"):
                assistant_msg_final["usage"] = resp["usage"]
            messages.append(assistant_msg_final)

    # 统一 JSON 解析 + retry（情况1和情况2共用）
    json_retry_count = 0
    while json_retry_count < max_json_retries:
        if conversation_dir:
            append_messages(
                conversation_dir=conversation_dir,
                phase="search_agent",
                messages=messages
            )

        content = messages[-1].get("content", "") if messages[-1].get("role") == "assistant" else ""
        result = parse_json_response(content)

        if not result:
            json_retry_count += 1
            if json_retry_count >= max_json_retries:
                logger.warning(f"[search_agent] JSON解析失败达到{max_json_retries}次，停止")
                break
            logger.warning(f"[search_agent] JSON解析失败，第{json_retry_count}次重试")
            messages.append({"role": "user", "content": build_json_retry_message()})
        else:
            all_files = (
                result.get("highly_relevant_files", [])
                + result.get("partially_relevant_files", [])
                + result.get("irrelevant_files", [])
            )
            json_file_names = {Path(f.get("sandbox_path", "")).name for f in all_files if f.get("sandbox_path")}
            missing = [n for n in json_file_names if n not in searched_file_names]
            if missing:
                json_retry_count += 1
                if json_retry_count >= max_json_retries:
                    logger.warning(f"[search_agent] 文件不存在达到上限: {missing}")
                    break
                logger.warning(f"[search_agent] JSON中文件不存在: {missing}，第{json_retry_count}次重试")
                messages.append({"role": "user", "content": build_file_not_found_message(missing, list(searched_file_names))})
            else:
                highly_relevant_files = result.get("highly_relevant_files", [])
                partially_relevant_files = result.get("partially_relevant_files", [])
                irrelevant_files = result.get("irrelevant_files", [])
                logger.info(f"[search_agent] JSON解析成功: {len(highly_relevant_files)} highly_relevant, {len(partially_relevant_files)} partially_relevant, {len(irrelevant_files)} irrelevant")
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
            phase="search_agent",
            messages=messages
        )

    # 为高度相关和部分相关的文件添加 relevance_type 字段
    for f in highly_relevant_files:
        f["relevance_type"] = "highly_relevant"
    for f in partially_relevant_files:
        f["relevance_type"] = "partially_relevant"

    # 合并相关文件列表（高度相关 + 部分相关）
    relevant_files = highly_relevant_files + partially_relevant_files

    # 从 tmp 目录拷贝文件到本地，并删除不相关的文件
    # only_high_relevant_files=True 时只拷贝高度相关文件，否则拷贝全部相关文件
    files_to_copy = highly_relevant_files if config.only_high_relevant_files else relevant_files
    downloaded_files: List[str] = []
    if files_to_copy:
        relevant_files_dir = run_dir / "relevant_files"
        relevant_files_dir.mkdir(parents=True, exist_ok=True)

        relevant_names = {
            Path(f.get("sandbox_path", "")).name
            for f in files_to_copy
            if f.get("sandbox_path")
        }

        if tmp_dir.is_dir():
            tmp_file_map: Dict[str, Path] = {}
            for f in tmp_dir.iterdir():
                if f.is_file():
                    tmp_file_map[unquote(f.name)] = f

            for name in relevant_names:
                src = tmp_file_map.get(name)
                if src and src.is_file():
                    dst = relevant_files_dir / name
                    shutil.copy2(str(src), str(dst))
                    downloaded_files.append(str(dst))
                    logger.info(f"[search_agent] copied from tmp: {name}")

            # 删除 tmp 中不相关的文件
            for name, path in tmp_file_map.items():
                if name not in relevant_names:
                    path.unlink()
                    logger.info(f"[search_agent] removed irrelevant tmp file: {name}")

        logger.info(f"[search_agent] copied {len(downloaded_files)} files to {relevant_files_dir}")

    return {
        "highly_relevant_files": highly_relevant_files,
        "partially_relevant_files": partially_relevant_files,
        "irrelevant_files": irrelevant_files,
        "search_info": search_info,
        "downloaded_files": downloaded_files,
        "relevant_files_dir": str(run_dir / "relevant_files") if relevant_files else "",
        "status": "completed" if completed_early else "max_steps_reached",
        "steps": steps,
    }
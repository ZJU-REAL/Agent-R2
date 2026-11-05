"""gdpval solver.py -- Solver Agent: 多轮对话，使用工具分析输入文件并创建产出文件。"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from settings import load_settings

from llm.router import call_model
from gdpval.prompt import (
    build_solver_system_prompt,
    build_solver_user_prompt,
    build_summary_user_prompt,
)
from gdpval.tool_manager import (
    get_solver_tools,
    execute_tool,
    build_tool_message_content,
)
from gdpval.sandbox.file_management import (
    download_files_from_sandbox,
    upload_files_to_sandbox,
    upload_tgz_and_extract,
    bundle_dir_to_tgz,
)
from gdpval.utils import append_messages

logger = logging.getLogger(__name__)

# Skills 目录路径
SKILLS_DIR = Path(__file__).resolve().parent / "skills"


@dataclass
class SolverConfig:
    family: str = "qwen"
    model_id: str = "qwen3.6-plus"
    max_steps: int = 30
    upload_skills: bool = True
    ports: List[int] = None
    max_tool_response_length: int = 4096
    summary: bool = False
    summary_token_threshold: int = 128000

    def get_url(self, default_url: str = "") -> str:
        """根据 ports 拼接多端点 URL，为空则 fallback 到 default_url。"""
        if self.ports:
            return ",".join(f"http://localhost:{p}/v1" for p in self.ports)
        return default_url


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# Solver Agent
# ──────────────────────────────────────────────

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
    """运行 Solver Agent。

    与 data_sythesis/agents/solver_agent.py 逻辑一致，
    但输入来自 benchmark 数据（prompt + reference_files）。

    Args:
        task_data: benchmark 中的任务数据
        session_id: sandbox 会话 ID
        env: Environment 对象
        config: Solver 配置
        sk: API 密钥
        api_url: API 地址
        output_tmp_dir: 本地临时目录

    Returns:
        解题结果，包含 output_file, steps, status, deliverable_sandbox_path
    """
    run_dir = Path(output_tmp_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = run_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    task_prompt = task_data.get("prompt", "")
    reference_files_local = task_data.get("reference_files_local", [])


    # Step 1: 上传 skills 到沙盒（如果配置允许）
    if config.upload_skills:
        logger.info("[solver] 上传 skills 到沙盒...")
        try:
            skills_tgz = bundle_dir_to_tgz(
                SKILLS_DIR,
                tmp_dir / "skills.tar.gz",
                arcname="skills",
            )
            await upload_tgz_and_extract(
                env=env,
                session_id=session_id,
                local_tgz=skills_tgz,
                remote_tgz="/home/workspace/skills.tar.gz",
                extract_dir="/home/workspace",
            )
            logger.info("[solver] skills 上传完成")
        except Exception as e:
            logger.warning(f"[solver] skills 上传失败: {e}")
    else:
        logger.info("[solver] 跳过 skills 上传 (upload_skills=False)")

    # Step 2: 上传 reference_files 到 sandbox
    sandbox_ref_paths: List[str] = []
    if reference_files_local:
        logger.info(f"[solver] 上传 {len(reference_files_local)} 个参考文件到沙盒...")
        local_files = []
        for local_path_str in reference_files_local:
            local_path = Path(local_path_str)
            if local_path.is_file():
                local_files.append(local_path)
            elif local_path.is_dir():
                for f in local_path.rglob("*"):
                    if f.is_file():
                        local_files.append(f)

        if local_files:
            # 找共同父目录
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
                sandbox_ref_paths = remote_paths
                logger.info(f"[solver] 上传完成: {len(remote_paths)} 个文件")
            except Exception as e:
                logger.warning(f"[solver] 上传文件失败: {e}")
                # fallback: 使用相对路径
                for rel in task_data.get("reference_files", []):
                    sandbox_ref_paths.append(f"/mnt/user-data/uploads/{Path(rel).name}")

    # Step 3: 构建初始 prompt
    system_prompt = build_solver_system_prompt(max_steps=config.max_steps, upload_skills=config.upload_skills)
    user_prompt = build_solver_user_prompt(
        task_prompt=task_prompt,
        reference_file_paths=sandbox_ref_paths,
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    tools = get_solver_tools()
    steps = 0
    max_steps = config.max_steps
    completed_early = False

    # Step 4: 多轮对话
    while steps < max_steps:
        steps += 1
        t0 = time.time()

        # 实时写入对话记录
        if conversation_dir:
            append_messages(
                conversation_dir=conversation_dir,
                phase="solver",
                messages=messages,
            )

        resp = await call_model(
            family=config.family,
            model_id=config.model_id,
            sk=sk,
            messages=messages,
            tools=tools,
            api_url=assigned_url if assigned_url else config.get_url(api_url),
        )

        elapsed = time.time() - t0
        logger.info(f"[solver step {steps}] model call took {elapsed:.1f}s")

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

        # ── Summary 压缩：检查 prompt_tokens 是否超阈值 ──
        if config.summary:
            usage = resp.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens", 0)
            if prompt_tokens >= config.summary_token_threshold:
                logger.info(
                    f"[solver step {steps}] prompt_tokens={prompt_tokens} >= {config.summary_token_threshold}，触发 summary 压缩"
                )
                summary_resp = await call_model(
                    family=config.family,
                    model_id=config.model_id,
                    sk=sk,
                    messages=messages + [{"role": "user", "content": build_summary_user_prompt(task_prompt)}],
                    tools=[],
                    api_url=assigned_url if assigned_url else config.get_url(api_url),
                )
                summary_text = summary_resp.get("content", "").strip()
                if summary_text:
                    # 重建 messages：system + user(原始任务) + assistant(summary)
                    messages = [
                        messages[0],  # system
                        messages[1],  # user (原始任务)
                        {"role": "assistant", "content": f"[Progress Summary]\n{summary_text}"},
                    ]
                    logger.info(f"[solver step {steps}] 上下文已压缩，summary 长度={len(summary_text)}")
                else:
                    logger.warning(f"[solver step {steps}] summary 响应为空，跳过压缩")

        tool_calls = resp.get("tool_calls") or []
        if not tool_calls:
            # 没有工具调用，模型已完成任务，结束对话
            completed_early = True
            logger.info(f"[solver step {steps}] 模型未调用工具，任务完成，提前结束")
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

            content = build_tool_message_content(tool_name, result, config.max_tool_response_length)


            messages.append({"role": "tool", "tool_call_id": call_id, "content": content})

    # 保存最终的对话记录
    if conversation_dir:
        append_messages(
            conversation_dir=conversation_dir,
            phase="solver",
            messages=messages,
        )

    # Step 5: 下载产出文件到本地（从 /mnt/user-data/outputs 目录）
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded_output: List[str] = []
    output_files: List[str] = []
    logger.info("[solver] 下载产出文件从 /mnt/user-data/outputs ...")
    try:
        # 列出 outputs 目录下的所有文件
        list_result = await env.session_exec(
                session_id=session_id,
                command="find /mnt/user-data/outputs -type f",
                exec_dir="/mnt/user-data/outputs",
                combine_output=True,
            )
        if list_result.get("output"):
            for line in list_result["output"].strip().split("\n"):
                line = line.strip()
                if line and line.startswith("/mnt/user-data/outputs/"):
                    output_files.append(line)

        if output_files:
            logger.info(f"[solver] 发现 {len(output_files)} 个产出文件: {output_files}")
            downloaded = await download_files_from_sandbox(
                env=env,
                session_id=session_id,
                remote_paths=output_files,
                save_dir=output_dir,
            )
            downloaded_output = downloaded
            logger.info(f"[solver] 下载完成: {len(downloaded)} 个文件")
        else:
            logger.warning("[solver] /mnt/user-data/outputs 目录下没有找到产出文件")
    except Exception as e:
        logger.warning(f"[solver] 下载产出文件失败: {e}")

    return {
        "output_file": {
            "sandbox_paths": output_files,
            "local_paths": downloaded_output,
            "local_dir": str(output_dir),
        },
        "sandbox_ref_paths": sandbox_ref_paths,
        "steps": steps,
        "status": "completed" if completed_early else "max_steps_reached",
        "messages": messages,
    }


# ──────────────────────────────────────────────
# 独立运行入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import asyncio
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    GDPVAL_ROOT = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="gdpval Solver 独立运行")
    parser.add_argument("--task-idx", type=int, default=0, help="任务索引 (默认: 0)")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    args = parser.parse_args()

    # 加载配置
    config_path = args.config or str(GDPVAL_ROOT / "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    sk = load_settings()["llm_api_key"]
    api_url = load_settings()["llm_api_url"]

    solver_cfg = config.get("solver", {})
    sandbox_cfg = config.get("sandbox", {})
    data_cfg = config.get("data", {})

    # 将 sandbox.path 注入环境变量，供 sandbox 子模块动态读取
    sandbox_session_dir = sandbox_cfg.get("path", "")
    if sandbox_session_dir:
        import os
        _p = Path(sandbox_session_dir)
        if not _p.is_absolute():
            _p = GDPVAL_ROOT.parent / _p
        os.environ["SANDBOX_SESSION_DIR"] = str(_p)

    solver_config = SolverConfig(
        family=solver_cfg.get("family", config.get("defaults", {}).get("family", "qwen")),
        model_id=solver_cfg.get("model_id", config.get("defaults", {}).get("model_id", "qwen3.6-plus")),
        max_steps=solver_cfg.get("max_steps", 30),
        upload_skills=solver_cfg.get("upload_skills", True),
        ports=solver_cfg.get("ports") or None,
    )

    # 加载数据
    data_path = data_cfg.get("path", "data/train-00000-of-00001.json")
    if not Path(data_path).is_absolute():
        data_path = str(GDPVAL_ROOT / data_path)
    base_dir = data_cfg.get("base_dir", ".")
    if not Path(base_dir).is_absolute():
        base_dir = str(GDPVAL_ROOT / base_dir)

    from gdpval.utils import load_benchmark_data
    tasks = load_benchmark_data(data_path=data_path, base_dir=base_dir)
    if args.task_idx >= len(tasks):
        logger.error(f"任务索引 {args.task_idx} 超出范围 (共 {len(tasks)} 个)")
        sys.exit(1)

    task = tasks[args.task_idx]
    task_id = task.get("task_id", str(args.task_idx))
    output_dir = args.output_dir or str(GDPVAL_ROOT / "results" / solver_config.model_id / task_id)

    logger.info(f"Solver 任务 {args.task_idx}: {task_id}")

    # 获取 sandbox session
    from rle_open_sdk.sdk import Environment
    from gdpval.sandbox.create import create_sessions

    sandbox_api_key = load_settings()["sandbox_api_key"]
    env = Environment(
        api_key=sandbox_api_key,
        tool_type="aio_sandbox",
        env="test",
        log_file=str(GDPVAL_ROOT / "sandbox" / "sandbox.log"),
        log_level="INFO",
    )

    sids = asyncio.run(create_sessions(1))
    if not sids:
        logger.error("创建 session 失败")
        sys.exit(1)
    session_id = sids[0]

    logger.info(f"使用 session: {session_id}")

    # 运行 Solver
    async def _run():
        conversation_dir = Path(output_dir)
        solver_result = await run_solver(
            task_data=task,
            session_id=session_id,
            env=env,
            config=solver_config,
            sk=sk,
            api_url=api_url,
            output_tmp_dir=output_dir,
            conversation_dir=conversation_dir,
        )
        logger.info(f"Solver 完成: {solver_result.get('status')}, steps={solver_result.get('steps')}")
        output_info = solver_result.get('output_file', {})
        logger.info(f"产出文件 (sandbox): {output_info.get('sandbox_paths', [])}")
        logger.info(f"产出文件 (local): {output_info.get('local_paths', [])}")
        return solver_result

    result = asyncio.run(_run())
    print(f"\nSolver 完成: {result.get('status')}, steps={result.get('steps')}")

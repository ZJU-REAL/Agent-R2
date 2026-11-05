"""sythesis_test eval.py -- Evaluator Agent: 多轮对话，使用工具检查产出文件并按 rubric 逐项打分。"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import asyncio
import sys

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from settings import load_settings

from llm.router import call_model
from sythesis_test.prompt import (
    build_evaluator_system_prompt,
    build_evaluator_user_prompt,
)
from sythesis_test.tool_manager import (
    get_evaluator_tools,
    execute_tool,
    build_tool_message_content,
)
from sythesis_test.sandbox.file_management import (
    upload_files_to_sandbox,
)
from sythesis_test.utils import append_messages, save_result

logger = logging.getLogger(__name__)


@dataclass
class EvaluatorConfig:
    family: str = "qwen"
    model_id: str = "qwen3.6-plus"
    max_steps: int = 50
    upload_skills: bool = False
    ports: List[int] = None
    max_tool_response_length: int = 4096

    def get_url(self, default_url: str = "") -> str:
        if self.ports:
            return ",".join(f"http://localhost:{p}/v1" for p in self.ports)
        return default_url


def _parse_evaluation_json(text: str) -> Optional[Dict[str, Any]]:
    json_block = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_block:
        try:
            return json.loads(json_block.group(1))
        except json.JSONDecodeError:
            pass

    brace_match = re.search(r'\{[\s\S]*\}', text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


async def run_evaluator(
    *,
    task_data: Dict[str, Any],
    solver_result: Dict[str, Any],
    session_id: str,
    env: Any,
    config: EvaluatorConfig,
    sk: str,
    api_url: str,
    output_tmp_dir: str,
    assigned_url: str = None,
    conversation_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """运行 Evaluator Agent。

    与 gdpval.eval.run_evaluator 逻辑一致，主要区别：
    - sythesis_test 无 ground truth deliverable_files，跳过参考交付物上传
    - rubric 已在 utils.load_benchmark_data 中统一为 rubric_item_id/criterion/score 格式
    """
    run_dir = Path(output_tmp_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = run_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    rubric_items = task_data.get("rubric", [])
    local_paths = solver_result.get("output_file", {}).get("local_paths", [])

    # 无交付物直接得 0 分
    local_output_files = [Path(p) for p in local_paths if Path(p).is_file() and Path(p).stat().st_size > 0]
    if not local_output_files:
        logger.info("[evaluator] Solver 无有效交付物，直接得 0 分")
        evaluations = [
            {
                "rubric_item_id": item.get("rubric_item_id", ""),
                "criterion": item.get("criterion", ""),
                "score": 0,
                "max_score": item.get("score", 0),
                "reasoning": "Solver produced no deliverable output.",
            }
            for item in rubric_items
        ]
        return {
            "evaluations": evaluations,
            "total_score": 0,
            "total_max_score": sum(item.get("score", 0) for item in rubric_items),
            "steps": 0,
            "status": "no_deliverable",
            "messages": [],
        }

    # Step 1: 清空 sandbox 工作目录
    logger.info("[evaluator] 清空 sandbox 工作目录 /home/workspace ...")
    try:
        await env.session_exec(
                session_id=session_id,
                command="rm -rf /home/workspace/* /home/workspace/.[!.]*",
                exec_dir="/home/workspace",
                combine_output=True,
            )
    except Exception as e:
        logger.warning(f"[evaluator] 清空工作目录失败: {e}")

    # Step 2: 上传交付物到 sandbox /home/workspace/outputs/
    logger.info(f"[evaluator] 上传 {len(local_output_files)} 个交付物到 sandbox /home/workspace/outputs/ ...")
    local_output_root = local_output_files[0].parent
    evaluator_sandbox_paths: List[str] = []
    try:
        remote_paths = await upload_files_to_sandbox(
            env=env,
            session_id=session_id,
            local_files=local_output_files,
            local_root=local_output_root,
            remote_root="/home/workspace/outputs",
        )
        evaluator_sandbox_paths = remote_paths
        logger.info(f"[evaluator] 交付物上传完成: {remote_paths}")
    except Exception as e:
        logger.warning(f"[evaluator] 上传交付物失败: {e}")
        evaluator_sandbox_paths = [f"/home/workspace/outputs/{f.name}" for f in local_output_files]

    deliverable_sandbox_path = evaluator_sandbox_paths[0] if evaluator_sandbox_paths else ""
    sandbox_ref_paths = solver_result.get("sandbox_ref_paths", [])

    # sythesis_test 没有 ground truth deliverable，跳过参考交付物上传

    # Step 3: 构建 prompt（使用 rubric_pretty 或 rubric_items）
    rubric_pretty = task_data.get("rubric_pretty", "")
    system_prompt = build_evaluator_system_prompt(max_steps=config.max_steps)
    user_prompt = build_evaluator_user_prompt(
        rubric_pretty=rubric_pretty,
        rubric_items=rubric_items,
        deliverable_sandbox_path=deliverable_sandbox_path,
        reference_sandbox_paths=sandbox_ref_paths,
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    tools = get_evaluator_tools()
    steps = 0
    max_steps = config.max_steps

    # 多轮对话
    while steps < max_steps:
        steps += 1
        t0 = time.time()

        if conversation_dir:
            append_messages(
                conversation_dir=conversation_dir,
                phase="evaluator",
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
        logger.info(f"[evaluator step {steps}] model call took {elapsed:.1f}s")

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
            content = resp.get("content", "")
            eval_result = _parse_evaluation_json(content)
            if eval_result and "evaluations" in eval_result:
                logger.info(f"[evaluator] 评估完成，step {steps}")
                if conversation_dir:
                    append_messages(
                        conversation_dir=conversation_dir,
                        phase="evaluator",
                        messages=messages,
                    )
                return _build_eval_output(eval_result, rubric_items, steps, messages)

            # 未解析到有效 JSON，要求模型重新生成
            logger.warning(f"[evaluator] 未能解析到有效的JSON评估结果，step {steps}，要求模型重新生成")
            messages.append({"role": "user", "content": "未能从你的回复中解析到有效的JSON评估结果。请严格按照要求的JSON格式重新输出评估结果。"})
            continue

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

    # 保存最终对话记录
    if conversation_dir:
        append_messages(
            conversation_dir=conversation_dir,
            phase="evaluator",
            messages=messages,
        )

    logger.warning(f"[evaluator] max_steps ({max_steps}) reached, evaluation failed")
    evaluations = [
        {
            "rubric_item_id": item.get("rubric_item_id", ""),
            "criterion": item.get("criterion", ""),
            "score": 0,
            "max_score": item.get("score", 0),
            "reasoning": "Failed to parse evaluation output",
        }
        for item in rubric_items
    ]

    return {
        "evaluations": evaluations,
        "total_score": 0,
        "total_max_score": sum(item.get("score", 0) for item in rubric_items),
        "steps": steps,
        "status": "parse_failed",
        "messages": messages,
    }


def _build_eval_output(
    eval_result: Dict[str, Any],
    rubric_items: List[Dict[str, Any]],
    steps: int,
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    evaluations_raw = eval_result.get("evaluations", [])
    rubric_map = {item.get("rubric_item_id", ""): item for item in rubric_items}

    evaluations = []
    total_score = 0
    for eval_item in evaluations_raw:
        rubric_item_id = eval_item.get("rubric_item_id", "")
        rubric_item = rubric_map.get(rubric_item_id, {})
        criterion = rubric_item.get("criterion", eval_item.get("criterion", ""))
        max_score = rubric_item.get("score", eval_item.get("max_score", 0))
        score = eval_item.get("score", 0)
        reasoning = eval_item.get("reasoning", "")

        evaluations.append({
            "rubric_item_id": rubric_item_id,
            "criterion": criterion,
            "score": score,
            "max_score": max_score,
            "reasoning": reasoning,
        })
        total_score += score

    total_max_score = sum(item.get("score", 0) for item in rubric_items)

    return {
        "evaluations": evaluations,
        "total_score": total_score,
        "total_max_score": total_max_score,
        "steps": steps,
        "status": "completed",
        "messages": messages,
    }


if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    SYTHESIS_ROOT = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="sythesis_test Evaluator 独立运行")
    parser.add_argument("--task-idx", type=int, default=None, help="任务索引")
    parser.add_argument("--task-id", type=str, default=None, help="任务 ID（与 --task-idx 二选一）")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录（需含 solver_result.json）")
    args = parser.parse_args()

    config_path = args.config or str(SYTHESIS_ROOT / "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    sk = load_settings()["llm_api_key"]
    api_url = load_settings()["llm_api_url"]

    evaluator_cfg = config.get("evaluator", {})
    sandbox_cfg = config.get("sandbox", {})
    data_cfg = config.get("data", {})

    sandbox_session_dir = sandbox_cfg.get("path", "")
    if sandbox_session_dir:
        import os
        _p = Path(sandbox_session_dir)
        if not _p.is_absolute():
            _p = SYTHESIS_ROOT.parent / _p
        os.environ["SANDBOX_SESSION_DIR"] = str(_p)

    evaluator_config = EvaluatorConfig(
        family=evaluator_cfg.get("family", config.get("defaults", {}).get("family", "qwen")),
        model_id=evaluator_cfg.get("model_id", config.get("defaults", {}).get("model_id", "qwen3.6-plus")),
        max_steps=evaluator_cfg.get("max_steps", 50),
        upload_skills=evaluator_cfg.get("upload_skills", False),
        ports=evaluator_cfg.get("ports") or None,
    )

    data_path = data_cfg.get("path", None)
    if data_path and not Path(data_path).is_absolute():
        data_path = str(SYTHESIS_ROOT / data_path)

    from sythesis_test.utils import load_benchmark_data
    tasks = load_benchmark_data(data_path=data_path)

    if args.task_id:
        task_id = args.task_id
        task = next((t for t in tasks if t.get("task_id") == task_id), None)
        if task is None:
            logger.error(f"未找到 task_id={task_id}")
            sys.exit(1)
    elif args.task_idx is not None:
        if args.task_idx >= len(tasks):
            logger.error(f"任务索引 {args.task_idx} 超出范围 (共 {len(tasks)} 个)")
            sys.exit(1)
        task = tasks[args.task_idx]
        task_id = task.get("task_id", str(args.task_idx))
    else:
        logger.error("请指定 --task-idx 或 --task-id")
        sys.exit(1)

    results_dir = args.output_dir or str(SYTHESIS_ROOT / "results" / evaluator_config.model_id)
    task_dir = Path(results_dir) / task_id
    solver_result_path = task_dir / "solver_result.json"

    if not solver_result_path.exists():
        logger.error(f"solver_result.json 不存在: {solver_result_path}")
        sys.exit(1)

    with open(solver_result_path, "r", encoding="utf-8") as f:
        solver_result = json.load(f)

    output_dir = str(task_dir)
    logger.info(f"Evaluator 任务: {task_id}")

    from rle_open_sdk.sdk import Environment
    from sythesis_test.sandbox.create import create_sessions

    sandbox_api_key = load_settings()["sandbox_api_key"]
    env = Environment(
        api_key=sandbox_api_key,
        tool_type="aio_sandbox",
        env="test",
        log_file=str(SYTHESIS_ROOT / "sandbox" / "sandbox.log"),
        log_level="INFO",
    )

    sids = asyncio.run(create_sessions(1))
    if not sids:
        logger.error("创建 session 失败")
        sys.exit(1)
    session_id = sids[0]

    logger.info(f"使用 session: {session_id}")

    async def _run():
        conversation_dir = Path(output_dir)
        eval_result = await run_evaluator(
            task_data=task,
            solver_result=solver_result,
            session_id=session_id,
            env=env,
            config=evaluator_config,
            sk=sk,
            api_url=api_url,
            output_tmp_dir=output_dir,
            conversation_dir=conversation_dir,
        )
        logger.info(f"Evaluator 完成: {eval_result.get('status')}, steps={eval_result.get('steps')}")
        logger.info(f"得分: {eval_result.get('total_score', 0)}/{eval_result.get('total_max_score', 0)}")

        result = {
            "task_id": task_id,
            "solver_status": solver_result.get("status"),
            "solver_steps": solver_result.get("steps"),
            "solver_output_sandbox": solver_result.get("output_file", {}).get("sandbox_paths", []),
            "evaluator_status": eval_result.get("status"),
            "evaluator_steps": eval_result.get("steps"),
            "score": eval_result.get("total_score", 0),
            "max_score": eval_result.get("total_max_score", 0),
            "evaluations": eval_result.get("evaluations", []),
        }
        save_result(result, results_dir, task_id)
        return result

    result = asyncio.run(_run())
    print(f"\n评估完成: {result['score']}/{result['max_score']}")

"""sythesis_test run.py -- 评估入口：加载数据、创建沙盒、调度 Solver 和 Evaluator。"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from settings import load_settings

from sythesis_test.utils import load_config, load_benchmark_data, save_result, save_summary
from sythesis_test.solver import SolverConfig, run_solver
from sythesis_test.eval import EvaluatorConfig, run_evaluator
from sythesis_test.sandbox.create import create_sessions
from sythesis_test.sandbox.destroy import destroy_session

logger = logging.getLogger(__name__)


def _is_solver_done(task_dir: Path) -> bool:
    return (task_dir / "solver_result.json").exists()


def _load_result_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"加载结果文件失败 {path}: {e}")
        return None


def _save_solver_result(solver_result: Dict[str, Any], task_dir: Path) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    data = {k: v for k, v in solver_result.items() if k != "messages"}
    with open(task_dir / "solver_result.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_solver_result(task_dir: Path) -> Optional[Dict[str, Any]]:
    result = _load_result_json(task_dir / "solver_result.json")
    if result is None:
        logger.warning(f"solver_result.json 不存在或损坏，需要重新运行 Solver")
    return result


def _cleanup_solver_residual(task_dir: Path) -> None:
    for name in ["output", "tmp"]:
        d = task_dir / name
        if d.exists():
            shutil.rmtree(d)
            logger.info(f"  删除残留目录: {d}")
    for name in ["solver_result.json", "solver_conversation.json"]:
        f = task_dir / name
        if f.exists():
            f.unlink()
            logger.info(f"  删除残留文件: {f}")


def _cleanup_eval_residual(task_dir: Path) -> None:
    for name in ["evaluator_conversation.json", "eval_result.json"]:
        f = task_dir / name
        if f.exists():
            f.unlink()
            logger.info(f"  删除残留文件: {f}")
    tmp_dir = task_dir / "tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
        logger.info(f"  删除残留目录: {tmp_dir}")


async def evaluate_one_task(
    task_data: Dict[str, Any],
    env: Any,
    solver_config: SolverConfig,
    evaluator_config: EvaluatorConfig,
    sk: str,
    api_url: str,
    output_dir: str,
    run_solver_flag: bool = True,
    run_evaluator_flag: bool = True,
    solver_assigned_url: str = None,
    evaluator_assigned_url: str = None,
) -> Dict[str, Any]:
    """评估单个任务：Solver 解题 -> Evaluator 打分。支持断点续跑。"""
    task_id = task_data.get("task_id", "unknown")
    task_output_dir = f"{output_dir}/{task_id}"
    task_dir = Path(task_output_dir)

    solver_result: Optional[Dict[str, Any]] = None
    eval_result: Optional[Dict[str, Any]] = None

    # Phase 1: Solver
    if run_solver_flag:
        solver_done = _is_solver_done(task_dir)

        if solver_done:
            logger.info(f"[{task_id}] Solver 已完成，跳过 Solver，仅运行 Evaluator")
        else:
            logger.info(f"[{task_id}] 从头开始运行 Solver")
            _cleanup_solver_residual(task_dir)

        if solver_done:
            solver_result = _load_solver_result(task_dir)
            if solver_result is None:
                logger.warning(f"[{task_id}] solver_result.json 损坏，重新运行 Solver")
                solver_done = False
                _cleanup_solver_residual(task_dir)
            else:
                logger.info(
                    f"[{task_id}] Solver 结果已加载 (跳过运行): "
                    f"status={solver_result.get('status')}, "
                    f"output={solver_result.get('output_file', {}).get('sandbox_path', 'N/A')}"
                )

        if not solver_done:
            logger.info(f"[{task_id}] === Solver 开始 ===")
            session_id = (await create_sessions(1))[0]
            try:
                solver_result = await run_solver(
                    task_data=task_data,
                    session_id=session_id,
                    env=env,
                    config=solver_config,
                    sk=sk,
                    api_url=api_url,
                    output_tmp_dir=task_output_dir,
                    assigned_url=solver_assigned_url,
                    conversation_dir=task_dir,
                )
            finally:
                await destroy_session(session_id)

            _save_solver_result(solver_result, task_dir)

            logger.info(
                f"[{task_id}] Solver 完成: status={solver_result.get('status')}, "
                f"steps={solver_result.get('steps')}, "
                f"output={solver_result.get('output_file', {}).get('sandbox_paths', [])}"
            )
    else:
        logger.info(f"[{task_id}] 跳过 Solver (agent_run.solver=false)")
        solver_result = _load_solver_result(task_dir)

    # Phase 2: Evaluator
    if run_evaluator_flag:
        eval_done = (task_dir / "eval_result.json").exists()

        if eval_done:
            logger.info(f"[{task_id}] Evaluator 已完成，跳过 Evaluator")
            cached = _load_result_json(task_dir / "eval_result.json")
            if cached is not None:
                return cached
        elif solver_result is None:
            logger.warning(f"[{task_id}] 没有 solver_result，跳过 Evaluator")
        else:
            _cleanup_eval_residual(task_dir)
            logger.info(f"[{task_id}] === Evaluator 开始 ===")
            session_id = (await create_sessions(1))[0]
            try:
                eval_result = await run_evaluator(
                    task_data=task_data,
                    solver_result=solver_result,
                    session_id=session_id,
                    env=env,
                    config=evaluator_config,
                    sk=sk,
                    api_url=api_url,
                    output_tmp_dir=task_output_dir,
                    assigned_url=evaluator_assigned_url,
                    conversation_dir=task_dir,
                )
            finally:
                await destroy_session(session_id)

            logger.info(
                f"[{task_id}] Evaluator 完成: status={eval_result.get('status')}, "
                f"score={eval_result.get('total_score', 0)}/{eval_result.get('total_max_score', 0)}, "
                f"steps={eval_result.get('steps')}"
            )
    else:
        logger.info(f"[{task_id}] 跳过 Evaluator (agent_run.evaluator=false)")

    # 合并结果
    result: Dict[str, Any] = {
        "task_id": task_id,
        "sector": task_data.get("sector", ""),
        "occupation": task_data.get("occupation", ""),
    }
    if solver_result is not None:
        result.update({
            "solver_status": solver_result.get("status"),
            "solver_steps": solver_result.get("steps"),
            "solver_output_sandbox": solver_result.get("output_file", {}).get("sandbox_paths", []),
            "solver_output_local": solver_result.get("output_file", {}).get("local_paths", []),
        })
    if eval_result is not None:
        result.update({
            "evaluator_status": eval_result.get("status"),
            "evaluator_steps": eval_result.get("steps"),
            "score": eval_result.get("total_score", 0),
            "max_score": eval_result.get("total_max_score", 0),
            "evaluations": eval_result.get("evaluations", []),
        })

    if run_evaluator_flag:
        save_result(result, output_dir, task_id)
    return result


async def run_evaluation(config: Dict[str, Any]) -> None:
    """运行完整评估流程。"""
    solver_cfg = config.get("solver", {})
    evaluator_cfg = config.get("evaluator", {})
    run_cfg = config.get("run", {})
    data_cfg = config.get("data", {})
    sandbox_cfg = config.get("sandbox", {})

    # 将 sandbox.path 注入环境变量
    sandbox_session_dir = sandbox_cfg.get("path", "")
    if sandbox_session_dir:
        import os
        _p = Path(sandbox_session_dir)
        if not _p.is_absolute():
            _p = _PROJECT_ROOT / _p
        os.environ["SANDBOX_SESSION_DIR"] = str(_p)

    sk = load_settings()["llm_api_key"]
    api_url = load_settings()["llm_api_url"]

    agent_run_cfg = config.get("agent_run", {})
    run_solver_flag = agent_run_cfg.get("solver", True)
    run_evaluator_flag = agent_run_cfg.get("evaluator", True)
    max_tool_response_length = agent_run_cfg.get("max_tool_response_length", 4096)

    solver_config = SolverConfig(
        family=solver_cfg.get("family", config.get("defaults", {}).get("family", "qwen")),
        model_id=solver_cfg.get("model_id", config.get("defaults", {}).get("model_id", "qwen3.6-plus")),
        max_steps=solver_cfg.get("max_steps", 50),
        upload_skills=solver_cfg.get("upload_skills", False),
        ports=solver_cfg.get("ports") or None,
        max_tool_response_length=max_tool_response_length,
    )
    evaluator_config = EvaluatorConfig(
        family=evaluator_cfg.get("family", config.get("defaults", {}).get("family", "qwen")),
        model_id=evaluator_cfg.get("model_id", config.get("defaults", {}).get("model_id", "qwen3.6-plus")),
        max_steps=evaluator_cfg.get("max_steps", 50),
        upload_skills=evaluator_cfg.get("upload_skills", False),
        ports=evaluator_cfg.get("ports") or None,
        max_tool_response_length=max_tool_response_length,
    )

    limit = run_cfg.get("limit", 10)
    concurrency = run_cfg.get("concurrency", 1)
    output_dir = run_cfg.get("output_dir", "results")

    sythesis_root = Path(__file__).resolve().parent
    if not Path(output_dir).is_absolute():
        output_dir = str(sythesis_root / output_dir)
    output_dir = str(Path(output_dir) / solver_config.model_id)

    # 加载数据
    data_path = data_cfg.get("path", None)
    if data_path and not Path(data_path).is_absolute():
        data_path = str(sythesis_root / data_path)

    logger.info(f"加载数据: {data_path or '默认 test.parquet'}")
    tasks = load_benchmark_data(data_path=data_path, limit=limit if limit > 0 else -1)
    logger.info(f"共加载 {len(tasks)} 个任务")

    if not tasks:
        logger.error("没有任务数据")
        return

    # 统计已完成进度
    solver_done_count = sum(1 for t in tasks if _is_solver_done(Path(output_dir) / t.get("task_id", "")))
    eval_done_count = sum(
        1 for t in tasks
        if (Path(output_dir) / t.get("task_id", "") / "eval_result.json").exists()
    )
    logger.info(f"已完成: Solver {solver_done_count}/{len(tasks)}, Eval {eval_done_count}/{len(tasks)}")

    from rle_open_sdk.sdk import Environment

    sandbox_api_key = load_settings()["sandbox_api_key"]
    env = Environment(
        api_key=sandbox_api_key,
        tool_type="aio_sandbox",
        env="test",
        log_file=str(sythesis_root / "sandbox" / "sandbox.log"),
        log_level="INFO",
    )

    solver_ports: List[int] = solver_config.ports or []
    evaluator_ports: List[int] = evaluator_config.ports or []
    if solver_ports:
        slots_per_port = max(1, concurrency // len(solver_ports))
        logger.info(f"Solver 使用 vLLM ports {solver_ports}，并发数 {concurrency}，每 port {slots_per_port} 个并发槽")

    sem = asyncio.Semaphore(concurrency)

    solver_port_queue: Optional[asyncio.Queue[str]] = None
    if solver_ports:
        solver_port_queue = asyncio.Queue()
        slots_per_port = max(1, concurrency // len(solver_ports))
        for p in solver_ports:
            for _ in range(slots_per_port):
                solver_port_queue.put_nowait(f"http://localhost:{p}/v1")

    evaluator_port_queue: Optional[asyncio.Queue[str]] = None
    if evaluator_ports:
        evaluator_port_queue = asyncio.Queue()
        evaluator_slots_per_port = max(1, concurrency // len(evaluator_ports))
        for p in evaluator_ports:
            for _ in range(evaluator_slots_per_port):
                evaluator_port_queue.put_nowait(f"http://localhost:{p}/v1")

    async def _run_with_sem(idx: int, task: Dict[str, Any]) -> Dict[str, Any]:
        solver_url = await solver_port_queue.get() if solver_port_queue else None
        evaluator_url = await evaluator_port_queue.get() if evaluator_port_queue else None
        async with sem:
            try:
                return await evaluate_one_task(
                    task_data=task,
                    env=env,
                    solver_config=solver_config,
                    evaluator_config=evaluator_config,
                    sk=sk,
                    api_url=api_url,
                    output_dir=output_dir,
                    run_solver_flag=run_solver_flag,
                    run_evaluator_flag=run_evaluator_flag,
                    solver_assigned_url=solver_url,
                    evaluator_assigned_url=evaluator_url,
                )
            except Exception as e:
                logger.exception(f"[{task.get('task_id', idx)}] 评估失败: {e}")
                return {
                    "task_id": task.get("task_id", str(idx)),
                    "error": str(e),
                    "score": 0,
                    "max_score": sum(r.get("score", 0) for r in task.get("rubric", [])),
                }
            finally:
                if solver_port_queue and solver_url:
                    await solver_port_queue.put(solver_url)
                if evaluator_port_queue and evaluator_url:
                    await evaluator_port_queue.put(evaluator_url)

    results = await asyncio.gather(*[_run_with_sem(i, task) for i, task in enumerate(tasks)])

    summary_path = save_summary(results, output_dir)
    logger.info(f"评估完成，汇总结果: {summary_path}")

    sample_scores = []
    for r in results:
        max_score = r.get("max_score", 0)
        score = r.get("score", 0)
        sample_scores.append(score / max_score if max_score > 0 else 0.0)
    accuracy = sum(sample_scores) / len(sample_scores) if sample_scores else 0.0
    logger.info(f"样本数: {len(sample_scores)}, 平均得分率: {accuracy:.4f}")


def main():
    import argparse

    _log_file = Path(__file__).resolve().parent / "run.log"
    _log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(_log_file, encoding="utf-8")],
    )

    parser = argparse.ArgumentParser(description="sythesis_test benchmark 评估")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--limit", type=int, default=None, help="评估任务数量")
    parser.add_argument("--concurrency", type=int, default=None, help="并发数")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.limit is not None:
        config.setdefault("run", {})["limit"] = args.limit
    if args.concurrency is not None:
        config.setdefault("run", {})["concurrency"] = args.concurrency
    if args.output_dir is not None:
        config.setdefault("run", {})["output_dir"] = args.output_dir

    asyncio.run(run_evaluation(config))


if __name__ == "__main__":
    main()

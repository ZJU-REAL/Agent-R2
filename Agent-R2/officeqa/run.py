"""officeqa run.py -- 评估入口：加载数据、创建沙盒、调度 Solver，用 reward.py 打分。"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from settings import load_settings

from officeqa.utils import load_config, load_benchmark_data, save_result, save_summary
from officeqa.solver import SolverConfig, run_solver
from officeqa.sandbox.create import create_sessions
from officeqa.sandbox.destroy import destroy_session

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


async def _run_evaluator(
    task_data: Dict[str, Any],
    solver_result: Dict[str, Any],
    sk: str,
    api_url: str,
    eval_model_id: str,
    eval_family: str,
) -> Dict[str, Any]:
    from officeqa.eval import run_evaluator as llm_eval

    ground_truth = task_data.get("answer", "")
    predicted = solver_result.get("final_answer") or ""

    result = await llm_eval(
        ground_truth=ground_truth,
        predicted=predicted,
        sk=sk,
        api_url=api_url,
        model_id=eval_model_id,
        family=eval_family,
    )
    return {
        "accuracy": result["accuracy"],
        "reason": result.get("reason", ""),
        "ground_truth": ground_truth,
        "predicted": predicted,
        "status": result.get("status", "scored"),
    }


async def evaluate_one_task(
    task_data: Dict[str, Any],
    env: Any,
    solver_config: SolverConfig,
    sk: str,
    api_url: str,
    output_dir: str,
    eval_model_id: str,
    eval_family: str,
    run_solver_flag: bool = True,
    solver_assigned_url: str = None,
) -> Dict[str, Any]:
    task_id = task_data.get("task_id", "unknown")
    task_dir = Path(output_dir) / task_id

    solver_result: Optional[Dict[str, Any]] = None

    # ── Phase 1: Solver ──
    if run_solver_flag:
        solver_done = _is_solver_done(task_dir)

        if solver_done:
            solver_result = _load_result_json(task_dir / "solver_result.json")
            if solver_result is None:
                solver_done = False

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
                    output_tmp_dir=str(task_dir),
                    assigned_url=solver_assigned_url,
                    conversation_dir=task_dir,
                )
            finally:
                await destroy_session(session_id)

            _save_solver_result(solver_result, task_dir)
            logger.info(
                f"[{task_id}] Solver 完成: status={solver_result.get('status')}, "
                f"steps={solver_result.get('steps')}, answer={solver_result.get('final_answer')}"
            )
    else:
        solver_result = _load_result_json(task_dir / "solver_result.json")

    # ── Phase 2: Evaluator（程序化打分）──
    eval_done = (task_dir / "eval_result.json").exists()
    if eval_done:
        cached = _load_result_json(task_dir / "eval_result.json")
        if cached is not None:
            return cached

    if solver_result is None:
        eval_result = {"accuracy": 0.0, "predicted": "", "reason": "", "ground_truth": task_data.get("answer", ""), "status": "no_solver_result"}
    else:
        eval_result = await _run_evaluator(task_data, solver_result, sk, api_url, eval_model_id, eval_family)
        logger.info(
            f"[{task_id}] Eval: accuracy={eval_result['accuracy']}, "
            f"gt={eval_result['ground_truth']!r}, pred={eval_result['predicted']!r}, "
            f"reason={eval_result.get('reason', '')!r}"
        )

    result: Dict[str, Any] = {
        "task_id": task_id,
        "difficulty": task_data.get("difficulty", ""),
        "accuracy": eval_result.get("accuracy", 0.0),
        "ground_truth": eval_result.get("ground_truth", ""),
        "predicted": eval_result.get("predicted", ""),
        "eval_reason": eval_result.get("reason", ""),
        "eval_status": eval_result.get("status"),
    }
    if solver_result is not None:
        result.update({
            "solver_status": solver_result.get("status"),
            "solver_steps": solver_result.get("steps"),
        })

    save_result(result, output_dir, task_id)
    return result


async def run_evaluation(config: Dict[str, Any]) -> None:
    solver_cfg = config.get("solver", {})
    run_cfg = config.get("run", {})
    data_cfg = config.get("data", {})
    sandbox_cfg = config.get("sandbox", {})

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
    max_tool_response_length = agent_run_cfg.get("max_tool_response_length", 4096)

    solver_config = SolverConfig(
        family=solver_cfg.get("family", config.get("defaults", {}).get("family", "qwen")),
        model_id=solver_cfg.get("model_id", config.get("defaults", {}).get("model_id", "qwen3.6-plus")),
        max_steps=solver_cfg.get("max_steps", 40),
        ports=solver_cfg.get("ports") or [],
        max_tool_response_length=max_tool_response_length,
    )

    eval_cfg = config.get("evaluator", {})
    eval_model_id = eval_cfg.get("model_id", config.get("defaults", {}).get("model_id", "qwen3.6-plus"))
    eval_family = eval_cfg.get("family", config.get("defaults", {}).get("family", "qwen"))

    start = run_cfg.get("start", 0)
    end = run_cfg.get("end", -1)
    concurrency = run_cfg.get("concurrency", 8)
    output_dir = run_cfg.get("output_dir", "results")

    officeqa_root = Path(__file__).resolve().parent
    if not Path(output_dir).is_absolute():
        output_dir = str(officeqa_root / output_dir)
    output_dir = str(Path(output_dir) / solver_config.model_id)

    data_path = data_cfg.get("path", "")
    docs_base = data_cfg.get("docs_base", "")

    logger.info(f"加载数据: {data_path}")
    tasks = load_benchmark_data(data_path=data_path, docs_base=docs_base, start=start, end=end)
    logger.info(f"共加载 {len(tasks)} 个任务 (start={start}, end={end})")

    if not tasks:
        logger.error("没有任务数据")
        return

    from rle_open_sdk.sdk import Environment

    env = Environment(
        api_key=load_settings()["sandbox_api_key"],
        tool_type="aio_sandbox",
        env="test",
        log_file=str(officeqa_root / "sandbox" / "sandbox.log"),
        log_level="INFO",
    )

    solver_ports: List[int] = solver_config.ports or []
    solver_port_queue: Optional[asyncio.Queue[str]] = None
    if solver_ports:
        solver_port_queue = asyncio.Queue()
        slots_per_port = max(1, concurrency // len(solver_ports))
        for p in solver_ports:
            for _ in range(slots_per_port):
                solver_port_queue.put_nowait(f"http://localhost:{p}/v1")

    sem = asyncio.Semaphore(concurrency)

    async def _run_with_sem(task: Dict[str, Any]) -> Dict[str, Any]:
        solver_url = await solver_port_queue.get() if solver_port_queue else None
        async with sem:
            try:
                return await evaluate_one_task(
                    task_data=task,
                    env=env,
                    solver_config=solver_config,
                    sk=sk,
                    api_url=api_url,
                    output_dir=output_dir,
                    eval_model_id=eval_model_id,
                    eval_family=eval_family,
                    run_solver_flag=run_solver_flag,
                    solver_assigned_url=solver_url,
                )
            except Exception as e:
                logger.exception(f"[{task.get('task_id')}] 评估失败: {e}")
                return {
                    "task_id": task.get("task_id", ""),
                    "error": str(e),
                    "accuracy": 0.0,
                }
            finally:
                if solver_port_queue and solver_url:
                    await solver_port_queue.put(solver_url)

    results = await asyncio.gather(*[_run_with_sem(task) for task in tasks])

    summary_path = save_summary(results, output_dir)
    logger.info(f"评估完成，汇总结果: {summary_path}")

    acc = sum(r.get("accuracy", 0) for r in results) / len(results) if results else 0
    logger.info(f"Accuracy: {acc:.4f}")


def main():
    import argparse

    _log_file = Path(__file__).resolve().parent / "run.log"
    _log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(_log_file, encoding="utf-8")],
    )

    parser = argparse.ArgumentParser(description="officeqa benchmark 评估")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    if args.start is not None:
        config.setdefault("run", {})["start"] = args.start
    if args.end is not None:
        config.setdefault("run", {})["end"] = args.end
    if args.concurrency is not None:
        config.setdefault("run", {})["concurrency"] = args.concurrency
    if args.output_dir is not None:
        config.setdefault("run", {})["output_dir"] = args.output_dir

    asyncio.run(run_evaluation(config))


if __name__ == "__main__":
    main()

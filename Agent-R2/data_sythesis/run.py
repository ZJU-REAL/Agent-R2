"""data-sythesis run.py -- 主入口，协调整个数据合成流程。

流程: scenario_agent -> search_agent -> question_agent -> solver_agent -> refine_agent
通过 run.py 运行时输出到 output 目录，单独运行各 agent 时输出到 test 目录。
每个需要 sandbox 的 agent 在运行前会清空 sandbox。
"""
from __future__ import annotations
import argparse
import asyncio
import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import yaml

# 添加项目根目录到 sys.path
_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root.parent))
from settings import load_settings

from utils import (
    load_sample_data,
    load_task_statements,
    load_discipline_to_title,
    build_title_to_tasks_mapping,
    map_discipline_to_title,
    save_output,
    PROJECT_ROOT,
)
from agents.scenario_agent import run_scenario_agent, ScenarioAgentConfig
from agents.search_agent import run_search_agent, SearchAgentConfig
from agents.question_agent import run_question_agent, QuestionAgentConfig
from agents.solver_agent import run_solver_agent, SolverAgentConfig
from agents.refine_agent import run_refine_agent, RefineAgentConfig

from rle_open_sdk.sdk import Environment
from data_sythesis.sandbox.create import create_sessions
from data_sythesis.sandbox.destroy import destroy_session

logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_SESSION_DIR = PROJECT_ROOT / "sandbox" / "sandbox_sessions" / "checked"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class AgentConfigs:
    """所有 agent 的模型配置。"""
    scenario_agent: ScenarioAgentConfig
    search_agent: SearchAgentConfig
    question_agent: QuestionAgentConfig
    solver_agent: SolverAgentConfig
    refine_agent: RefineAgentConfig

    @classmethod
    def from_dict(cls, data: Dict[str, Any], defaults: Dict[str, Any]) -> "AgentConfigs":
        """从字典创建配置，使用 defaults 填充缺失值。"""
        agents = data.get("agents", {})
        fs = data.get("file_search", {})

        def _family(name: str) -> str:
            return agents.get(name, {}).get("family", defaults.get("family", "qwen"))

        def _model_id(name: str) -> str:
            return agents.get(name, {}).get("model_id", defaults.get("model_id", "qwen3.6-plus"))

        def _max_steps(name: str, fallback: int) -> int:
            return agents.get(name, {}).get("max_steps", defaults.get("max_steps", fallback))

        def _max_retry(name: str) -> int:
            return agents.get(name, {}).get("max_retry", defaults.get("max_retry", 3))

        def _upload_skills(name: str) -> bool:
            return agents.get(name, {}).get("upload_skills", True)

        return cls(
            scenario_agent=ScenarioAgentConfig(
                family=_family("scenario_agent"),
                model_id=_model_id("scenario_agent"),
                max_retry=_max_retry("scenario_agent"),
            ),
            search_agent=SearchAgentConfig(
                family=_family("search_agent"),
                model_id=_model_id("search_agent"),
                max_steps=_max_steps("search_agent", 20),
                max_retry=_max_retry("search_agent"),
                sandbox_base_dir=fs.get("sandbox_base_dir", "/home/workspace/search_files"),
                num=fs.get("num", 2),
                search_mode=fs.get("search_mode", "sync"),
                size_limit=fs.get("size_limit", 1),
                page_limit=fs.get("page_limit", 40),
                top=fs.get("top", 5),
                only_high_relevant_files=agents.get("search_agent", {}).get("only_high_relevant_files", False),
            ),
            question_agent=QuestionAgentConfig(
                family=_family("question_agent"),
                model_id=_model_id("question_agent"),
                max_steps=_max_steps("question_agent", 20),
                max_retry=_max_retry("question_agent"),
                only_raw_files=agents.get("question_agent", {}).get("only_raw_files", False),
                soffice=agents.get("question_agent", {}).get("soffice", True),
                preprocess_timeout=agents.get("question_agent", {}).get("preprocess_timeout", 120),
            ),
            solver_agent=SolverAgentConfig(
                family=_family("solver_agent"),
                model_id=_model_id("solver_agent"),
                max_steps=_max_steps("solver_agent", 30),
                upload_skills=_upload_skills("solver_agent"),
            ),
            refine_agent=RefineAgentConfig(
                family=_family("refine_agent"),
                model_id=_model_id("refine_agent"),
                max_steps=_max_steps("refine_agent", 20),
                max_retry=_max_retry("refine_agent"),
            ),
        )


def load_config(config_path: Optional[Path]) -> Dict[str, Any]:
    """加载配置文件。"""
    if config_path and config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None):
    """配置日志。"""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def create_env() -> Environment:
    """创建sandbox环境。"""
    return Environment(
        api_key=load_settings()["sandbox_api_key"],
        tool_type="aio_sandbox",
        env="test",
        log_file="sandbox.log",
        log_level="INFO"
    )


async def create_session_wrapper() -> str:
    """创建单个 session。"""
    return (await create_sessions(1))[0]


async def destroy_session_wrapper(session_id: str):
    """销毁单个 session。"""
    await destroy_session(session_id)


def get_solver_prefix(config: Dict[str, Any]) -> str:
    """从配置中读取 solver_agent 的 family，返回文件前缀，如 'gemini'。"""
    family = config.get("agents", {}).get("solver_agent", {}).get(
        "family", config.get("defaults", {}).get("family", "")
    )
    return family


def build_agent_steps(solver_prefix: str) -> List[Dict[str, Any]]:
    """根据 solver family 前缀构建 AGENT_STEPS。"""
    result_file = f"{solver_prefix}_solver_result.json" if solver_prefix else "solver_result.json"
    tmp_dir = f"{solver_prefix}_solver_tmp" if solver_prefix else "solver_tmp"
    return [
        {"name": "scenario", "result_file": "scenario_result.json", "tmp_dir": None},
        {"name": "search",   "result_file": "search_result.json",   "tmp_dir": "search_tmp"},
        {"name": "question", "result_file": "question_result.json", "tmp_dir": "question_tmp"},
        {"name": "solver",   "result_file": result_file,            "tmp_dir": tmp_dir},
        {"name": "refine",   "result_file": "refine_result.json",   "tmp_dir": "refine_tmp"},
    ]


# Agent 运行顺序（与 AGENT_STEPS 对应）
AGENT_ORDER = ["scenario_agent", "search_agent", "question_agent", "solver_agent", "refine_agent"]


def get_enabled_agents(agent_run: dict) -> set:
    """根据 agent_run 配置返回启用的 agent 名称集合。"""
    return {name for name in AGENT_ORDER if agent_run.get(name, False)}


def _load_result_json(path: Path) -> Optional[Dict[str, Any]]:
    """加载结果JSON文件。"""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"加载结果文件失败 {path}: {e}")
        return None




async def process_one_sample(
    *,
    sample: Dict[str, Any],
    sample_idx: int,
    title_to_tasks: Dict[str, List[str]],
    discipline_to_title: Dict[str, str],
    output_dir: Path,
    agent_configs: AgentConfigs,
    sk: str,
    api_url: str,
    enabled_agents: set,
    solver_prefix: str = "",
) -> Dict[str, Any]:
    """处理单个样本的完整流程。

    流程：scenario -> search -> question -> solver -> refine

    各 agent 按需独立运行：仅当该 agent 在 enabled_agents 中、且对应结果文件不存在时才执行。
    每个需要 sandbox 的 agent 都会新建和销毁环境。
    """
    discipline = sample.get("discipline", "")
    original_document = sample.get("original_document", "")

    # 创建输出目录
    sample_output_dir = output_dir / f"sample_{sample_idx}"
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    conversation_dir = sample_output_dir / "conversation"

    result = {
        "sample_idx": sample_idx,
        "discipline": discipline,
        "status": "processing",
        "output_dir": str(sample_output_dir),
    }

    # ── 加载已有结果文件 ──
    scenario_result_path  = sample_output_dir / "scenario_result.json"
    search_result_path    = sample_output_dir / "search_result.json"
    question_result_path  = sample_output_dir / "question_result.json"
    _solver_result_name   = f"{solver_prefix}_solver_result.json" if solver_prefix else "solver_result.json"
    _solver_tmp_name      = f"{solver_prefix}_solver_tmp" if solver_prefix else "solver_tmp"
    solver_result_path    = sample_output_dir / _solver_result_name
    refine_result_path    = sample_output_dir / "refine_result.json"

    scenario_data   = _load_result_json(scenario_result_path)
    search_result   = _load_result_json(search_result_path)
    question_result = _load_result_json(question_result_path)
    solver_result   = _load_result_json(solver_result_path)
    refine_result   = _load_result_json(refine_result_path)

    # title 仅依赖输入数据，始终可计算
    title = map_discipline_to_title(discipline, discipline_to_title)

    try:
        # 初始化变量（当某些 agent 被禁用时，这些变量保持默认值）
        scenario = {}
        workflow = {}
        required_files = []
        relevant_files = []
        relevant_files_dir = ""

        # ── Step 1: 场景生成 ──
        if "scenario_agent" in enabled_agents and not scenario_result_path.exists():
            if not title:
                result["status"] = "error"
                result["error"] = f"未找到学科 {discipline} 对应的职业"
                return result

            candidate_tasks = title_to_tasks.get(title, [])
            if not candidate_tasks:
                result["status"] = "error"
                result["error"] = f"未找到职业 {title} 对应的任务"
                return result

            scenario_data = await run_scenario_agent(
                original_document=original_document,
                discipline=discipline,
                title=title,
                candidate_tasks=candidate_tasks,
                config=agent_configs.scenario_agent,
                sk=sk,
                api_url=api_url,
                conversation_dir=conversation_dir,
            )

            if not scenario_data:
                result["status"] = "error"
                result["error"] = "场景生成失败"
                return result

            save_output(scenario_data, scenario_result_path)

        # 从 scenario_data 提取后续步骤需要的变量
        if scenario_data is not None:
            # workflow 为 false 表示该语料不适合出题，跳过后续所有步骤
            if not scenario_data.get("workflow", True):
                logger.info(f"[Sample {sample_idx}] scenario_agent 判断该语料不适合出题 (workflow=false)，跳过搜索和出题")
                result["status"] = "skipped"
                result["skip_reason"] = "workflow=false"
                result["summary"] = scenario_data.get("summary", "")
                return result

            required_files = scenario_data.get("required_files", [])
            scenario = scenario_data.get("scenario", {})
            workflow = scenario.get("workflow", {})

        # ── Step 2: 文件搜索 ──
        if "search_agent" in enabled_agents and not search_result_path.exists():
            session_id = await create_session_wrapper()
            env = create_env()
            try:
                search_result = await run_search_agent(
                    scenario=scenario,
                    workflow=workflow,
                    required_files=required_files,
                    session_id=session_id,
                    env=env,
                    output_tmp_dir=str(sample_output_dir / "search_tmp"),
                    config=agent_configs.search_agent,
                    sk=sk,
                    api_url=api_url,
                    conversation_dir=conversation_dir,
                )
            finally:
                await destroy_session_wrapper(session_id)
                del env

            save_output(search_result, search_result_path)
            highly = search_result.get("highly_relevant_files", [])
            partially = search_result.get("partially_relevant_files", [])
            relevant_files = highly + partially
            relevant_files_dir = search_result.get("relevant_files_dir", "")
            search_info = search_result.get("search_info", [])
            logger.info(f"[Sample {sample_idx}] 文件搜索完成，相关文件: {len(relevant_files)} (高度相关: {len(highly)}, 部分相关: {len(partially)})")

            downloaded_files = search_result.get("downloaded_files", [])
            if not downloaded_files:
                result["status"] = "skipped"
                result["skip_reason"] = "no downloaded_files"
                return result
        elif search_result_path.exists():
            highly = search_result.get("highly_relevant_files", [])
            partially = search_result.get("partially_relevant_files", [])
            relevant_files = highly + partially
            relevant_files_dir = search_result.get("relevant_files_dir", "")
            search_info = search_result.get("search_info", [])
            downloaded_files = search_result.get("downloaded_files", [])
            if not downloaded_files:
                logger.info(f"[Sample {sample_idx}] 没有可用文件，跳过问题设计")
                result["status"] = "skipped"
                result["skip_reason"] = "no downloaded_files"
                return result

        # ── Step 3: 问题设计 ──
        if "question_agent" in enabled_agents and not question_result_path.exists():
            session_id = await create_session_wrapper()
            env = create_env()
            try:
                question_result = await run_question_agent(
                    scenario=scenario,
                    workflow=workflow,
                    relevant_files=relevant_files,
                    relevant_files_dir=relevant_files_dir,
                    search_info=search_info,
                    discipline=discipline,
                    title=title,
                    session_id=session_id,
                    env=env,
                    output_tmp_dir=str(sample_output_dir / "question_tmp"),
                    config=agent_configs.question_agent,
                    sk=sk,
                    api_url=api_url,
                    conversation_dir=conversation_dir,
                )
            finally:
                await destroy_session_wrapper(session_id)
                del env

            if not question_result:
                result["status"] = "error"
                result["error"] = "问题设计失败"
                return result

            save_output(question_result, question_result_path)

        # ── Step 4: 解题 ──
        if "solver_agent" in enabled_agents and not solver_result_path.exists():
            if not question_result:
                raise RuntimeError(f"[Sample {sample_idx}] solver_agent 已启用但 question_result.json 不存在")
            # 获取本地输入附件路径
            downloaded_input = question_result.get("downloaded_input", [])
            input_dir = question_result.get("input_dir", "")
            local_attachments = downloaded_input if downloaded_input else []
            if not local_attachments and input_dir:
                local_attachments = [input_dir]

            session_id = await create_session_wrapper()
            env = create_env()
            try:
                solver_result = await run_solver_agent(
                    question_data=question_result,
                    local_attachments=local_attachments,
                    session_id=session_id,
                    env=env,
                    output_tmp_dir=str(sample_output_dir / _solver_tmp_name),
                    config=agent_configs.solver_agent,
                    sk=sk,
                    api_url=api_url,
                    conversation_dir=conversation_dir,
                )
            finally:
                await destroy_session_wrapper(session_id)
                del env

            if not solver_result:
                result["status"] = "error"
                result["error"] = "解题失败"
                return result

            save_output(solver_result, solver_result_path)

        # ── Step 5: 核验与修订 ──
        if "refine_agent" in enabled_agents and not refine_result_path.exists():
            if not question_result:
                raise RuntimeError(f"[Sample {sample_idx}] refine_agent 已启用但 question_result.json 不存在")
            if not solver_result:
                raise RuntimeError(f"[Sample {sample_idx}] refine_agent 已启用但 solver_result.json 不存在")

            downloaded_input = question_result.get("downloaded_input", [])
            input_dir = question_result.get("input_dir", "")
            local_attachments = downloaded_input if downloaded_input else []
            if not local_attachments and input_dir:
                local_attachments = [input_dir]

            session_id = await create_session_wrapper()
            env = create_env()
            try:
                refine_result = await run_refine_agent(
                    question_data=question_result,
                    solver_result=solver_result,
                    local_attachments=local_attachments,
                    session_id=session_id,
                    env=env,
                    output_tmp_dir=str(sample_output_dir / "refine_tmp"),
                    config=agent_configs.refine_agent,
                    sk=sk,
                    api_url=api_url,
                    conversation_dir=conversation_dir,
                )
            finally:
                await destroy_session_wrapper(session_id)
                del env

            if not refine_result:
                result["status"] = "error"
                result["error"] = "核验与修订失败"
                return result

            save_output(refine_result, refine_result_path)
            if refine_result.get("rubric"):
                question_result = dict(question_result)
                question_result["rubric"] = refine_result["rubric"]
                save_output(question_result, question_result_path)

        # ── 保存最终汇总 ──
        final_result = {
            "task_id": str(uuid.uuid4()),
            "created_at": datetime.now().isoformat(),
            "discipline": discipline,
            "title": title,
            "scenario": scenario,
            "workflow": workflow,
            "required_files": required_files,
        }
        if search_result is not None:
            final_result["highly_relevant_files"] = search_result.get("highly_relevant_files", [])
            final_result["partially_relevant_files"] = search_result.get("partially_relevant_files", [])
            final_result["irrelevant_files"] = search_result.get("irrelevant_files", [])
        if question_result is not None:
            final_result["question"] = question_result.get("question", {})
            final_result["rubric"] = question_result.get("rubric", [])
            final_result["reference_file"] = question_result.get("reference_file", {})
        if solver_result is not None:
            final_result["solver_output_file"] = solver_result.get("output_files", solver_result.get("output_file", {}))
        if refine_result is not None:
            final_result["rubric"] = refine_result.get("rubric", final_result.get("rubric", []))
            final_result["refine_output_file"] = refine_result.get("output_files", {})
            final_result["deliverable_revised"] = refine_result.get("deliverable_revised", False)
        final_result["metadata"] = {"sample_idx": sample_idx}
        if search_result is not None:
            final_result["metadata"]["search_steps"] = search_result.get("steps")
        if question_result is not None:
            final_result["metadata"]["question_steps"] = question_result.get("steps")
        if solver_result is not None:
            final_result["metadata"]["solver_steps"] = solver_result.get("steps")
        if refine_result is not None:
            final_result["metadata"]["refine_steps"] = refine_result.get("steps")

        save_output(final_result, sample_output_dir / "final_result.json")
        result["status"] = "completed"
        result["output_file"] = str(sample_output_dir / "final_result.json")
        logger.info(f"[Sample {sample_idx}] 全部完成")

    except Exception as e:
        logger.exception(f"[Sample {sample_idx}] 处理失败")
        result["status"] = "error"
        result["error"] = str(e)

    return result


async def run_synthesis_pipeline(
    *,
    begin: Optional[int],
    end: Optional[int],
    output_dir: Path,
    agent_configs: AgentConfigs,
    sk: str,
    api_url: str,
    enabled_agents: set,
    solver_prefix: str = "",
    concurrency: int = 10,
) -> List[Dict[str, Any]]:
    """运行数据合成流水线。"""
    # 加载映射
    task_df = load_task_statements()
    title_to_tasks = build_title_to_tasks_mapping(task_df)
    discipline_to_title = load_discipline_to_title()
    logger.info(f"加载了 {len(title_to_tasks)} 个职业的任务描述")

    upstream_agents = {"scenario_agent", "search_agent", "question_agent"}
    need_jsonl = bool(enabled_agents & upstream_agents)

    if need_jsonl:
        logger.info("加载种子数据...")
        samples_with_idx = [
            (i + (begin or 0), s)
            for i, s in enumerate(load_sample_data(begin=begin, end=end))
        ]
        logger.info(f"加载了 {len(samples_with_idx)} 个样本")
    else:
        # 从 output 目录扫描已有 sample，读取 discipline 从 scenario_result.json
        import re
        all_dirs = sorted(output_dir.glob("sample_*"), key=lambda p: int(p.name.split("_")[1]))
        samples_with_idx = []
        for d in all_dirs:
            m = re.fullmatch(r"sample_(\d+)", d.name)
            if not m:
                continue
            idx = int(m.group(1))
            if begin is not None and idx < begin:
                continue
            if end is not None and idx >= end:
                continue
            scenario_path = d / "scenario_result.json"
            discipline = ""
            if scenario_path.exists():
                with open(scenario_path) as f:
                    discipline = json.load(f).get("discipline", "")
            samples_with_idx.append((idx, {"discipline": discipline, "original_document": ""}))
        logger.info(f"加载了 {len(samples_with_idx)} 个样本")

    sem = asyncio.Semaphore(concurrency)

    async def _process(sample_idx: int, sample: Dict[str, Any]):
        async with sem:
            return await process_one_sample(
                sample=sample,
                sample_idx=sample_idx,
                title_to_tasks=title_to_tasks,
                discipline_to_title=discipline_to_title,
                output_dir=output_dir,
                agent_configs=agent_configs,
                sk=sk,
                api_url=api_url,
                enabled_agents=enabled_agents,
                solver_prefix=solver_prefix,
            )

    results = await asyncio.gather(*[_process(idx, s) for idx, s in samples_with_idx])
    return list(results)


def main():
    parser = argparse.ArgumentParser(description="GDPval数据合成流水线")
    parser.add_argument("--config", type=Path, default=None, help="配置文件路径 (默认: config.yaml)")
    parser.add_argument("--begin", type=int, default=None, help="起始索引（包含）")
    parser.add_argument("--end", type=int, default=None, help="结束索引（不包含）")
    parser.add_argument("--output", type=Path, default=None, help="输出目录")
    parser.add_argument("--family", type=str, default=None, help="默认模型类型")
    parser.add_argument("--model-id", type=str, default=None, help="默认模型ID")
    parser.add_argument("--sk", type=str, default=None, help="API密钥")
    parser.add_argument("--api-url", type=str, default=None, help="API地址")
    parser.add_argument("--max-steps", type=int, default=None, help="默认最大步数")
    parser.add_argument("--concurrency", type=int, default=None, help="并发数")
    parser.add_argument("--log-level", type=str, default=None, help="日志级别")
    args = parser.parse_args()

    # 加载配置文件
    config_path = args.config or DEFAULT_CONFIG_PATH
    config = load_config(config_path)

    # 合并配置：命令行参数 > 配置文件 > 默认值
    def get_value(key: str, config_section: str = None):
        cli_val = getattr(args, key.replace("-", "_"), None)
        if cli_val is not None:
            return cli_val
        if config_section:
            return config.get(config_section, {}).get(key)
        return config.get(key)

    # 解析默认配置
    defaults = config.get("defaults", {})
    default_family = get_value("family") or defaults.get("family", "qwen")
    default_model_id = get_value("model_id") or defaults.get("model_id", "qwen3.6-plus")
    default_max_steps = get_value("max_steps") or defaults.get("max_steps", 50)

    # 构建 agent 配置
    agent_configs = AgentConfigs.from_dict(config, {
        "family": default_family,
        "model_id": default_model_id,
        "max_steps": default_max_steps,
    })

    # 其他配置
    sk = args.sk or load_settings()["llm_api_key"]
    api_url = args.api_url or load_settings()["llm_api_url"]
    run_config = config.get("run", {})
    begin = args.begin if args.begin is not None else run_config.get("begin")
    end = args.end if args.end is not None else run_config.get("end")
    output_dir = args.output or Path(run_config.get("output_dir", str(DEFAULT_OUTPUT_DIR)))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    concurrency = args.concurrency if args.concurrency is not None else config.get("run", {}).get("concurrency", 1)
    log_level = get_value("log_level") or "INFO"

    log_file = Path(__file__).resolve().parent / "run.log"
    setup_logging(log_level.upper(), log_file=log_file)

    # 解析 agent_run 配置
    agent_run = config.get("agent_run", {})
    enabled_agents = get_enabled_agents(agent_run)

    # 计算 solver 文件前缀
    solver_prefix = get_solver_prefix(config)
    if solver_prefix:
        logger.info(f"Solver 文件前缀: {solver_prefix} (solver_result -> {solver_prefix}_solver_result.json)")

    # 运行流水线
    results = asyncio.run(run_synthesis_pipeline(
        begin=begin,
        end=end,
        output_dir=output_dir,
        agent_configs=agent_configs,
        sk=sk,
        api_url=api_url,
        enabled_agents=enabled_agents,
        solver_prefix=solver_prefix,
        concurrency=concurrency,
    ))

    # 统计结果
    completed = sum(1 for r in results if r.get("status") == "completed")
    errors = sum(1 for r in results if r.get("status") == "error")

    logger.info(f"\n处理完成: {completed}/{len(results)}")
    logger.info(f"失败: {errors}")

    # 保存汇总
    summary_path = output_dir / "summary.json"
    save_output({
        "total": len(results),
        "completed": completed,
        "errors": errors,
        "results": results,
        "config": {
            "scenario_agent": agent_configs.scenario_agent.to_dict(),
            "search_agent": agent_configs.search_agent.to_dict(),
            "question_agent": agent_configs.question_agent.to_dict(),
            "solver_agent": agent_configs.solver_agent.to_dict(),
            "refine_agent": agent_configs.refine_agent.to_dict(),
        }
    }, summary_path)
    logger.info(f"汇总保存到: {summary_path}")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

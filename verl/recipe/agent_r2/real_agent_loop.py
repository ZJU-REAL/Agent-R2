"""real_agent_loop.py -- RealAgentLoop for sandbox-based multi-turn RL training.

Key difference from the upstream ToolAgentLoop:
  - A fresh sandbox session is created at trajectory start and destroyed at the end.
    (Previously: sessions were acquired from a pre-built pool and returned after use.)
  - All tools are create'd once at trajectory start and release'd once at the end.
  - _call_tool only calls execute; it never calls create / release.
  - Session lifecycle (create / stop) is managed here, not inside tools.
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

from recipe.agent_r2.metric_utils import TRACKED_TOOL_NAMES
from recipe.agent_r2.sandbox.file_management import create_session, init_sandbox

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Tool names whose call counts are tracked individually (see metric_utils.compute_real_metrics)
_TRACKED_TOOLS = TRACKED_TOOL_NAMES


def _append_assistant_msg(agent_data, tool_calls, content: str = "") -> None:
    """Append a structured assistant message to agent_data.messages for conversation logging."""
    text = content.replace("<|im_end|>", "")

    think_end = text.find("</think>")
    if think_end != -1:
        reasoning_content = text[:think_end].strip()
        plain_content = text[think_end + len("</think>"):].strip()
    else:
        reasoning_content = ""
        plain_content = text.strip()

    msg: dict = {"role": "assistant", "content": plain_content}
    if reasoning_content:
        msg["reasoning_content"] = reasoning_content
    if tool_calls:
        msg["tool_calls"] = [
            {
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
                "id": f"call_{agent_data.assistant_turns}_{i}",
            }
            for i, tc in enumerate(tool_calls)
        ]
    agent_data.messages.append(msg)


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"


class AgentData:
    """Encapsulates all state variables for the agent loop."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: Any,
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
    ):
        self.messages = messages
        self.image_data = image_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs

        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        self.tool_calls: list[FunctionCall] = []

        # ── Trajectory-level counters ──────────────────────────────────────
        self.tool_call_counts: dict[str, int] = {t: 0 for t in _TRACKED_TOOLS}
        self.parse_failure_count: int = 0
        self.tool_exec_failure_count: int = 0       # all tools
        self.terminal_exec_failure_count: int = 0   # terminal / view_image only


@register("real_agent")
class RealAgentLoop(AgentLoopBase):
    """Agent loop where all tools share one sandbox session per trajectory.

    Session lifecycle:
      - create_session() called at trajectory start (fresh session each time)
      - session.stop() called at trajectory end (destroys the session)
    """

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level RealAgentLoop initialization")

        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        cls.continue_rollout_on_parse_failure = config.actor_rollout_ref.rollout.multi_turn.get(
            "continue_rollout_on_parse_failure", False
        )
        cls.parse_penalty = config.actor_rollout_ref.rollout.multi_turn.get("parse_penalty", True)
        cls.parse_penalty_value = config.actor_rollout_ref.rollout.multi_turn.get("parse_penalty_value", 0.5)
        cls.terminal_penalty = config.actor_rollout_ref.rollout.multi_turn.get("terminal_penalty", True)
        cls.terminal_penalty_value = config.actor_rollout_ref.rollout.multi_turn.get("terminal_penalty_value", 0.5)

        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        cls.tool_parser = ToolParser.get_tool_parser(
            config.actor_rollout_ref.rollout.multi_turn.format,
            cls.tokenizer,
        )
        print(f"Initialized tools: {cls.tools}")

        # Initialize shared Environment once at class init
        cls.sandbox_config = dict(config.actor_rollout_ref.rollout.multi_turn.get("sandbox_config", {}))
        init_sandbox(cls.sandbox_config)

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.max_model_len = config.actor_rollout_ref.rollout.max_model_len
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )

        cls.output_dir = config.actor_rollout_ref.rollout.multi_turn.get(
            "output_dir",
            str(Path(__file__).resolve().parent / "output"),
        )
        cls.reward_config = dict(config.actor_rollout_ref.rollout.multi_turn.get("reward_config", {}))

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        metrics = {}
        request_id = uuid4().hex
        extra_info = kwargs.get("extra_info", {}) or {}
        # tools_kwargs may be at top level or nested inside extra_info
        tools_kwargs = kwargs.get("tools_kwargs") or extra_info.get("tools_kwargs", {})

        agent_data = AgentData(
            messages=messages,
            image_data=image_data,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
        )

        # ── Trajectory start: create a fresh sandbox session ──
        session = await create_session()

        task_id = extra_info.get("task_id", request_id)
        traj_dir = Path(self.output_dir) / str(task_id) / request_id

        # ── Create all tools, passing the shared session ──
        create_kwargs = tools_kwargs.get("__create_kwargs__", {})
        for tool_name, tool in self.tools.items():
            per_tool_create_kwargs = create_kwargs.get(
                tool_name, tools_kwargs.get(tool_name, {}).get("create_kwargs", {})
            )
            if tool_name == "view_image_text" and "tmp_dir" not in per_tool_create_kwargs:
                per_tool_create_kwargs = dict(per_tool_create_kwargs)
                per_tool_create_kwargs["tmp_dir"] = str(traj_dir / "solver" / "tmp")
            try:
                await tool.create(instance_id=request_id, session=session, **per_tool_create_kwargs)
            except Exception as e:
                logger.warning(f"[RealAgentLoop] tool.create failed for {tool_name}: {e}")

        total_reward: Optional[float] = None
        reward_info: dict = {}

        try:
            state = AgentState.PENDING
            while state != AgentState.TERMINATED:
                if state == AgentState.PENDING:
                    state = await self._handle_pending_state(agent_data, sampling_params)
                elif state == AgentState.GENERATING:
                    state = await self._handle_generating_state(agent_data, sampling_params)
                elif state == AgentState.PROCESSING_TOOLS:
                    state = await self._handle_processing_tools_state(agent_data)
                else:
                    logger.error(f"Invalid state: {state}")
                    state = AgentState.TERMINATED
        finally:
            # ── Compute reward via mllm_reward ────────────────────────────────
            rubric = extra_info.get("rubric", [])
            if isinstance(rubric, str):
                try:
                    rubric = json.loads(rubric)
                except Exception:
                    rubric = []
            question = extra_info.get("question", None)
            if isinstance(question, str):
                try:
                    question = json.loads(question)
                except Exception:
                    question = None

            try:
                from recipe.agent_r2.reward.mllm_reward import compute_reward, resolve_reference_files
                reference_files = resolve_reference_files(extra_info, self.__class__.reward_config)
                logger.warning(
                    f"[RealAgentLoop] mllm_reward start | request_id={request_id} "
                    f"task_id={task_id} assistant_turns={agent_data.assistant_turns} "
                    f"rubric_len={len(rubric)}"
                )
                total_reward, reward_info = await compute_reward(
                    session=session,
                    rubric=rubric,
                    question=question,
                    request_id=request_id,
                    traj_dir=traj_dir,
                    llm_config=self.__class__.reward_config,
                    reference_files=reference_files,
                )
                logger.warning(
                    f"[RealAgentLoop] mllm_reward done  | request_id={request_id} "
                    f"total_reward={total_reward} reward_info={reward_info}"
                )
            except Exception as e:
                logger.warning(f"[RealAgentLoop] mllm_reward failed: {e}")
                total_reward = 0.0

            await self._save_conversation(agent_data, extra_info, reward_info, traj_dir)

            for tool_name, tool in self.tools.items():
                try:
                    await tool.release(request_id)
                except Exception as e:
                    logger.warning(f"[RealAgentLoop] tool.release failed for {tool_name}: {e}")
            await session.stop()

        # Apply parse / tool-exec penalties (once per failing turn)
        penalty = 0.0
        if self.continue_rollout_on_parse_failure and self.parse_penalty:
            penalty += agent_data.parse_failure_count * self.parse_penalty_value
        if self.terminal_penalty:
            penalty += agent_data.terminal_exec_failure_count * self.terminal_penalty_value
        penalized_reward = (total_reward or 0.0) - penalty if penalty > 0 else total_reward

        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask):]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {"image": agent_data.image_data} if agent_data.image_data is not None else {}
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            reward_score=penalized_reward,
            extra_fields={
                "reward_scores": {"mllm": total_reward} if total_reward is not None else {},
                # One row per trajectory; always the last (only) step for metric_utils.
                "is_last_step": 1,
                "terminal_exec_failure_count": agent_data.terminal_exec_failure_count,
                **{f"tool_call_counts_{t}": agent_data.tool_call_counts.get(t, 0)
                   for t in _TRACKED_TOOLS},
                "parse_failure_count": agent_data.parse_failure_count,
                "tool_exec_failure_count": agent_data.tool_exec_failure_count,
                "has_output": int(bool(reward_info.get("has_output", False))),
            },
        )
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    agent_data.messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=agent_data.image_data, return_tensors="pt")
            agent_data.prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            agent_data.prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    agent_data.messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
        return AgentState.GENERATING

    async def _handle_generating_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
            )

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if len(agent_data.response_mask) >= self.response_length:
            content, tool_calls, _ = await self.tool_parser.extract_tool_calls(agent_data.response_ids)
            _append_assistant_msg(agent_data, tool_calls, content)
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            content, tool_calls, _ = await self.tool_parser.extract_tool_calls(agent_data.response_ids)
            _append_assistant_msg(agent_data, tool_calls, content)
            return AgentState.TERMINATED

        estimated_next_turn = self.max_tool_response_length
        if len(agent_data.prompt_ids) + estimated_next_turn >= self.max_model_len:
            logger.warning(
                f"Prompt length ({len(agent_data.prompt_ids)}) + estimated next turn ({estimated_next_turn}) "
                f"would exceed max_model_len ({self.max_model_len}). Terminating early."
            )
            content, tool_calls, _ = await self.tool_parser.extract_tool_calls(agent_data.response_ids)
            _append_assistant_msg(agent_data, tool_calls, content)
            return AgentState.TERMINATED

        content, agent_data.tool_calls, parse_penalty_flag = await self.tool_parser.extract_tool_calls(
            agent_data.response_ids
        )
        _append_assistant_msg(agent_data, agent_data.tool_calls, content)
        has_parse_failure = parse_penalty_flag or any(tc.parse_error for tc in agent_data.tool_calls)
        if has_parse_failure:
            if self.continue_rollout_on_parse_failure:
                agent_data.parse_failure_count += 1
            else:
                return AgentState.TERMINATED

        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []

        tasks = [
            self._call_tool(tool_call, agent_data.request_id)
            for tool_call in agent_data.tool_calls[: self.max_parallel_calls]
        ]

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        has_exec_failure = False
        has_terminal_exec_failure = False
        for tc, (tool_response, tool_reward, _) in zip(
            agent_data.tool_calls[: self.max_parallel_calls], responses
        ):
            tool_name = tc.name

            # Detect tool execution failure: JSON {"error": ...} with "error" as top-level key
            text = tool_response.text or ""
            _is_json_error = False
            if text.startswith("{"):
                try:
                    _is_json_error = "error" in json.loads(text)
                except (json.JSONDecodeError, ValueError):
                    pass
            if _is_json_error:
                has_exec_failure = True
                if any(tool_name.startswith(p) for p in ("terminal", "view_image")):
                    has_terminal_exec_failure = True

            if tool_response.image or tool_response.video:
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            if tool_response.image:
                if agent_data.image_data is None:
                    agent_data.image_data = []
                elif not isinstance(agent_data.image_data, list):
                    agent_data.image_data = [agent_data.image_data]
                imgs = tool_response.image if isinstance(tool_response.image, list) else [tool_response.image]
                for img in imgs:
                    if img is not None:
                        agent_data.image_data.append(img)
                        new_images_this_turn.append(img)

            # Per-tool call counts
            if tool_name in agent_data.tool_call_counts:
                agent_data.tool_call_counts[tool_name] += 1
            else:
                for bucket in _TRACKED_TOOLS:
                    if tool_name.startswith(bucket):
                        agent_data.tool_call_counts[bucket] += 1
                        break

        agent_data.messages.extend(add_messages)

        if has_exec_failure:
            agent_data.tool_exec_failure_count += 1
        if has_terminal_exec_failure:
            agent_data.terminal_exec_failure_count += 1

        if self.processor is not None:
            raw_tool_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            current_images = new_images_this_turn if new_images_this_turn else None
            model_inputs = self.processor(text=[raw_tool_response], images=current_images, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
            )
        response_ids = response_ids[len(self.system_prompt):]
        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _call_tool(
        self, tool_call: FunctionCall, trajectory_id: str
    ) -> tuple[ToolResponse, float, dict]:
        """Execute a tool call. create/release are NOT called here — managed at trajectory level."""
        if tool_call.parse_error:
            msg = {"error": f"Failed to parse arguments for tool '{tool_call.name}'. Please provide valid JSON arguments."}
            logger.warning(f"[_call_tool] parse_error for tool '{tool_call.name}', returning error message")
            return ToolResponse(text=json.dumps(msg)), 0.0, {}
        try:
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            tool_execution_response, tool_reward, res = await tool.execute(trajectory_id, tool_args)
        except Exception as e:
            logger.warning({"error": f"Tool {tool_call.name} failed: {e}"})
            return ToolResponse(text=json.dumps({"error": f"Tool {tool_call.name} failed: {e}"})), 0.0, {}

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length:]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        tool_response_kwargs = {"text": tool_response_text}
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res

    async def _save_conversation(
        self,
        agent_data: AgentData,
        extra_info: dict,
        reward_info: dict,
        traj_dir: Path,
    ) -> None:
        """Save solver conversation history to traj_dir/solver/conversation.json."""
        try:
            solver_dir = traj_dir / "solver"
            solver_dir.mkdir(parents=True, exist_ok=True)

            def _serialize_msg(msg):
                if isinstance(msg, dict):
                    out = {}
                    for k, v in msg.items():
                        if k in ("image", "video") or isinstance(v, bytes):
                            out[k] = f"<{k}_data_omitted>"
                        elif isinstance(v, list):
                            out[k] = [
                                _serialize_msg(item) if isinstance(item, dict) else item
                                for item in v
                            ]
                        else:
                            out[k] = v
                    return out
                return msg

            conversation = [_serialize_msg(m) for m in agent_data.messages]

            with open(solver_dir / "conversation.json", "w", encoding="utf-8") as f:
                json.dump(conversation, f, ensure_ascii=False, indent=4)

        except Exception as e:
            logger.warning(f"[RealAgentLoop] _save_conversation failed: {e}")

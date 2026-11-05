"""agent_loop_split.py -- AgentLoopWorker variant that handles split (per-turn) outputs.

Replaces AgentLoopWorker in agent_loop.py for the split training mode:
  - _run_agent_loop: calls agent_loop.run() which returns list[AgentLoopOutput],
    pads each turn output, broadcasts the single trajectory reward to all splits.
  - _postprocess: flattens all per-turn outputs from all trajectories into one batch.

All other infrastructure (AsyncLLMServerManager, AgentLoopManager, etc.) is reused
from the original agent_loop.py via import.

Usage: set the env var USE_AGENT_LOOP_SPLIT=1, or call install_split_worker() before
the trainer initializes, to replace AgentLoopWorker with AgentLoopWorkerSplit inside
AgentLoopManager.
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any

import hydra
import numpy as np
import ray
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopOutput,
    AsyncLLMServerManager,
    BatchExecutor,
    RewardManagerWorker,
    _DummyConfig,
    _InternalAgentLoopOutput,
    _agent_loop_registry,
    get_trajectory_info,
)
from verl.protocol import DataProto
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_attr

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@ray.remote
class AgentLoopWorkerSplit:
    """Agent loop worker that handles split per-turn outputs from RealAgentLoopSplit."""

    def __init__(
        self,
        config: DictConfig,
        server_handles: list[ray.actor.ActorHandle],
        rm_executor: BatchExecutor = None,
        session_ids: list[str] = None,
    ):
        self.config = config
        self.server_manager = AsyncLLMServerManager(config, server_handles)
        self.rm_executor = rm_executor
        self._session_ids = session_ids  # per-worker shard; forwarded to init_class via instantiate

        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.processor = hf_processor(local_path, trust_remote_code=True)

        agent_loop_config_path = config.actor_rollout_ref.rollout.agent.agent_loop_config_path
        if agent_loop_config_path:
            agent_loop_configs = OmegaConf.load(agent_loop_config_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config
        if self.config.actor_rollout_ref.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.actor_rollout_ref.model.custom_chat_template
            self.tokenizer.chat_template = self.config.actor_rollout_ref.model.custom_chat_template

        self.reward_manager_worker = RewardManagerWorker.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False,
            ),
        ).remote(self.config, local_path, self.rm_executor)

        trace_config = self.config.actor_rollout_ref.rollout.get("trace", {})
        RolloutTraceConfig.init(
            self.config.trainer.project_name,
            self.config.trainer.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
        )

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences; each input sample may expand into multiple output samples (one per turn)."""
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        index = batch.non_tensor_batch.get("index", np.arange(len(batch)))
        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        # Use the uid from the incoming batch so that the n rollouts of the same
        # prompt share the same uid.  GRPO groups by uid to compute mean/std; if
        # every trajectory gets a fresh uuid the group size is always 1 and the
        # advantage collapses to 0.
        prompt_uids = batch.non_tensor_batch.get("uid", None)

        tasks = []
        traj_uids = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            traj_uid = str(prompt_uids[i]) if prompt_uids is not None else str(uuid.uuid4())
            traj_uids.append(traj_uid)
            tasks.append(asyncio.create_task(self._run_agent_loop(sampling_params, trajectory_info[i], **kwargs)))

        # outputs: list of list[_InternalAgentLoopOutput] (one list per trajectory)
        per_traj_outputs = await asyncio.gather(*tasks)

        # Flatten: list[_InternalAgentLoopOutput], carry per-trajectory uid
        all_outputs = []
        all_uids = []
        for traj_uid, traj_snaps in zip(traj_uids, per_traj_outputs):
            for snap in traj_snaps:
                all_outputs.append(snap)
                all_uids.append(traj_uid)
        return self._postprocess(all_outputs, all_uids)

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        **kwargs,
    ) -> list[_InternalAgentLoopOutput]:
        """Run agent loop; returns a list of padded per-turn outputs sharing the same reward."""
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered: {list(_agent_loop_registry.keys())}"
            )
            agent_loop = hydra.utils.instantiate(
                config=_agent_loop_registry[agent_name],
                trainer_config=_DummyConfig(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
                session_ids=self._session_ids,
            )
            # run() returns list[AgentLoopOutput]
            outputs: list[AgentLoopOutput] = await agent_loop.run(sampling_params, **kwargs)

            if not outputs:
                return []

            # ── Compute reward once for the trajectory ────────────────────────
            # Use the last turn's output to build the reward DataProto (full trajectory context)
            last = outputs[-1]
            reward_score = last.reward_score
            enable_async_reward = (
                self.rm_executor is not None and self.config.reward_model.enable_resource_pool
            ) or not self.config.reward_model.enable

            if reward_score is None and enable_async_reward:
                # Build DataProto from last turn for reward computation
                reward_data = self._build_reward_data(last, kwargs)
                result = await self.reward_manager_worker.compute_score.remote(reward_data)
                reward_score = result["reward_score"]
                reward_extra_info = result["reward_extra_info"]
                # Broadcast to all outputs (external reward applies uniformly to every turn)
                for o in outputs:
                    o.reward_score = reward_score
                    o.extra_fields["reward_extra_info"] = reward_extra_info
            else:
                # Only fill in outputs whose reward_score is still None.
                # Per-turn penalties (e.g. parse_penalty=-1.0) set by the agent loop
                # must not be overwritten with the trajectory-level score.
                for o in outputs:
                    if o.reward_score is None:
                        o.reward_score = reward_score

            # ── Pad each per-turn output ──────────────────────────────────────
            return [self._pad_output(o) for o in outputs]

    def _build_reward_data(self, output: AgentLoopOutput, kwargs: dict) -> DataProto:
        """Build a single-sample DataProto for reward computation from the last turn output."""
        self.tokenizer.padding_side = "left"
        prompt_out = self.tokenizer.pad(
            {"input_ids": output.prompt_ids},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        self.tokenizer.padding_side = "right"
        response_out = self.tokenizer.pad(
            {"input_ids": output.response_ids},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        attention_mask = torch.cat([prompt_out["attention_mask"], response_out["attention_mask"]], dim=1)
        input_ids = torch.cat([prompt_out["input_ids"], response_out["input_ids"]], dim=1)
        position_ids = compute_position_id_with_mask(attention_mask)
        batch = TensorDict(
            {
                "prompts": prompt_out["input_ids"],
                "responses": response_out["input_ids"],
                "attention_mask": attention_mask,
                "input_ids": input_ids,
                "position_ids": position_ids,
            },
            batch_size=1,
        )
        non_tensor_batch = {
            **{k: np.array([v]) for k, v in kwargs.items()},
            "__num_turns__": np.array([output.num_turns]),
        }
        for key, val in output.extra_fields.items():
            non_tensor_batch[key] = np.array([val], dtype=object)
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    def _pad_output(self, output: AgentLoopOutput) -> _InternalAgentLoopOutput:
        """Pad a single AgentLoopOutput into _InternalAgentLoopOutput tensors."""
        self.tokenizer.padding_side = "left"
        prompt_out = self.tokenizer.pad(
            {"input_ids": output.prompt_ids},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if prompt_out["input_ids"].dim() == 1:
            prompt_out["input_ids"] = prompt_out["input_ids"].unsqueeze(0)
            prompt_out["attention_mask"] = prompt_out["attention_mask"].unsqueeze(0)

        self.tokenizer.padding_side = "right"
        response_out = self.tokenizer.pad(
            {"input_ids": output.response_ids},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if response_out["input_ids"].dim() == 1:
            response_out["input_ids"] = response_out["input_ids"].unsqueeze(0)
            response_out["attention_mask"] = response_out["attention_mask"].unsqueeze(0)

        response_mask_out = self.tokenizer.pad(
            {"input_ids": output.response_mask},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        if response_mask_out["input_ids"].dim() == 1:
            response_mask_out["input_ids"] = response_mask_out["input_ids"].unsqueeze(0)

        response_logprobs = None
        if output.response_logprobs is not None:
            pad_size = self.config.actor_rollout_ref.rollout.response_length - len(output.response_logprobs)
            response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

        response_mask = response_mask_out["input_ids"] * response_out["attention_mask"]
        attention_mask = torch.cat([prompt_out["attention_mask"], response_out["attention_mask"]], dim=1)
        input_ids = torch.cat([prompt_out["input_ids"], response_out["input_ids"]], dim=1)
        position_ids = compute_position_id_with_mask(attention_mask)

        return _InternalAgentLoopOutput(
            prompt_ids=prompt_out["input_ids"],
            response_ids=response_out["input_ids"],
            input_ids=input_ids,
            position_ids=position_ids,
            response_mask=response_mask,
            attention_mask=attention_mask,
            response_logprobs=response_logprobs,
            multi_modal_inputs=None,
            multi_modal_data=output.multi_modal_data,
            reward_score=output.reward_score,
            num_turns=output.num_turns,
            metrics=output.metrics,
            extra_fields=output.extra_fields,
        )

    def _postprocess(self, inputs: list[_InternalAgentLoopOutput], uids: list[str] = None) -> DataProto:
        """Stack all per-turn outputs (flattened across trajectories) into a DataProto batch."""
        if not inputs:
            raise ValueError("No outputs to postprocess")

        prompt_ids = torch.cat([x.prompt_ids for x in inputs], dim=0)
        response_ids = torch.cat([x.response_ids for x in inputs], dim=0)
        response_mask = torch.cat([x.response_mask for x in inputs], dim=0)
        attention_mask = torch.cat([x.attention_mask for x in inputs], dim=0)
        input_ids = torch.cat([x.input_ids for x in inputs], dim=0)
        position_ids = torch.cat([x.position_ids for x in inputs], dim=0)

        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([x.response_logprobs for x in inputs], dim=0)

        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids,
                "response_mask": response_mask,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [x.reward_score for x in inputs]
        if all(s is not None for s in scores):
            prompt_length = prompt_ids.size(1)
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([x.num_turns for x in inputs], dtype=np.int32),
            "uid": np.array(uids if uids is not None else [str(uuid.uuid4()) for _ in inputs], dtype=object),
        }

        reward_extra_infos = [x.extra_fields.get("reward_extra_info", {}) for x in inputs]
        reward_extra_keys = list(reward_extra_infos[0].keys()) if reward_extra_infos[0] else []
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        metrics = [x.metrics.model_dump() for x in inputs]
        extra_fields: dict = {}
        all_keys = set(key for x in inputs for key in x.extra_fields)
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [x.extra_fields.get(key) for x in inputs]
            extra_fields[key] = temp_arr
        non_tensor_batch.update(extra_fields)

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={"metrics": metrics, "reward_extra_keys": reward_extra_keys},
        )

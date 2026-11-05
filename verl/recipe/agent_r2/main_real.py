"""main_real.py -- Entry point for the real recipe RL training.

This module implements multi-turn GRPO training for the productivity task
solver agent, using sandbox-based rollouts via the SandboxTerminalTool.

The reward is an outcome reward: 1.0 if output files were produced in
/mnt/user-data/outputs, 0.0 otherwise.
"""
from __future__ import annotations

import os
import socket
from pathlib import Path

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import is_cuda_available
from verl.utils.import_utils import load_extern_type

# Config path relative to this file's location
_CONFIG_PATH = str(Path(__file__).parent / "config")


@hydra.main(config_path=_CONFIG_PATH, config_name="ppo_trainer_real", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)
class TaskRunner:
    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        from verl.single_controller.ray import RayWorkerGroup

        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup
        else:
            raise NotImplementedError(f"Unsupported actor strategy: {config.actor_rollout_ref.actor.strategy}")

        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config):
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import CriticWorker
        elif config.critic.strategy == "megatron":
            from verl.workers.megatron_workers import CriticWorker
        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)

    def init_resource_pool_mgr(self, config):
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        self.mapping[Role.ActorRollout] = global_pool_id
        self.mapping[Role.Critic] = global_pool_id

        if config.reward_model.get("enable_resource_pool", False):
            reward_pool = [config.reward_model.n_gpus_per_node] * config.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=self.mapping,
        )
        return resource_pool_manager

    def add_reward_model_worker(self, config):
        from verl.trainer.ppo.ray_trainer import Role

        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError

            self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            if config.reward_model.get("enable_resource_pool", False):
                self.mapping[Role.RewardModel] = "reward_pool"
            else:
                self.mapping[Role.RewardModel] = "global_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        from verl.trainer.ppo.ray_trainer import Role

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
            self.mapping[Role.RefPolicy] = "global_pool"

    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local
        
        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        resource_pool_manager = self.init_resource_pool_mgr(config)

        train_dataset = _create_rl_dataset(
            config.data.train_files, config.data, tokenizer, processor
        )
        val_files = config.data.get("val_files", None)
        if val_files:
            val_dataset = _create_rl_dataset(val_files, config.data, tokenizer, processor)
        else:
            # No validation set: reuse training set to avoid None path error in _create_dataloader
            val_dataset = train_dataset
        train_sampler = _create_rl_sampler(config.data, train_dataset)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        # Patch compute_data_metrics so RayPPOTrainer logs real/ metrics for both
        # real_agent and real_agent_split rollouts.
        from recipe.agent_r2.metric_utils import install_compute_data_metrics_patch

        install_compute_data_metrics_patch()

        # If using split agent loop, patch AgentLoopWorker inside agent_loop module
        # so AgentLoopManager._init_agent_loop_workers picks up the split variant.
        agent_loop = config.actor_rollout_ref.rollout.agent.get("default_agent_loop", "")
        if "split" in agent_loop:
            import verl.experimental.agent_loop.agent_loop as _al_module
            from recipe.agent_r2.split.agent_loop_split import AgentLoopWorkerSplit
            from recipe.agent_r2.split.ray_trainer_split import install
            _al_module.AgentLoopWorker = AgentLoopWorkerSplit
            install()

        trainer.init_workers()
        trainer.fit()


def _create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create RLHFDataset from parquet file(s)."""
    from verl.utils.dataset.rl_dataset import RLHFDataset

    if data_paths is None:
        return None

    dataset = RLHFDataset(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )
    return dataset


def _create_rl_sampler(data_config, dataset):
    """Create sampler for dataset."""
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    shuffle = data_config.get("shuffle", True)
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(data_config.get("seed", 42))
        sampler = RandomSampler(data_source=dataset, generator=generator)
    else:
        sampler = SequentialSampler(data_source=dataset)
    return sampler


if __name__ == "__main__":
    main()

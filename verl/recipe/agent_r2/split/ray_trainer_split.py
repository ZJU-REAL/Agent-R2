"""ray_trainer_split.py -- Patches for split per-turn trajectory training.

1. DataProto.union() patch: gen_batch_output has more rows than the prompt batch
   (one per assistant turn). Instead of union()-ing, we directly use gen_batch_output
   trimmed to a multiple of n_gpus_per_node.

2. compute_data_metrics patch: see recipe.agent_r2.metric_utils.install_compute_data_metrics_patch
   (installed from main_real.py for all real-agent training, including non-split).
"""
import verl.trainer.ppo.ray_trainer as _rt_module
from verl.protocol import DataProto


def install():
    """Install all patches required for split training."""
    _patch_union()


def _patch_union():
    """Patch RayPPOTrainer.fit() so that when gen_batch_output is a split-expanded
    batch, batch.union(gen_batch_output) returns gen_batch_output directly instead
    of trying to merge mismatched sizes."""

    _orig_fit = _rt_module.RayPPOTrainer.fit
    _orig_union = DataProto.union

    def fit_split(trainer_self):
        # Step 1: wrap generate_sequences to tag split outputs
        if trainer_self.async_rollout_mode:
            _orig_gen = trainer_self.async_rollout_manager.generate_sequences
            def _gen(gen_batch):
                out = _orig_gen(gen_batch)
                out.meta_info["_is_split"] = True
                return out
            trainer_self.async_rollout_manager.generate_sequences = _gen
        else:
            _orig_gen = trainer_self.actor_rollout_wg.generate_sequences
            def _gen(gen_batch):
                out = _orig_gen(gen_batch)
                out.meta_info["_is_split"] = True
                return out
            trainer_self.actor_rollout_wg.generate_sequences = _gen

        # Step 2: patch DataProto.union to intercept the one mismatched call
        def _union(self_batch: DataProto, other: DataProto) -> DataProto:
            if other.meta_info.get("_is_split") and len(other) != len(self_batch):
                # Skip union; use gen_batch_output directly, trimmed to GPU multiple
                n_gpus = trainer_self.config.trainer.n_gpus_per_node * trainer_self.config.trainer.nnodes
                output = other
                remainder = len(output) % n_gpus
                if remainder != 0:
                    output = output[:-remainder]
                # uid is already set per-trajectory in AgentLoopWorkerSplit._postprocess
                return output
            return _orig_union(self_batch, other)

        DataProto.union = _union
        try:
            _orig_fit(trainer_self)
        finally:
            DataProto.union = _orig_union

    _rt_module.RayPPOTrainer.fit = fit_split

"""metric_utils.py -- Extended metrics for the real recipe.

Supplements verl's built-in metric_utils with:
  - num_turns: only aggregated over the LAST step of each trajectory (is_last_step=True)
  - Per-tool call counts (terminal, web_search, web_fetch, view_image)
  - parse_failure_count: number of turns where tool-call parsing failed
  - tool_exec_failure_count: number of turns where tool execution returned an error

Data contract (non_tensor_batch keys):
  - real_agent_loop.py: one row per trajectory; sets is_last_step=1 and trajectory counters
    in AgentLoopOutput.extra_fields (merged into non_tensor_batch by agent_loop._postprocess).
  - real_agent_loop_split.py: one row per assistant turn; is_last_step marks the final turn.

Call `install_compute_data_metrics_patch()` from main_real.py so RayPPOTrainer.fit()
logs real/ metrics via the patched compute_data_metrics.  All new keys use the "real/"
namespace to avoid collision with upstream keys.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from verl import DataProto


# Tool names tracked individually (shared with real_agent_loop / real_agent_loop_split)
TRACKED_TOOL_NAMES = ["terminal", "web_search", "web_fetch", "view_image"]
_TOOL_NAMES = TRACKED_TOOL_NAMES


def install_compute_data_metrics_patch() -> None:
    """Patch compute_data_metrics so RayPPOTrainer logs real/ metrics.

    ray_trainer.py imports compute_data_metrics into its module namespace, so we patch
    both verl.trainer.ppo.metric_utils and verl.trainer.ppo.ray_trainer.
    """
    import logging

    import verl.trainer.ppo.metric_utils as _mu_module
    import verl.trainer.ppo.ray_trainer as _rt_module

    if getattr(_mu_module.compute_data_metrics, "_real_metrics_patched", False):
        return

    _orig_compute = _mu_module.compute_data_metrics
    _log = logging.getLogger(__name__)

    def _patched(batch: DataProto, use_critic: bool = True):
        result = _orig_compute(batch=batch, use_critic=use_critic)
        try:
            result.update(compute_real_metrics(batch))
        except Exception as e:
            _log.warning("compute_real_metrics failed: %s", e)
        return result

    _patched._real_metrics_patched = True
    _mu_module.compute_data_metrics = _patched
    _rt_module.compute_data_metrics = _patched


def compute_real_metrics(batch: DataProto) -> dict[str, Any]:
    """Compute real-recipe-specific metrics from a split batch.

    Expected non_tensor_batch keys (set in real_agent_loop.py or real_agent_loop_split.py):
      - is_last_step (bool/int): 1 for the final turn of a trajectory, 0 otherwise
      - __num_turns__ (int): total assistant turns in the trajectory (same value shared
          across all splits of the same trajectory)
      - tool_call_counts_{name} (int): cumulative call count for each tool up to this turn
          (for the last step, this equals the total for the whole trajectory)
      - parse_failure_count (int): number of parse-failure turns in this trajectory
          (same value shared across splits; meaningful only on last step)
      - tool_exec_failure_count (int): number of tool-exec-failure turns in this trajectory

    Returns:
        dict of metric name → scalar float.
    """
    metrics: dict[str, Any] = {}
    nb = batch.non_tensor_batch

    # ── is_last_step mask ─────────────────────────────────────────────────────
    if "is_last_step" in nb:
        raw = nb["is_last_step"]
        # may be object array (from extra_fields) or int32; cast safely
        last_mask = np.array([bool(int(x)) for x in raw], dtype=bool)
    else:
        # Fallback: treat all samples as last step (backward compat)
        n = len(batch)
        last_mask = np.ones(n, dtype=bool)

    if not last_mask.any():
        # Nothing to aggregate on last steps
        return metrics

    def _to_float_arr(arr, mask):
        """Cast an object or numeric ndarray to float, filtered by mask."""
        return np.array([float(x) for x in arr[mask]], dtype=float)

    # ── num_turns (last-step only) ────────────────────────────────────────────
    if "__num_turns__" in nb:
        turns = _to_float_arr(nb["__num_turns__"], last_mask)
        metrics["real/num_turns/mean"] = float(np.mean(turns))
        metrics["real/num_turns/max"] = float(np.max(turns))
        metrics["real/num_turns/min"] = float(np.min(turns))

    # ── per-tool call counts (last-step only) ─────────────────────────────────
    for tool_name in _TOOL_NAMES:
        key = f"tool_call_counts_{tool_name}"
        if key in nb:
            counts = _to_float_arr(nb[key], last_mask)
            metrics[f"real/tool_calls/{tool_name}/mean"] = float(np.mean(counts))
            metrics[f"real/tool_calls/{tool_name}/sum"] = float(np.sum(counts))

    # ── parse failure count (last-step only) ──────────────────────────────────
    if "parse_failure_count" in nb:
        pf = _to_float_arr(nb["parse_failure_count"], last_mask)
        metrics["real/parse_failure/mean"] = float(np.mean(pf))
        metrics["real/parse_failure/sum"] = float(np.sum(pf))

    # ── tool execution failure count (last-step only) ─────────────────────────
    if "tool_exec_failure_count" in nb:
        ef = _to_float_arr(nb["tool_exec_failure_count"], last_mask)
        metrics["real/tool_exec_failure/mean"] = float(np.mean(ef))
        metrics["real/tool_exec_failure/sum"] = float(np.sum(ef))

    # ── has_output: count of trajectories with/without deliverable ───────────
    if "has_output" in nb:
        has_out = np.array([int(x) for x in nb["has_output"][last_mask]], dtype=int)
        metrics["real/has_output/with_deliverable"] = int(np.sum(has_out))
        metrics["real/has_output/without_deliverable"] = int(np.sum(has_out == 0))

    # ── reward on last-step only (for reference) ──────────────────────────────
    if "rm_scores" in batch.batch:
        # rm_scores is token-level; sum over response dim to get per-sample reward
        rm = batch.batch["rm_scores"].sum(dim=-1).float().cpu().numpy()
        last_rewards = rm[last_mask]
        if len(last_rewards) > 0:
            metrics["real/reward_last_step/mean"] = float(np.mean(last_rewards))
            # Only average over samples with deliverable (has_output=1)
            if "has_output" in nb:
                has_out = np.array([int(x) for x in nb["has_output"][last_mask]], dtype=bool)
                with_deliverable = last_rewards[has_out]
                if len(with_deliverable) > 0:
                    metrics["real/reward_last_step/with_deliverable_mean"] = float(np.mean(with_deliverable))

    return metrics

#!/usr/bin/env bash
# run_agent_r2.sh -- Launch GRPO training for productivity task solver agent.
#
# Prerequisites:
#   1. conda activate verl
#   2. Preprocess the dataset:
#        python recipe/agent_r2/preprocess.py \
#            --input ../Agent-R2/data_sythesis/output/train.parquet \
#            --output recipe/agent_r2/data/train.parquet
#   3. Fill URLs / keys in ../Agent-R2/settings/settings.yaml
#
# Run from the verl project root:
#   bash recipe/agent_r2/run_agent_r2.sh

set -x

# Enable core dumps with limited size (for debugging)
ulimit -c 0  # Disable core dumps to save disk space

ulimit -n 65535

export VLLM_USE_V1=1
export CUDA_VISIBLE_DEVICES="0,1,2,3"
# Keep PyTorch rendezvous ports out of the host's Ray/OpenMPI ephemeral port range.
export VERL_MASTER_PORT_RANGE="${VERL_MASTER_PORT_RANGE:-35000:61000}"
# Resolve project root relative to this script so it works regardless of Cwd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-"$(cd "$SCRIPT_DIR/../.." && pwd)"}"
# Inner Agent-R2/ code directory (sibling of verl/): llm package and data_sythesis outputs
AGENT_DIR="${AGENT_DIR:-"$(cd "$PROJECT_DIR/../Agent-R2" && pwd)"}"
RECIPE_DIR="$PROJECT_DIR/recipe/agent_r2"
CONFIG_PATH="$RECIPE_DIR/config"
TOOL_CONFIG_PATH="$RECIPE_DIR/config/tool_config/sandbox_tool_config.yaml"
AGENT_LOOP_CONFIG_PATH="$RECIPE_DIR/config/agent_loop_config.yaml"
AGENT_LOOP="${AGENT_LOOP:-real_agent}"
SESSION_POOL_DIR="$RECIPE_DIR/sandbox/sandbox_sessions/checked"
DATA_FILE="$RECIPE_DIR/data/train.parquet"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the SFT checkpoint (HF format)}"

# Ensure recipe is on PYTHONPATH so `recipe.agent_r2.tools.terminal_tool` is importable
# Also add the inner Agent-R2/ directory ($AGENT_DIR) for the llm module used by view_image_text_tool
export PYTHONPATH="$PROJECT_DIR:$AGENT_DIR:$PYTHONPATH"

# Preprocess data if not already done
# if [ ! -f "$DATA_FILE" ]; then
#     echo "Preprocessing dataset..."
#     python "$RECIPE_DIR/preprocess.py" \
#         --input ../Agent-R2/data_sythesis/output/train.parquet \
#         --output "$DATA_FILE"
# fi

python3 -m recipe.agent_r2.main_real \
    --config-path="$CONFIG_PATH" \
    --config-name="ppo_trainer_real" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_MASTER_PORT_RANGE="$VERL_MASTER_PORT_RANGE" \
    data.train_files="$DATA_FILE" \
    data.val_files=null \
    data.train_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=65536 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=False \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.max_model_len=69632 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=40 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=4096 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=5 \
    actor_rollout_ref.rollout.multi_turn.parse_penalty=False \
    actor_rollout_ref.rollout.multi_turn.parse_penalty_value=0.05 \
    actor_rollout_ref.rollout.multi_turn.continue_rollout_on_parse_failure=False \
    actor_rollout_ref.rollout.multi_turn.terminal_penalty=False \
    actor_rollout_ref.rollout.multi_turn.terminal_penalty_value=0.05 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH" \
    actor_rollout_ref.rollout.multi_turn.sandbox_session_pool_dir="$SESSION_POOL_DIR" \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_LOOP_CONFIG_PATH" \
    actor_rollout_ref.rollout.agent.default_agent_loop="$AGENT_LOOP" \
    actor_rollout_ref.rollout.agent.num_workers=64 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.99 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name="real_rl" \
    trainer.experiment_name="solver_grpo_qwen3_4b_thinking_2507_1_2" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    trainer.total_epochs=3 \
    trainer.val_before_train=False \
    actor_rollout_ref.rollout.trace.token2text=True \
    "$@"

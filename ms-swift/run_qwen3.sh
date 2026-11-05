#!/bin/bash
# Qwen3-VL-4B Full 全参数微调 (无LoRA) + 序列并行
# 4×A100 80G 可跑

set -e

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the base model (e.g. Qwen3-4B-Thinking-2507)}"
# SFT data written by Agent-R2/data_sythesis/prepare_sft_data_split.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${DATASET:-$SCRIPT_DIR/../Agent-R2/data_sythesis/data/sft.jsonl}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export MAX_PIXELS=1003520
export VIDEO_MAX_PIXELS=50176
export FPS_MAX_FRAMES=12
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
CELOSS_PARALLEL_SIZE=2048 \
swift sft \
    --model "$MODEL_PATH" \
    --dataset "$DATASET" \
    --add_non_thinking_prefix true \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --split_dataset_ratio 0.01 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-5 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --gradient_checkpointing true \
    --gradient_accumulation_steps 8 \
    --max_length 60000 \
    --warmup_ratio 0.05 \
    --attn_impl flash_attn \
    --loss_scale last_round+ignore_empty_think \
    --truncation_strategy right \
    --sequence_parallel_size 2 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --padding_free true \
    --use_logits_to_keep false
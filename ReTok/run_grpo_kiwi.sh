#!/usr/bin/env bash
# GRPO with dual reward (COMET + cometkiwi + repetition).
# Starts fresh from CPO-merged to give the new reward signal a clean canvas.
set -euo pipefail

PY=/home/tfbao/new/HY-MT/.venv/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$HERE/data/grpo_data.jsonl"

MODEL="$HERE/output_v18_tie_cpo_v3_plus_7b_merged"
OUT="$HERE/output_v18_tie_grpo_kiwi"

mkdir -p "$OUT"
cd "$HERE"

$PY -u train_grpo_kiwi.py \
    --model_path "$MODEL" \
    --data_path "$DATA" \
    --output_dir "$OUT" \
    --lr 1e-6 \
    --beta 0.04 \
    --num_generations 8 \
    --max_completion_length 384 \
    --batch_size 2 \
    --grad_accum 8 \
    --max_prompts 3000 \
    --full_finetune \
    --vllm_gpu_memory_utilization 0.2 \
    --vllm_max_model_length 1024 \
    --comet_weight 1.0 \
    --kiwi_weight 1.0 \
    --rep_weight 0.3 \
    2>&1 | tee "$OUT/train.log"

#!/usr/bin/env bash
# 2nd-round GRPO from 1st-round output. Matches Translator program.md advice:
# "For more gain, do a 2nd GRPO round, not more data."
# Same config, same 3000 prompts, just continue from 1st-round merged model.
set -euo pipefail

PY=/home/tfbao/new/HY-MT/.venv/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$HERE/../Translator/grpo_data.jsonl"

MODEL="$HERE/output_v18_tie_grpo_full"
OUT="$HERE/output_v18_tie_grpo_full_r2"

mkdir -p "$OUT"
cd "$HERE"

$PY -u train_grpo.py \
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
    2>&1 | tee "$OUT/train.log"

#!/usr/bin/env bash
# GRPO stage for ReTok.
# A/B with Qwen/output_1.7b_grpo_full (started from CPO-merged): same data,
# same reward (COMET wmt22-comet-da + 4-gram repetition penalty 1.0:0.3),
# same loss_type=dapo, num_generations=8, vllm colocate at 0.2 mem, max_completion=384.
#
# Usage:
#   bash run_grpo.sh                # start from CPO-merged (matches Qwen best lineage)
#   bash run_grpo.sh from_sft       # start from SFT (matches Qwen/output_1.7b_grpo_from_sft)
set -euo pipefail

PY=/home/tfbao/new/HY-MT/.venv/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$HERE/data/grpo_data.jsonl"

START="${1:-cpo}"
if [ "$START" = "from_sft" ]; then
    MODEL="$HERE/output_v18_tie_sft"
    OUT="$HERE/output_v18_tie_grpo_from_sft"
else
    MODEL="$HERE/output_v18_tie_cpo_v3_plus_7b_merged"
    OUT="$HERE/output_v18_tie_grpo_full"
fi

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

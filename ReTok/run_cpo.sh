#!/usr/bin/env bash
# CPO stage (LoRA) for ReTok.
# A/B with Qwen/output_1.7b_cpo_v3_plus_7b_merged: same data, same hyperparams,
# only the SFT base differs (output_v18_tie_sft vs output_1.7b_base_v2).
set -euo pipefail

PY=/home/tfbao/new/HY-MT/.venv/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_BASE="$HERE/output_v18_tie_sft"
DATA="$HERE/data/cpo_v3_plus_7b.jsonl"
OUT="$HERE/output_v18_tie_cpo_v3_plus_7b"
MERGED="$HERE/output_v18_tie_cpo_v3_plus_7b_merged"

mkdir -p "$OUT"
cd "$HERE"

# Train LoRA. Effective batch = 16, same as Qwen's best-practice run.
# Qwen used bs=1 ga=16 + gradient_checkpointing; we use bs=4 ga=4 without
# gradient_checkpointing because LoRA (~17M trainable) needs <6G memory at bs=4,
# and the original config underutilizes 4090 (~36% GPU, 5G/24G mem).
# Effective batch is unchanged, so the optimizer trajectory is loss-equivalent.
$PY -u train_cpo.py \
    --model_path "$SFT_BASE" \
    --data_path "$DATA" \
    --output_dir "$OUT" \
    --lr 1e-4 \
    --beta 0.05 \
    --nll_weight 1.0 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_length 512 \
    --logging_steps 50 \
    --save_steps 500 \
    2>&1 | tee "$OUT/train.log"

# Merge LoRA into base for eval / downstream GRPO
$PY -u merge_lora.py \
    --base_model_path "$SFT_BASE" \
    --adapter_model_path "$OUT" \
    --output_path "$MERGED" \
    --save_dtype bf16 \
    2>&1 | tee "$OUT/merge.log"

echo "CPO done. Merged model: $MERGED"

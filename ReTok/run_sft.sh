#!/usr/bin/env bash
# SFT for ReTok.
#
# A/B contract with the Qwen baseline (output_1.7b_base_v2):
#   - same data (alma_combined_sft_clean.jsonl)
#   - same prompt template ("Translate the following text from X to Y…")
#   - same hyperparams (AdamW lr=2e-5, inverse_sqrt, 1 epoch, bs=2*ga=8, gc)
#   - ONLY differs in base + tokenizer (Qwen3-1.7B-Base + Qwen3 HF tok
#     →  ReTok phase2 v18 + PieceTokenizer)
#
# Usage:
#   bash run_sft.sh              # full 1-epoch run (~2300 steps on 4090)
#   bash run_sft.sh smoke        # 50-step sanity check
set -euo pipefail

PY=/home/tfbao/new/HY-MT/.venv/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE=/home/tfbao/Shiyu/Summer/output/phase2_ckpt_v18_tie
DATA="$HERE/data/alma_combined_sft_clean.jsonl"

MODE="${1:-full}"
if [ "$MODE" = "smoke" ]; then
    OUT="$HERE/output_smoke"
    EXTRA="--max_steps 50 --warmup_steps 5 --save_steps 0 --logging_steps 5"
else
    OUT="$HERE/output_v18_tie_sft"
    # Qwen baseline used: --num_epochs 1, save every 1000 by default,
    # logging every 10. Match those.
    EXTRA="--num_epochs 1 --save_steps 1000 --logging_steps 10"
fi

mkdir -p "$OUT"
cd "$HERE"

exec $PY -u train.py \
    --model_path "$BASE" \
    --train_data "$DATA" \
    --output_dir "$OUT" \
    --lr 2e-5 \
    --lr_scheduler inverse_sqrt \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --max_seq_length 512 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    $EXTRA \
    2>&1 | tee "$OUT/train.log"

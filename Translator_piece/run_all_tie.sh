#!/usr/bin/env bash
# Full SFT → CPO+merge → GRPO chain on v18_tie, each followed by WMT23 eval.
# Total ETA ≈ 2h45m on a clean 4090.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/tfbao/new/HY-MT/.venv/bin/python
cd "$HERE"

eval_wmt23 () {
    # $1 = model dir, $2 = log filename
    local model="$1" log="$2"
    echo "=== eval $model -> $log ==="
    $PY -u eval_vllm_piece.py \
        --model_path "$model" \
        --testset wmt23 --direction both \
        --max_model_len 1024 \
        --gpu_memory_utilization 0.85 \
        > "$log" 2>&1
    tail -5 "$log" | grep -E "BLEU|COMET" || true
}

echo "===== [1/3] SFT $(date '+%H:%M') ====="
bash run_sft.sh
eval_wmt23 ./output_v18_tie_sft eval_v18_tie_sft_wmt23.log

echo "===== [2/3] CPO + merge $(date '+%H:%M') ====="
bash run_cpo.sh
eval_wmt23 ./output_v18_tie_cpo_v3_plus_7b_merged eval_v18_tie_cpo_wmt23.log

echo "===== [3/3] GRPO $(date '+%H:%M') ====="
bash run_grpo.sh
eval_wmt23 ./output_v18_tie_grpo_full eval_v18_tie_grpo_wmt23.log

echo "===== CHAIN DONE $(date '+%H:%M') ====="
for log in eval_v18_tie_sft_wmt23.log eval_v18_tie_cpo_wmt23.log eval_v18_tie_grpo_wmt23.log; do
    echo "--- $log ---"
    tail -5 "$log" | grep -E "BLEU|COMET|Summary"
done

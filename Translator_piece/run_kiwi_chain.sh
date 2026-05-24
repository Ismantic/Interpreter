#!/usr/bin/env bash
# Full kiwi chain: train + eval + results.tsv update.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/tfbao/new/HY-MT/.venv/bin/python
cd "$HERE"

echo "===== [1/3] GRPO with COMET+kiwi reward $(date '+%H:%M') ====="
bash run_grpo_kiwi.sh

echo "===== [2/3] eval $(date '+%H:%M') ====="
$PY -u eval_vllm_piece.py \
    --model_path ./output_v18_tie_grpo_kiwi \
    --testset wmt23 --direction both \
    --max_model_len 1024 --gpu_memory_utilization 0.85 \
    > eval_v18_tie_grpo_kiwi_wmt23.log 2>&1

echo "===== [3/3] update results.tsv $(date '+%H:%M') ====="
# 提取 BLEU / COMET
ZE_LINE=$(grep "zh-en:" eval_v18_tie_grpo_kiwi_wmt23.log | tail -1)
EZ_LINE=$(grep "en-zh:" eval_v18_tie_grpo_kiwi_wmt23.log | tail -1)
ZE_BLEU=$(echo "$ZE_LINE" | grep -oE 'BLEU = [0-9.]+' | grep -oE '[0-9.]+')
ZE_COMET=$(echo "$ZE_LINE" | grep -oE 'COMET = [0-9.]+' | grep -oE '[0-9.]+')
EZ_BLEU=$(echo "$EZ_LINE" | grep -oE 'BLEU = [0-9.]+' | grep -oE '[0-9.]+')
EZ_COMET=$(echo "$EZ_LINE" | grep -oE 'COMET = [0-9.]+' | grep -oE '[0-9.]+')

printf "piece_tie_grpo_kiwi\t./output_v18_tie_grpo_kiwi\twmt23\t%s\t%s\t%s\t%s\tGRPO with dual reward (COMET 1.0 + cometkiwi 1.0 + rep 0.3); from CPO-merged\n" \
    "$ZE_BLEU" "$ZE_COMET" "$EZ_BLEU" "$EZ_COMET" \
    >> results.tsv

echo "===== CHAIN DONE $(date '+%H:%M') ====="
echo "zh-en: BLEU=$ZE_BLEU COMET=$ZE_COMET"
echo "en-zh: BLEU=$EZ_BLEU COMET=$EZ_COMET"
column -t -s$'\t' results.tsv | tail -3

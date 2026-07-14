#!/bin/bash
# Post-QAT evaluation: materialize the 1.25-bit model into a plain Qwen3
# checkpoint, then run the full multi-testset COMET eval (WMT23/24 + Flores)
# with vLLM -- bit-identical to the fake-quant forward, but fast.
#
# Usage:  ./run_eval.sh [model_dir]      (default: ./output_qat_125bit)
set -e
cd "$(dirname "$0")"
PY=/home/tfbao/new/HY-MT/.venv/bin/python
MODEL=${1:-./output_qat_125bit}
MAT=${MODEL}_mat

echo "=== materialize $MODEL -> $MAT ==="
CUDA_VISIBLE_DEVICES="" $PY -u materialize.py --in "$MODEL" --out "$MAT"

echo "=== multi-testset COMET eval (1.25-bit QAT model) ==="
$PY -u ../eval/eval_multi.py --model_path "$MAT"

echo
echo "FP baseline (output_1.7b_grpo_full): wmt23 0.8054/0.8542 | wmt24 en-zh 0.8433"
echo "                                     flores 0.8649/0.8839"
echo "For the no-QAT RTN baseline:  $PY ../eval/eval_multi.py --model_path ./output_qat_rtn_mat"

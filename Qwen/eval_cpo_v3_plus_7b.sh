#!/bin/bash
set -e
cd /home/tfbao/Shiyu/Interpreter/Qwen
PY=/home/tfbao/new/HY-MT/.venv/bin/python

# Merge LoRA and eval final
$PY -u merge_lora_qwen3.py \
  --base_model_path ./output_1.7b_base_v2 \
  --adapter_model_path ./output_1.7b_cpo_v3_plus_7b \
  --output_path ./output_1.7b_cpo_v3_plus_7b_merged \
  --save_dtype bf16

$PY -u eval_vllm.py --model_path ./output_1.7b_cpo_v3_plus_7b_merged \
  --testset wmt23 --direction both

# Also eval checkpoint-1000 as backup (more stable phase)
$PY -u merge_lora_qwen3.py \
  --base_model_path ./output_1.7b_base_v2 \
  --adapter_model_path ./output_1.7b_cpo_v3_plus_7b/checkpoint-1000 \
  --output_path ./output_1.7b_cpo_v3_plus_7b_ckpt1k_merged \
  --save_dtype bf16

$PY -u eval_vllm.py --model_path ./output_1.7b_cpo_v3_plus_7b_ckpt1k_merged \
  --testset wmt23 --direction both

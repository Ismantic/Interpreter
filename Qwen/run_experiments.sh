#!/bin/bash
# Helper commands for the 10-hour experiment sequence
# (NOT for batch running — use individually after each task)

PY=/home/tfbao/new/HY-MT/.venv/bin/python
cd /home/tfbao/Shiyu/Interpreter/Qwen

# === Eval functions ===
eval_full() {
  local MODEL=$1
  $PY -u eval_vllm.py --model_path $MODEL --testset wmt23 --direction both
}

eval_lora() {
  local LORA=$1
  local OUT=${2:-${LORA}_merged}
  $PY -u merge_lora_qwen3.py --base_model_path ./output_1.7b_base_v2 --adapter_model_path $LORA --output_path $OUT --save_dtype bf16
  $PY -u eval_vllm.py --model_path $OUT --testset wmt23 --direction both
}

# === CPO + 7B chosen (LoRA) — Task #5 ===
cpo_7b_lora() {
  $PY -u train_cpo.py \
    --model_path ./output_1.7b_base_v2 \
    --data_path ./cpo_hymt7b_vs_ours.jsonl \
    --output_dir ./output_1.7b_cpo_7b_lora \
    --lr 1e-4 \
    --beta 0.05 \
    --nll_weight 1.0 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_length 512 \
    --logging_steps 50 \
    --save_steps 500
}

# === CPO + 7B chosen (Full) — Task #8 ===
cpo_7b_full() {
  $PY -u train_cpo.py \
    --model_path ./output_1.7b_base_v2 \
    --data_path ./cpo_hymt7b_vs_ours.jsonl \
    --output_dir ./output_1.7b_cpo_7b_full \
    --lr 1e-5 \
    --beta 0.05 \
    --nll_weight 1.0 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_length 512 \
    --logging_steps 50 \
    --save_steps 500 \
    --full_finetune
}

# === SFT + COIG-CQIA — Task #6 ===
sft_cqia() {
  $PY -u train.py \
    --model_path ./Qwen3-1.7B-Base \
    --train_data ./sft_with_cqia.jsonl \
    --output_dir ./output_1.7b_cqia \
    --max_seq_length 512 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --num_epochs 1 \
    --lr 2e-5 \
    --min_lr 1e-6 \
    --lr_scheduler inverse_sqrt \
    --warmup_ratio 0.01 \
    --weight_decay 0.01 \
    --logging_steps 200 \
    --save_steps 2000
}

# === SFT + OpenAssistant — Task #7 ===
sft_oasst() {
  $PY -u train.py \
    --model_path ./Qwen3-1.7B-Base \
    --train_data ./sft_with_oasst.jsonl \
    --output_dir ./output_1.7b_oasst \
    --max_seq_length 512 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --num_epochs 1 \
    --lr 2e-5 \
    --min_lr 1e-6 \
    --lr_scheduler inverse_sqrt \
    --warmup_ratio 0.01 \
    --weight_decay 0.01 \
    --logging_steps 200 \
    --save_steps 2000
}

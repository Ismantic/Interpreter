#!/bin/bash
VENV=/home/tfbao/new/HY-MT/.venv/bin/python
DATA_CN=/home/tfbao/Shiyu/Tokenizer/data/cn_sentences.txt
DATA_EN=/home/tfbao/Shiyu/Tokenizer/data/en_sentences.txt

# Phase 1: freeze transformer, train embedding only
$VENV -u finetune_muon.py \
    --model_path ./HY-MT1.5-1.8B-new-tok \
    --train_data ./private/train_data.pt \
    --output_dir ./private/phase1_v5 \
    --freeze_transformer \
    --gradient_checkpointing \
    --max_seq_length 384 \
    --batch_size 32 \
    --gradient_accumulation_steps 8 \
    --max_steps 2000 \
    --warmup_steps 100 \
    --adam_lr 1e-4 \
    --logging_steps 10 \
    --save_steps 0

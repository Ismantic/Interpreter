#!/bin/bash
# Phase 1 experiments: find optimal training config
PYTHON=/home/tfbao/new/HY-MT/.venv/bin/python
MODEL=./HY-MT1.5-1.8B-new-tok
DATA=./private/pseudo_mono_wiki.pt
EVAL_ARGS="--testset wmt23 --direction both --max_samples 200 --no_comet --batch_size 8"

run_and_eval() {
    local name=$1
    local output_dir=./private/exp_${name}
    shift
    echo "=========================================="
    echo "Experiment: $name"
    echo "=========================================="
    $PYTHON -u finetune_muon.py \
        --model_path $MODEL \
        --train_data $DATA \
        --output_dir $output_dir \
        --freeze_transformer \
        --gradient_checkpointing \
        --max_seq_length 384 \
        --logging_steps 50 \
        --save_steps 0 \
        "$@"

    # Copy tokenizer files
    cp $MODEL/piece.model $output_dir/
    cp $MODEL/token_mapping.json $output_dir/
    cp $MODEL/generation_config.json $output_dir/

    # Eval
    echo "--- Eval: $name ---"
    $PYTHON -u eval.py --model_path $output_dir $EVAL_ARGS 2>&1 | grep -E "BLEU|COMET|Summary|zh-en|en-zh"
    echo ""
}

# Experiment 1: Lower LR (5e-5)
run_and_eval "lr5e5_200step" \
    --batch_size 32 --gradient_accumulation_steps 8 \
    --max_steps 200 --warmup_steps 20 --adam_lr 5e-5

# Experiment 2: Even lower LR (3e-5)
run_and_eval "lr3e5_200step" \
    --batch_size 32 --gradient_accumulation_steps 8 \
    --max_steps 200 --warmup_steps 20 --adam_lr 3e-5

# Experiment 3: Original LR but 100 steps only
run_and_eval "lr1e4_100step" \
    --batch_size 32 --gradient_accumulation_steps 8 \
    --max_steps 100 --warmup_steps 10 --adam_lr 1e-4

# Experiment 4: Larger batch (32x16 instead of 32x8)
run_and_eval "bigbatch_200step" \
    --batch_size 32 --gradient_accumulation_steps 16 \
    --max_steps 200 --warmup_steps 20 --adam_lr 1e-4

# Experiment 5: Chat template format data
CHAT_DATA=./private/pseudo_chat.pt
run_and_eval_chat() {
    local name=$1
    local output_dir=./private/exp_${name}
    shift
    echo "=========================================="
    echo "Experiment: $name"
    echo "=========================================="
    $PYTHON -u finetune_muon.py \
        --model_path $MODEL \
        --train_data $CHAT_DATA \
        --output_dir $output_dir \
        --freeze_transformer \
        --gradient_checkpointing \
        --max_seq_length 384 \
        --logging_steps 50 \
        --save_steps 0 \
        "$@"

    cp $MODEL/piece.model $output_dir/
    cp $MODEL/token_mapping.json $output_dir/
    cp $MODEL/generation_config.json $output_dir/

    echo "--- Eval: $name ---"
    $PYTHON -u eval.py --model_path $output_dir $EVAL_ARGS 2>&1 | grep -E "BLEU|COMET|Summary|zh-en|en-zh"
    echo ""
}

run_and_eval_chat "chat_lr1e4_200step" \
    --batch_size 32 --gradient_accumulation_steps 8 \
    --max_steps 200 --warmup_steps 20 --adam_lr 1e-4

run_and_eval_chat "chat_lr5e5_200step" \
    --batch_size 32 --gradient_accumulation_steps 8 \
    --max_steps 200 --warmup_steps 20 --adam_lr 5e-5

echo "=========================================="
echo "ALL EXPERIMENTS DONE"
echo "=========================================="

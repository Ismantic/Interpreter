"""
CPO (Contrastive Preference Optimization) training for translation.
Based on ALMA-R: no reference model needed.

Usage:
    python train_cpo.py \
        --model_path ./output_1.7b_base_v2 \
        --data_path ./cpo_preference.jsonl \
        --output_dir ./output_1.7b_cpo
"""
import os
import json
import argparse

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def load_preference_data(data_path, tokenizer, max_length=512):
    """Load preference data and format for CPO trainer."""
    examples = []
    with open(data_path, 'r', encoding='utf8') as f:
        for line in f:
            ex = json.loads(line)
            # Format as ChatML
            prompt = f"<|im_start|>user\n{ex['prompt']}<|im_end|>\n<|im_start|>assistant\n"
            chosen = ex['chosen'] + "<|im_end|>"
            rejected = ex['rejected'] + "<|im_end|>"
            examples.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            })
    return Dataset.from_list(examples)


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
    )

    # Use LoRA for CPO (ALMA-R style: rank=16, all linear layers)
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_preference_data(args.data_path, tokenizer)
    print(f"Loaded {len(dataset)} preference pairs")

    # Split train/eval
    split = dataset.train_test_split(test_size=0.02, seed=42)
    train_dataset = split['train']
    eval_dataset = split['test']
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=args.lr,
        lr_scheduler_type="inverse_sqrt",
        warmup_ratio=0.01,
        bf16=True,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        max_length=args.max_length,
        beta=args.beta,
        loss_type="sigmoid",
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=True,
    )

    # CPO = DPO without reference model
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="./cpo_preference.jsonl")
    parser.add_argument("--output_dir", type=str, default="./output_1.7b_cpo")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()
    main(args)

"""GRPO training for translation — TRL GRPOTrainer.

Reward = reference-based COMET (wmt22-comet-da) + repetition penalty.
Full-parameter or LoRA via --full_finetune flag.
"""
import os
import re
import argparse
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

COMET_CKPT = os.path.expanduser(
    "~/.cache/comet/models--Unbabel--wmt22-comet-da/snapshots/2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt"
)

_comet = None


def get_comet():
    global _comet
    if _comet is None:
        from comet import load_from_checkpoint
        _comet = load_from_checkpoint(COMET_CKPT)
        _comet.eval()
    return _comet


def extract_text(completion):
    """completion may be a string or a conversational list."""
    if isinstance(completion, str):
        return completion.strip()
    # conversational: list of {role, content}
    return completion[-1]["content"].strip()


_reward_call = 0


def comet_reward(prompts, completions, source, reference, **kwargs):
    """Reference-based COMET score as reward."""
    global _reward_call
    comet = get_comet()
    mts = [extract_text(c) for c in completions]
    data = [{"src": s, "mt": mt, "ref": r}
            for s, mt, r in zip(source, mts, reference)]
    scores = [float(s) for s in
              comet.predict(data, batch_size=64, gpus=1, progress_bar=False).scores]
    _reward_call += 1
    avg = sum(scores) / len(scores)
    lens = [len(m.split()) for m in mts]
    print(f"[reward#{_reward_call}] n={len(scores)} comet avg={avg:.4f} "
          f"min={min(scores):.4f} max={max(scores):.4f} "
          f"len avg={sum(lens)/len(lens):.0f}", flush=True)
    return scores


def repetition_penalty(prompts, completions, **kwargs):
    """Penalize repetitive n-gram patterns. Returns 0 (clean) to -1 (very repetitive)."""
    rewards = []
    for c in completions:
        text = extract_text(c)
        tokens = text.split()
        if len(tokens) < 8:
            rewards.append(0.0)
            continue
        # 4-gram repetition rate
        ngrams = [tuple(tokens[i:i+4]) for i in range(len(tokens)-3)]
        if not ngrams:
            rewards.append(0.0)
            continue
        unique_ratio = len(set(ngrams)) / len(ngrams)
        # unique_ratio 1.0 = no rep -> 0 penalty; 0.5 -> -0.5
        rewards.append(min(0.0, (unique_ratio - 1.0)))
    return rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./output_1.7b_cpo_v3_plus_7b_merged")
    parser.add_argument("--data_path", default="./grpo_data.jsonl")
    parser.add_argument("--output_dir", default="./output_1.7b_grpo")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_completion_length", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--full_finetune", action="store_true")
    parser.add_argument("--max_prompts", type=int, default=0, help="0 = all")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # ChatML uses <|im_end|> as turn end; make generation stop there
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    tokenizer.eos_token = "<|im_end|>"
    print(f"eos set to <|im_end|> ({im_end})")

    dataset = load_dataset("json", data_files=args.data_path, split="train")
    if args.max_prompts > 0:
        dataset = dataset.select(range(min(args.max_prompts, len(dataset))))
    print(f"GRPO prompts: {len(dataset)}")

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    # Force generation to stop at <|im_end|>, not <|endoftext|>
    model.config.eos_token_id = im_end
    model.generation_config.eos_token_id = im_end
    print(f"model eos_token_id forced to {im_end}")

    peft_config = None
    if not args.full_finetune:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0,
            target_modules="all-linear", task_type="CAUSAL_LM",
        )

    config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        beta=args.beta,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=1,
        temperature=1.0,
        loss_type="dapo",
        scale_rewards="group",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        optim="adafactor" if args.full_finetune else "adamw_torch",
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.2,
        vllm_max_model_length=1024,
        reward_weights=[1.0, 0.3],
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[comet_reward, repetition_penalty],
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()

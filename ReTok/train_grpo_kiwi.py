"""GRPO with dual reward: wmt22-comet-da (ref-based) + wmt22-cometkiwi-da (ref-free QE).

Variant of train_grpo.py — adds cometkiwi as a 3rd reward (alongside COMET and
repetition penalty). Motivation: COMET alone can be hacked toward ref-style;
cometkiwi is reference-free (QE), so it scores intrinsic quality without seeing
the reference. Combining both should give a stronger, less ref-biased signal.

Reward weights: [comet 1.0, cometkiwi 1.0, repetition 0.3].
Both COMET-family models score 0~1, so equal weights are natural.

Usage:
    bash run_grpo_kiwi.sh
"""
import os
import sys
import argparse

# Monkey-patch vLLM BEFORE TRL import (same as train_grpo.py)
import vllm as _vllm  # noqa: E402
_original_LLM_init = _vllm.LLM.__init__
def _patched_LLM_init(self, *args, **kwargs):
    kwargs.setdefault("skip_tokenizer_init", True)
    kwargs.setdefault("trust_remote_code", True)
    return _original_LLM_init(self, *args, **kwargs)
_vllm.LLM.__init__ = _patched_LLM_init

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from piece_hf_tokenizer import PieceTokenizerForTRL  # noqa: E402
from train import _copy_tokenizer_artifacts  # noqa: E402

COMET_CKPT = os.path.expanduser(
    "~/.cache/comet/models--Unbabel--wmt22-comet-da/snapshots/"
    "2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt"
)
COMETKIWI_CKPT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Unbabel--wmt22-cometkiwi-da/snapshots/"
    "1ad785194e391eebc6c53e2d0776cada8f83179a/checkpoints/model.ckpt"
)

_comet = None
_kiwi = None


def get_comet():
    global _comet
    if _comet is None:
        from comet import load_from_checkpoint
        _comet = load_from_checkpoint(COMET_CKPT)
        _comet.eval()
    return _comet


def get_kiwi():
    global _kiwi
    if _kiwi is None:
        from comet import load_from_checkpoint
        _kiwi = load_from_checkpoint(COMETKIWI_CKPT)
        _kiwi.eval()
    return _kiwi


def extract_text(completion):
    if isinstance(completion, str):
        return completion.strip()
    return completion[-1]["content"].strip()


_reward_call = 0


def comet_reward(prompts, completions, source, reference, **kwargs):
    """Reference-based COMET (wmt22-comet-da)."""
    global _reward_call
    comet = get_comet()
    mts = [extract_text(c) for c in completions]
    data = [{"src": s, "mt": mt, "ref": r}
            for s, mt, r in zip(source, mts, reference)]
    scores = [float(s) for s in
              comet.predict(data, batch_size=64, gpus=1, progress_bar=False).scores]
    _reward_call += 1
    avg = sum(scores) / len(scores)
    print(f"[reward#{_reward_call}] n={len(scores)} COMET avg={avg:.4f} "
          f"[{min(scores):.4f}, {max(scores):.4f}]", flush=True)
    return scores


def cometkiwi_reward(prompts, completions, source, **kwargs):
    """Reference-FREE QE (wmt22-cometkiwi-da). Only src + mt, no ref."""
    kiwi = get_kiwi()
    mts = [extract_text(c) for c in completions]
    data = [{"src": s, "mt": mt} for s, mt in zip(source, mts)]
    scores = [float(s) for s in
              kiwi.predict(data, batch_size=64, gpus=1, progress_bar=False).scores]
    avg = sum(scores) / len(scores)
    print(f"             KIWI  avg={avg:.4f} "
          f"[{min(scores):.4f}, {max(scores):.4f}]", flush=True)
    return scores


def repetition_penalty(prompts, completions, **kwargs):
    rewards = []
    for c in completions:
        text = extract_text(c)
        tokens = text.split()
        if len(tokens) < 8:
            rewards.append(0.0)
            continue
        ngrams = [tuple(tokens[i:i + 4]) for i in range(len(tokens) - 3)]
        if not ngrams:
            rewards.append(0.0)
            continue
        unique_ratio = len(set(ngrams)) / len(ngrams)
        rewards.append(min(0.0, (unique_ratio - 1.0)))
    return rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", default="./data/grpo_data.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_completion_length", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--full_finetune", action="store_true")
    parser.add_argument("--max_prompts", type=int, default=0)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.2)
    parser.add_argument("--vllm_max_model_length", type=int, default=1024)
    parser.add_argument("--comet_weight", type=float, default=1.0)
    parser.add_argument("--kiwi_weight", type=float, default=1.0)
    parser.add_argument("--rep_weight", type=float, default=0.3)
    args = parser.parse_args()

    tokenizer = PieceTokenizerForTRL(args.model_path)
    print(f"Tokenizer: vocab={tokenizer.vocab_size}, eos={tokenizer.eos_token_id}")

    dataset = load_dataset("json", data_files=args.data_path, split="train")
    if args.max_prompts > 0:
        dataset = dataset.select(range(min(args.max_prompts, len(dataset))))
    print(f"GRPO prompts: {len(dataset)}")

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    model.config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    print(f"model eos_token_id forced to {tokenizer.eos_token_id} (</s>)")

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
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_length=args.vllm_max_model_length,
        reward_weights=[args.comet_weight, args.kiwi_weight, args.rep_weight],
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[comet_reward, cometkiwi_reward, repetition_penalty],
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    _copy_tokenizer_artifacts(args.model_path, args.output_dir)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()

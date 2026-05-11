"""
Translation SFT training for Qwen3-0.6B.
Minimal script: full fine-tuning with mask_prompt on translation data.

Usage:
    python train.py --train_data ../private/sft_distill_ft.jsonl --output_dir ./output_v1
"""
import os
import sys
import json
import math
import time
import argparse
import signal

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

IGNORE_INDEX = -100


class TranslationSFTDataset(Dataset):
    """ChatML-format translation dataset for Qwen3 (base or instruct).
    Format: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>
    Loss only on assistant response (after <|im_start|>assistant\n).
    Uses <|im_end|> as eos (following LLaMA-Factory's replace_eos=True for Qwen3).
    """

    def __init__(self, data_file, tokenizer, max_seq_length=512):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

        with open(data_file, 'r', encoding='utf8') as f:
            self.data = [json.loads(line) for line in f if line.strip()]
        print(f"Loaded {len(self.data)} samples from {data_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        messages = data['messages']
        user_content = messages[0]['content']
        asst_content = messages[1]['content']

        # Build ChatML: <|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>
        prefix = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        response_ids = self.tokenizer.encode(asst_content, add_special_tokens=False)

        # response + <|im_end|> as eos
        token_ids = prefix_ids + response_ids + [self.im_end_id]
        token_ids = token_ids[:self.max_seq_length]
        tokens = torch.tensor(token_ids, dtype=torch.long)

        # Labels: IGNORE_INDEX for prefix, actual tokens for response + im_end
        labels = torch.full_like(tokens, IGNORE_INDEX)
        prefix_len = min(len(prefix_ids), len(tokens))
        labels[prefix_len:] = tokens[prefix_len:]

        attention_mask = torch.ones_like(tokens, dtype=torch.bool)
        return dict(input_ids=tokens, labels=labels, attention_mask=attention_mask)


def collate_fn(batch, pad_token_id):
    input_ids = [b['input_ids'] for b in batch]
    labels = [b['labels'] for b in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    attention_mask = input_ids.ne(pad_token_id)
    return dict(input_ids=input_ids, labels=labels, attention_mask=attention_mask)


def train(args):
    device = torch.device("cuda")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
    ).to(device)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Freeze embedding if requested (TranslateGemma approach)
    if args.freeze_embedding:
        for name, param in model.named_parameters():
            if "embed" in name or "lm_head" in name:
                param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total_params/1e6:.1f}M params, {trainable_params/1e6:.1f}M trainable")

    # Dataset
    dataset = TranslationSFTDataset(args.train_data, tokenizer, args.max_seq_length)
    pad_token_id = tokenizer.pad_token_id

    # Val split
    val_size = max(100, len(dataset) // 100)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42))
    print(f"Split: {train_size} train / {val_size} val")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id),
    )

    # Optimizer
    if args.optimizer == "adafactor":
        from transformers import Adafactor
        optimizer = Adafactor(
            model.parameters(), lr=args.lr, relative_step=False,
            scale_parameter=False, warmup_init=False, weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, betas=(0.9, 0.95),
            eps=1e-8, weight_decay=args.weight_decay,
        )

    # LR scheduler
    min_lr_ratio = args.min_lr / args.lr if args.lr > 0 else 0
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        if args.lr_scheduler == "inverse_sqrt":
            return max(min_lr_ratio, (args.warmup_steps / max(step, 1)) ** 0.5)
        else:  # cosine
            progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Compute max_steps from epochs if not set
    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    if args.max_steps <= 0:
        args.max_steps = steps_per_epoch * args.num_epochs
    if args.warmup_steps <= 0:
        args.warmup_steps = max(1, int(args.max_steps * args.warmup_ratio))
    print(f"Training: {args.max_steps} steps ({args.num_epochs} epochs, {steps_per_epoch} steps/epoch), warmup={args.warmup_steps}")

    # Training loop
    model.train()
    step = 0
    micro_step = 0
    t0 = time.time()
    interrupted = False

    def _sigint_handler(sig, frame):
        nonlocal interrupted
        if interrupted:
            raise KeyboardInterrupt
        interrupted = True
        print(f"\nCtrl+C at step {step}, saving...")
    signal.signal(signal.SIGINT, _sigint_handler)

    while step < args.max_steps and not interrupted:
        for batch in train_loader:
            if step >= args.max_steps or interrupted:
                break
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            micro_step += 1

            if micro_step % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % args.logging_steps == 0:
                    elapsed = time.time() - t0
                    lr = optimizer.param_groups[0]['lr']
                    real_loss = loss.item() * args.gradient_accumulation_steps
                    print(f"step {step}/{args.max_steps} | loss {real_loss:.4f} | "
                          f"lr {lr:.2e} | {elapsed:.0f}s")

                if args.save_steps > 0 and step % args.save_steps == 0:
                    # Val loss
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for vb in val_loader:
                            vb = {k: v.to(device) for k, v in vb.items()}
                            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                                vl = model(**vb).loss.item()
                            val_losses.append(vl)
                    val_loss = sum(val_losses) / len(val_losses)
                    print(f"step {step} | val_loss {val_loss:.4f}")
                    model.train()

                    # Save
                    save_path = os.path.join(args.output_dir, f"checkpoint-{step}")
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    print(f"Saved to {save_path}")

                    # Inline eval
                    if args.eval_steps > 0 and step % args.eval_steps == 0:
                        import subprocess
                        model.cpu()
                        torch.cuda.empty_cache()
                        eval_dir = args.eval_direction or "en-zh"
                        cmd = [sys.executable, "-u",
                               os.path.join(os.path.dirname(__file__), "eval.py"),
                               "--model_path", save_path,
                               "--testset", "wmt22",
                               "--direction", eval_dir,
                               "--batch_size", "8",
                               "--max_samples", "500"]
                        print(f"[eval] Running {eval_dir}...")
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                        for line in result.stdout.split('\n'):
                            if any(k in line for k in ['BLEU', 'COMET', 'zh-en:', 'en-zh:']):
                                print(f"[eval] {line.strip()}")
                        if result.returncode != 0:
                            print(f"[eval] ERROR: {result.stderr[-300:]}")
                        model.to(device)

    # Final save
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Saved final model to {args.output_dir}")

    elapsed = time.time() - t0
    print(f"Training complete: {step} steps in {elapsed:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translation SFT for Qwen3")
    parser.add_argument("--model_path", type=str, default="./Qwen3-0.6B")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output_v1")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--freeze_embedding", action="store_true",
                        help="Freeze embed_tokens + lm_head (TranslateGemma approach)")
    parser.add_argument("--max_steps", type=int, default=0, help="Max steps (0 = use num_epochs)")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=0, help="Override warmup_ratio if > 0")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--lr_scheduler", type=str, default="inverse_sqrt",
                        choices=["inverse_sqrt", "cosine"])
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "adafactor"])
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=0,
                        help="Run WMT22 eval every N steps (must align with save_steps)")
    parser.add_argument("--eval_direction", type=str, default="en-zh",
                        choices=["zh-en", "en-zh", "both"])
    args = parser.parse_args()
    train(args)

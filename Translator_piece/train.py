"""
Translation SFT for the ReTok phase2 v18 base + PieceTokenizer.

Tokenizer-only fork of ../Translator/train.py — every hyperparameter, optimizer,
LR schedule, loss-mask boundary, and gradient-accumulation behavior is preserved
verbatim from the Translator best-practice recipe (AdamW lr=2e-5, inverse_sqrt,
1 epoch, bs=2*ga=8, gradient checkpointing). The ONLY differences are:

  - Tokenizer: AutoTokenizer (Qwen3 ChatML) → PieceTokenizerWrapper
  - Chat template: <|im_start|>user\\n{u}<|im_end|>\\n<|im_start|>assistant\\n{r}<|im_end|>
                   → <bos> <user> {u_ids} <assistant> {r_ids} <eos>
  - Eos for loss/generation: <|im_end|> → </s> (id=2 in PieceTokenizer)
  - Checkpoint save: tokenizer.save_pretrained → _copy_tokenizer_artifacts (5 files)

Usage (matches Translator/train.py CLI 1:1):
    python train.py \\
        --model_path /home/tfbao/Shiyu/Summer/output/phase2_ckpt_v18 \\
        --train_data ../Translator/alma_combined_sft_clean.jsonl \\
        --output_dir ./output_v18_sft \\
        --num_epochs 1 --lr 2e-5 --lr_scheduler inverse_sqrt \\
        --batch_size 2 --gradient_accumulation_steps 8 --gradient_checkpointing
"""
import os
import sys
import json
import math
import time
import shutil
import argparse
import signal

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM

# Pull PieceTokenizerWrapper from the parent Interpreter repo (sibling to this dir)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer_wrapper import PieceTokenizerWrapper  # noqa: E402

IGNORE_INDEX = -100

_TOKENIZER_ARTIFACTS = [
    "piece.model",
    "dict.txt",
    "token_mapping.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
]


def _copy_tokenizer_artifacts(base_dir, save_dir):
    """Mirror PieceTokenizer's 5 files from the original base dir into a save dir.
    Translator's HF-tokenizer path used tokenizer.save_pretrained(); piece has no
    such hook, so each ckpt must be self-contained the same way Summer/Interpreter
    write theirs.
    """
    os.makedirs(save_dir, exist_ok=True)
    for fn in _TOKENIZER_ARTIFACTS:
        src = os.path.join(base_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(save_dir, fn))


class TranslationSFTDataset(Dataset):
    """PieceTokenizer chat-format translation dataset.

    Format: <bos> <user> {user_ids} <assistant> {response_ids} <eos>
    Loss only on the assistant response (after <assistant>), including <eos>,
    mirroring Translator's behavior of supervising on `response + <|im_end|>`.
    """

    def __init__(self, data_file, tokenizer, max_seq_length=512):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.bos_id = tokenizer.bos_token_id
        self.user_id = tokenizer.user_token_id
        self.asst_id = tokenizer.assistant_token_id
        self.eos_id = tokenizer.eos_token_id

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

        user_ids = self.tokenizer.encode(user_content, add_special_tokens=False)
        response_ids = self.tokenizer.encode(asst_content, add_special_tokens=False)

        prefix_ids = [self.bos_id, self.user_id] + user_ids + [self.asst_id]
        token_ids = prefix_ids + response_ids + [self.eos_id]
        token_ids = token_ids[:self.max_seq_length]
        tokens = torch.tensor(token_ids, dtype=torch.long)

        # Loss is masked out for prefix (bos + user + content + assistant marker)
        # and supervised on response + eos, exactly like Translator's prefix split.
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
    tokenizer = PieceTokenizerWrapper(args.model_path)
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
                    _copy_tokenizer_artifacts(args.model_path, save_path)
                    print(f"Saved to {save_path}")

                    # Inline eval (disabled for piece path until we port a piece-aware eval)
                    if args.eval_steps > 0 and step % args.eval_steps == 0:
                        print("[eval] inline eval not yet wired for piece tokenizer; skip.")

    # Final save
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
        _copy_tokenizer_artifacts(args.model_path, args.output_dir)
        print(f"Saved final model to {args.output_dir}")

    elapsed = time.time() - t0
    print(f"Training complete: {step} steps in {elapsed:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translation SFT for PieceTokenizer base")
    parser.add_argument("--model_path", type=str, required=True)
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
                        help="Inline eval every N steps (disabled until piece eval is wired)")
    parser.add_argument("--eval_direction", type=str, default="en-zh",
                        choices=["zh-en", "en-zh", "both"])
    args = parser.parse_args()
    train(args)

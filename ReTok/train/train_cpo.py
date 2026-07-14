"""
CPO (Contrastive Preference Optimization) for PieceTokenizer translation models.

Tokenizer-only fork of ../Qwen/train_cpo.py — the CPO loss math
L = -log σ(β·(logπ(yw|x) - logπ(yl|x))) + λ·NLL(yw|x), LoRA r=16 all-linear,
AdamW betas=(0.9,0.95) wd=0.01, inverse_sqrt schedule, and all default
hyperparameters are preserved verbatim. Only differences:

  - Tokenizer: AutoTokenizer → PieceTokenizerWrapper
  - ChatML prefix → <bos> <user> {prompt_ids} <assistant>;
    eos used for chosen/rejected response = </s> (id=2) instead of <|im_end|>
  - tokenizer.save_pretrained → _copy_tokenizer_artifacts (5 piece files)

Usage (matches Qwen/train_cpo.py CLI):
    python train_cpo.py \\
        --model_path ./output_v18_sft \\
        --data_path ./data/cpo_v3_plus_7b.jsonl \\
        --output_dir ./output_v18_cpo_v3_plus_7b \\
        --lr 1e-4 --beta 0.05 --nll_weight 1.0 \\
        --batch_size 1 --gradient_accumulation_steps 16 --max_length 512
"""
import os
import sys
import json
import time
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# Reuse ReTok/train.py helpers (tokenizer wrapper + 5-file copy)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
from tokenizer_wrapper import PieceTokenizerWrapper  # noqa: E402
from tok_artifacts import _copy_tokenizer_artifacts  # noqa: E402

IGNORE_INDEX = -100


class CPODataset(Dataset):
    """Preference dataset built with PieceTokenizer chat template.
    Each example: <bos> <user> {prompt} <assistant> {chosen|rejected} <eos>
    Loss-mask labels: IGNORE_INDEX for prefix, response tokens supervised.
    """
    def __init__(self, data_path, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bos_id = tokenizer.bos_token_id
        self.user_id = tokenizer.user_token_id
        self.asst_id = tokenizer.assistant_token_id
        self.eos_id = tokenizer.eos_token_id

        with open(data_path, 'r', encoding='utf8') as f:
            self.data = [json.loads(line) for line in f if line.strip()]
        print(f"Loaded {len(self.data)} preference pairs")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        prompt_ids = self.tokenizer.encode(ex['prompt'], add_special_tokens=False)
        prefix_ids = [self.bos_id, self.user_id] + prompt_ids + [self.asst_id]

        chosen_resp = self.tokenizer.encode(ex['chosen'], add_special_tokens=False) + [self.eos_id]
        rejected_resp = self.tokenizer.encode(ex['rejected'], add_special_tokens=False) + [self.eos_id]

        max_resp = max(1, self.max_length - len(prefix_ids))
        chosen_resp = chosen_resp[:max_resp]
        rejected_resp = rejected_resp[:max_resp]

        chosen_input = prefix_ids + chosen_resp
        rejected_input = prefix_ids + rejected_resp
        chosen_labels = [IGNORE_INDEX] * len(prefix_ids) + chosen_resp
        rejected_labels = [IGNORE_INDEX] * len(prefix_ids) + rejected_resp

        return {
            'chosen_ids': torch.tensor(chosen_input, dtype=torch.long),
            'chosen_labels': torch.tensor(chosen_labels, dtype=torch.long),
            'rejected_ids': torch.tensor(rejected_input, dtype=torch.long),
            'rejected_labels': torch.tensor(rejected_labels, dtype=torch.long),
        }


def collate_fn(batch, pad_id):
    def pad_tensors(tensors, pad_value):
        max_len = max(t.size(0) for t in tensors)
        padded = torch.full((len(tensors), max_len), pad_value, dtype=tensors[0].dtype)
        for i, t in enumerate(tensors):
            padded[i, :t.size(0)] = t
        return padded

    return {
        'chosen_ids': pad_tensors([b['chosen_ids'] for b in batch], pad_id),
        'chosen_labels': pad_tensors([b['chosen_labels'] for b in batch], IGNORE_INDEX),
        'rejected_ids': pad_tensors([b['rejected_ids'] for b in batch], pad_id),
        'rejected_labels': pad_tensors([b['rejected_labels'] for b in batch], IGNORE_INDEX),
    }


def compute_logps(model, input_ids, labels, pad_id):
    """Per-sequence log-prob of labels (sum over supervised tokens) + per-token NLL.
    Identical to Qwen/train_cpo.py:compute_logps."""
    attention_mask = input_ids.ne(pad_id)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.clamp(min=0).unsqueeze(2)).squeeze(2)

    mask = shift_labels.ne(IGNORE_INDEX).float()
    seq_log_probs = (token_log_probs * mask).sum(dim=1)
    nll = -(token_log_probs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return seq_log_probs, nll


def cpo_loss(chosen_logps, rejected_logps, chosen_nll, beta=0.1, nll_weight=1.0):
    """CPO loss = -log σ(β·(logπ_w - logπ_l)) + λ·mean(NLL_chosen). Verbatim."""
    logits_diff = beta * (chosen_logps - rejected_logps)
    preference_loss = -F.logsigmoid(logits_diff).mean()
    bc_loss = chosen_nll.mean()
    return preference_loss + nll_weight * bc_loss, preference_loss.item(), bc_loss.item()


def main(args):
    device = torch.device("cuda")

    tokenizer = PieceTokenizerWrapper(args.model_path)
    pad_id = tokenizer.pad_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
    ).to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # LoRA or full fine-tuning (Qwen finding: full-param CPO collapses; use LoRA)
    if not args.full_finetune:
        lora_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({trainable/total*100:.1f}%)")

    dataset = CPODataset(args.data_path, tokenizer, args.max_length)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=lambda batch: collate_fn(batch, pad_id),
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01,
    )

    warmup_steps = max(1, int(len(dataloader) * args.warmup_ratio / args.gradient_accumulation_steps))
    max_steps = len(dataloader) // args.gradient_accumulation_steps
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.01, (warmup_steps / max(step, 1)) ** 0.5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"Training: {max_steps} steps, warmup={warmup_steps}")

    model.train()
    step = 0
    micro_step = 0
    t0 = time.time()

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        chosen_logps, chosen_nll = compute_logps(model, batch['chosen_ids'], batch['chosen_labels'], pad_id)
        rejected_logps, rejected_nll = compute_logps(model, batch['rejected_ids'], batch['rejected_labels'], pad_id)

        loss, pref_loss, bc_loss = cpo_loss(
            chosen_logps, rejected_logps, chosen_nll,
            beta=args.beta, nll_weight=args.nll_weight,
        )

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()
        micro_step += 1

        if micro_step % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            with torch.no_grad():
                acc = (chosen_logps > rejected_logps).float().mean().item()

            if step % args.logging_steps == 0:
                elapsed = time.time() - t0
                lr = optimizer.param_groups[0]['lr']
                print(f"step {step}/{max_steps} | loss {loss.item()*args.gradient_accumulation_steps:.4f} "
                      f"| pref {pref_loss:.4f} | bc {bc_loss:.4f} | acc {acc:.3f} "
                      f"| lr {lr:.2e} | {elapsed:.0f}s")

            if args.save_steps > 0 and step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{step}")
                os.makedirs(save_path, exist_ok=True)
                model.save_pretrained(save_path)
                _copy_tokenizer_artifacts(args.model_path, save_path)
                print(f"Saved to {save_path}")

    # Final save
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    _copy_tokenizer_artifacts(args.model_path, args.output_dir)
    elapsed = time.time() - t0
    print(f"Training complete: {step} steps in {elapsed:.0f}s")
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--nll_weight", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--full_finetune", action="store_true",
                        help="Full fine-tune instead of LoRA (NOT recommended — collapses model per Qwen)")
    args = parser.parse_args()
    main(args)

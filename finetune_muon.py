"""
Minimal fine-tuning script for HY-MT models with Muon optimizer.
Dependencies: torch, transformers (model/tokenizer loading only), muon.py (local)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import argparse
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from muon import SingleDeviceMuonWithAuxAdam

IGNORE_INDEX = -100


class SFTDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_seq_length=2048, model_size="1.8B"):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.model_size = model_size
        self.pad_token_id = tokenizer.encode(tokenizer.pad_token)[0]
        with open(data_file, 'r', encoding='utf8') as f:
            self.data_list = f.readlines()
        print(f"Loaded {len(self.data_list)} samples from {data_file}")

        # Precompute special token ids
        if model_size == "7B":
            self.sep_token_id = tokenizer.convert_tokens_to_ids('<|extra_0|>')
            self.eos_token_id = tokenizer.convert_tokens_to_ids('<|eos|>')
        else:
            self.sep_token_id = tokenizer.convert_tokens_to_ids('<｜hy_Assistant｜>')
            self.eos_token_id = tokenizer.convert_tokens_to_ids('<｜hy_place▁holder▁no▁2｜>')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = json.loads(self.data_list[index])
        token_ids = self.tokenizer.apply_chat_template(data['messages'], tokenize=True, return_dict=False)
        if not isinstance(token_ids, list):
            token_ids = token_ids.tolist() if hasattr(token_ids, 'tolist') else list(token_ids)
        tokens = torch.tensor(token_ids, dtype=torch.long)

        # Build labels: only compute loss on assistant responses
        labels = torch.full_like(tokens, IGNORE_INDEX)
        begins = (tokens == self.sep_token_id).nonzero(as_tuple=True)[0].tolist()
        ends = (tokens == self.eos_token_id).nonzero(as_tuple=True)[0].tolist()
        for b, e in zip(begins, ends):
            labels[b:e + 1] = tokens[b:e + 1]

        # Truncate
        tokens = tokens[:self.max_seq_length]
        labels = labels[:self.max_seq_length]
        attention_mask = tokens.ne(self.pad_token_id)

        return dict(input_ids=tokens, labels=labels, attention_mask=attention_mask)


def collate_fn(batch, pad_token_id):
    input_ids = [b['input_ids'] for b in batch]
    labels = [b['labels'] for b in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    attention_mask = input_ids.ne(pad_token_id)
    return dict(input_ids=input_ids, labels=labels, attention_mask=attention_mask)


def build_optimizer(model, muon_lr, adam_lr, muon_momentum, weight_decay):
    muon_params = []
    adam_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and "embed" not in name and "lm_head" not in name:
            muon_params.append(param)
        else:
            adam_params.append(param)

    muon_count = sum(p.numel() for p in muon_params)
    adam_count = sum(p.numel() for p in adam_params)
    print(f"Muon params: {muon_count:,} | Adam params: {adam_count:,}")

    param_groups = [
        dict(params=muon_params, lr=muon_lr, momentum=muon_momentum, weight_decay=weight_decay, use_muon=True),
        dict(params=adam_params, lr=adam_lr, betas=(0.9, 0.95), eps=1e-10, weight_decay=weight_decay, use_muon=False),
    ]
    return SingleDeviceMuonWithAuxAdam(param_groups)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Dataset & DataLoader
    dataset = SFTDataset(args.train_data, tokenizer, args.max_seq_length, args.model_size)
    pad_token_id = tokenizer.encode(tokenizer.pad_token)[0]
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id),
    )

    # Optimizer
    optimizer = build_optimizer(model, args.muon_lr, args.adam_lr, args.muon_momentum, args.weight_decay)

    # LR scheduler (linear warmup + linear decay)
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return max(0.0, 1.0 - progress)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    model.train()
    step = 0
    t0 = time.time()
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)

    while step < args.max_steps:
        for batch in dataloader:
            if step >= args.max_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=not args.fp16):
                outputs = model(**batch)
                loss = outputs.loss

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    if args.fp16:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step += 1
            if step % args.logging_steps == 0:
                elapsed = time.time() - t0
                lrs = [f"{g['lr']:.6f}" for g in optimizer.param_groups]
                print(f"step {step}/{args.max_steps} | loss {loss.item():.4f} | "
                      f"lr [{', '.join(lrs)}] | {elapsed:.1f}s")

            if args.save_steps > 0 and step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{step}")
                os.makedirs(save_path, exist_ok=True)
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"Saved checkpoint to {save_path}")

    # Save final model
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Saved final model to {args.output_dir}")

    elapsed = time.time() - t0
    print(f"Training complete: {step} steps in {elapsed:.1f}s ({step/elapsed:.2f} steps/s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HY-MT fine-tuning with Muon optimizer")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="1.8B", choices=["0.5B", "1.8B", "4B", "7B"])
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--muon_lr", type=float, default=0.001)
    parser.add_argument("--adam_lr", type=float, default=1e-4)
    parser.add_argument("--muon_momentum", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=0, help="Save every N steps. 0 = only save at end.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 instead of bf16")
    args = parser.parse_args()
    train(args)

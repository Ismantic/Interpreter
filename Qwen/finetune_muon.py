"""
Minimal fine-tuning script for HY-MT models with Muon optimizer.
Dependencies: torch, transformers (model/tokenizer loading only), muon.py (local)

Supports:
  - SFT mode: JSONL with {"messages": [...]} (for translation fine-tuning)
  - CLM mode: Plain text files, one sentence per line (for embedding pre-training)
  - --freeze_transformer: Only train embed_tokens + lm_head
  - Auto-detect piece_tokenizer vs HuggingFace tokenizer
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import argparse
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM
from muon import SingleDeviceMuonWithAuxAdam

IGNORE_INDEX = -100


def load_tokenizer(model_path):
    if os.path.exists(os.path.join(model_path, "piece.model")):
        from tokenizer_wrapper import PieceTokenizerWrapper
        return PieceTokenizerWrapper(model_path)
    else:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


class SFTDataset(Dataset):
    """Chat-format dataset for translation SFT."""
    def __init__(self, data_file, tokenizer, max_seq_length=2048):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id
        with open(data_file, 'r', encoding='utf8') as f:
            self.data_list = f.readlines()
        print(f"[SFT] Loaded {len(self.data_list)} samples from {data_file}")

        # For label masking: find assistant and eos token ids
        self.assistant_id = getattr(tokenizer, 'assistant_token_id', None)
        self.eos_id = tokenizer.eos_token_id

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = json.loads(self.data_list[index])
        token_ids = self.tokenizer.apply_chat_template(data['messages'], tokenize=True)
        if not isinstance(token_ids, list):
            token_ids = token_ids.tolist() if hasattr(token_ids, 'tolist') else list(token_ids)
        tokens = torch.tensor(token_ids, dtype=torch.long)

        # Build labels: only compute loss on assistant responses
        labels = torch.full_like(tokens, IGNORE_INDEX)
        if self.assistant_id is not None:
            begins = (tokens == self.assistant_id).nonzero(as_tuple=True)[0].tolist()
            ends = (tokens == self.eos_id).nonzero(as_tuple=True)[0].tolist()
            for b, e in zip(begins, ends):
                labels[b:e + 1] = tokens[b:e + 1]

        tokens = tokens[:self.max_seq_length]
        labels = labels[:self.max_seq_length]
        attention_mask = tokens.ne(self.pad_token_id)
        return dict(input_ids=tokens, labels=labels, attention_mask=attention_mask)


class CLMDataset(torch.utils.data.IterableDataset):
    """Streaming plain text dataset for causal language modeling.
    Reads files line by line without loading everything into memory.
    Shuffles by maintaining a buffer."""
    def __init__(self, data_files, tokenizer, max_seq_length=512, buffer_size=10000):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.buffer_size = buffer_size
        if isinstance(data_files, str):
            data_files = data_files.split(",")
        self.data_files = [f.strip() for f in data_files]
        print(f"[CLM] Streaming from {len(self.data_files)} files")

    def _line_iterator(self):
        """Round-robin read from all files, shuffle via buffer."""
        import random
        # Open all files simultaneously, round-robin read
        handles = [open(f, 'r', encoding='utf8') for f in self.data_files]
        buf = []
        exhausted = [False] * len(handles)

        while not all(exhausted):
            for i, fh in enumerate(handles):
                if exhausted[i]:
                    continue
                line = fh.readline()
                if not line:
                    exhausted[i] = True
                    continue
                line = line.strip()
                if line:
                    buf.append(line)
                if len(buf) >= self.buffer_size:
                    random.shuffle(buf)
                    yield from buf
                    buf.clear()

        for fh in handles:
            fh.close()
        if buf:
            random.shuffle(buf)
            yield from buf

    def __iter__(self):
        """Pack multiple sentences into max_seq_length windows: <s>sent1</s><s>sent2</s>..."""
        buf = []
        for text in self._line_iterator():
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            buf.append(self.bos_id)
            buf.extend(ids)
            buf.append(self.eos_id)

            while len(buf) >= self.max_seq_length:
                chunk = buf[:self.max_seq_length]
                buf = buf[self.max_seq_length:]
                tokens = torch.tensor(chunk, dtype=torch.long)
                yield dict(input_ids=tokens, labels=tokens.clone(), attention_mask=torch.ones_like(tokens, dtype=torch.bool))


class PreTokenizedDataset(Dataset):
    """Pre-tokenized dataset from a .pt file (shape: [N, seq_len], dtype: int32)."""
    def __init__(self, pt_file):
        self.data = torch.load(pt_file, weights_only=True).long()
        print(f"[PreTok] Loaded {self.data.shape[0]} chunks from {pt_file}, seq_len={self.data.shape[1]}")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        tokens = self.data[index]
        return dict(input_ids=tokens, labels=tokens.clone(), attention_mask=torch.ones_like(tokens, dtype=torch.bool))


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

    if muon_params:
        param_groups = [
            dict(params=muon_params, lr=muon_lr, momentum=muon_momentum, weight_decay=weight_decay, use_muon=True),
            dict(params=adam_params, lr=adam_lr, betas=(0.9, 0.95), eps=1e-10, weight_decay=weight_decay, use_muon=False),
        ]
        return SingleDeviceMuonWithAuxAdam(param_groups)
    else:
        # No Muon params (e.g. freeze_transformer), use plain Adam
        return torch.optim.AdamW(adam_params, lr=adam_lr, betas=(0.9, 0.95), eps=1e-10, weight_decay=weight_decay)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer & model
    tokenizer = load_tokenizer(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Freeze transformer if requested
    if args.freeze_transformer:
        for name, param in model.named_parameters():
            if "embed" not in name and "lm_head" not in name:
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Frozen transformer: training {trainable:,} / {total:,} params "
              f"({trainable/total*100:.1f}%)")

    # Freeze specific embedding rows (one-to-one mapped tokens)
    if args.freeze_mapped_embeds:
        with open(args.freeze_mapped_embeds) as f:
            frozen_ids = json.load(f)
        frozen_mask = torch.zeros(model.config.vocab_size, 1, device=device, dtype=torch.bfloat16)
        frozen_mask[frozen_ids] = 1.0
        trainable_count = model.config.vocab_size - len(frozen_ids)
        print(f"Freezing {len(frozen_ids)} / {model.config.vocab_size} embedding rows, "
              f"training {trainable_count} rows ({trainable_count/model.config.vocab_size*100:.1f}%)")

        def _zero_frozen_grads(grad):
            return grad * (1.0 - frozen_mask)
        model.model.embed_tokens.weight.register_hook(_zero_frozen_grads)

    # Dataset
    pad_token_id = tokenizer.pad_token_id
    if args.train_data.endswith(".pt"):
        dataset = PreTokenizedDataset(args.train_data)
    elif args.mode == "clm":
        dataset = CLMDataset(args.train_data, tokenizer, args.max_seq_length)
    else:
        dataset = SFTDataset(args.train_data, tokenizer, args.max_seq_length)

    is_iterable = isinstance(dataset, torch.utils.data.IterableDataset)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=not is_iterable, num_workers=0,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id),
    )

    # Optimizer
    optimizer = build_optimizer(model, args.muon_lr, args.adam_lr, args.muon_momentum, args.weight_decay)

    # LR scheduler
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return max(0.0, 1.0 - progress)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    # step counts optimizer updates, not forward passes
    model.train()
    step = 0
    micro_step = 0
    t0 = time.time()
    interrupted = False

    import signal
    def _sigint_handler(sig, frame):
        nonlocal interrupted
        if interrupted:  # second Ctrl+C = force quit
            raise KeyboardInterrupt
        interrupted = True
        print(f"\nCtrl+C received at step {step}, saving checkpoint...")
    signal.signal(signal.SIGINT, _sigint_handler)

    while step < args.max_steps and not interrupted:
        for batch in dataloader:
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
                    lrs = [f"{g['lr']:.6f}" for g in optimizer.param_groups]
                    real_loss = loss.item() * args.gradient_accumulation_steps
                    print(f"step {step}/{args.max_steps} | loss {real_loss:.4f} | "
                          f"lr [{', '.join(lrs)}] | {elapsed:.1f}s")

                if args.save_steps > 0 and step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{step}")
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    print(f"Saved checkpoint to {save_path}")

    # Save final model
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
        print(f"Saved final model to {args.output_dir}")

    elapsed = time.time() - t0
    print(f"Training complete: {step} steps in {elapsed:.1f}s ({step/elapsed:.2f} steps/s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HY-MT fine-tuning with Muon optimizer")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True, help="JSONL for sft mode, or comma-separated text files for clm mode")
    parser.add_argument("--mode", type=str, default="sft", choices=["sft", "clm"])
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--freeze_transformer", action="store_true", help="Only train embed_tokens + lm_head")
    parser.add_argument("--freeze_mapped_embeds", type=str, default=None,
                        help="Path to JSON list of token IDs to freeze (one-to-one mapped tokens)")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--muon_lr", type=float, default=0.001)
    parser.add_argument("--adam_lr", type=float, default=1e-4)
    parser.add_argument("--muon_momentum", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    train(args)

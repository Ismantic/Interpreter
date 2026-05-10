"""
Compute Bits Per Byte (BPB) for a model on held-out data.
Tokenizer-independent metric for comparing models with different vocabularies.

Usage:
    python eval_bpb.py --model_path ./HY-MT1.5-1.8B --data ./private/phase1_v18_ft.pt --num_chunks 500
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math
import argparse
import torch
from transformers import AutoModelForCausalLM

IGNORE_INDEX = -100


def load_tokenizer(model_path):
    if os.path.exists(os.path.join(model_path, "piece.model")):
        from tokenizer_wrapper import PieceTokenizerWrapper
        return PieceTokenizerWrapper(model_path)
    else:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def compute_token_bytes(tokenizer, vocab_size):
    """Compute byte length of each token in the vocabulary."""
    token_bytes = torch.zeros(vocab_size, dtype=torch.long)
    for i in range(vocab_size):
        try:
            text = tokenizer.decode([i], skip_special_tokens=False)
            token_bytes[i] = len(text.encode('utf-8'))
        except:
            token_bytes[i] = 0
    return token_bytes


@torch.no_grad()
def evaluate_bpb(model, tokenizer, data_path, num_chunks=500, batch_size=16):
    """
    Bits per byte on held-out data.
    Uses the LAST num_chunks chunks from the .pt file as validation.
    """
    device = next(model.parameters()).device

    # Load data - use last N chunks as val
    data = torch.load(data_path, weights_only=True).long()
    total = data.shape[0]
    val_data = data[total - num_chunks:]
    seq_len = val_data.shape[1]
    print(f"Val data: {len(val_data)} chunks, seq_len={seq_len}")

    # Compute token byte lengths
    vocab_size = model.config.vocab_size
    token_bytes = compute_token_bytes(tokenizer, vocab_size).to(device)
    print(f"Token bytes computed for {vocab_size} tokens")

    # Evaluate
    total_nats = 0.0
    total_bytes = 0

    for i in range(0, len(val_data), batch_size):
        batch = val_data[i:i + batch_size].to(device)
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=input_ids).logits

        # Per-token cross-entropy (nats)
        loss_flat = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction='none'
        )

        # Byte lengths of target tokens
        target_flat = targets.reshape(-1)
        nbytes = token_bytes[target_flat]

        # Exclude padding and special tokens (0 bytes)
        mask = nbytes > 0
        total_nats += (loss_flat.float() * mask).sum().item()
        total_bytes += nbytes[mask].sum().item()

        if (i // batch_size + 1) % 10 == 0:
            bpb_so_far = total_nats / (math.log(2) * max(total_bytes, 1))
            print(f"  {i + batch_size}/{len(val_data)} | BPB so far: {bpb_so_far:.6f}")

    bpb = total_nats / (math.log(2) * total_bytes)
    ppl = math.exp(total_nats / (len(val_data) * (seq_len - 1)))
    print(f"\nBPB: {bpb:.6f}")
    print(f"Approx PPL: {ppl:.2f}")
    print(f"Total bytes: {total_bytes:,}")
    return bpb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data", type=str, required=True, help="Pre-tokenized .pt file")
    parser.add_argument("--num_chunks", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).cuda()
    model.eval()

    bpb = evaluate_bpb(model, tokenizer, args.data, args.num_chunks, args.batch_size)


if __name__ == "__main__":
    main()

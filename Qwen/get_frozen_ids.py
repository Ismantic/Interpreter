"""
Find one-to-one mapped token IDs (tokens that map to exactly one old token).
These can be frozen during Phase 1 training.

Usage:
    python get_frozen_ids.py \
        --new_tokenizer ./piece_mt.model \
        --old_model_path ./HY-MT1.5-1.8B \
        --output ./private/frozen_ids.json
"""
import json
import argparse
import piece_tokenizer as pt
from transformers import AutoTokenizer


def main(args):
    old_tok = AutoTokenizer.from_pretrained(args.old_model_path, trust_remote_code=True)
    new_tok = pt.Tokenizer()
    new_tok.load(args.new_tokenizer, cn_dict=args.cn_dict if args.cn_dict else "")

    one_to_one = []
    multi = []
    fail = []

    for i in range(new_tok.vocab_size()):
        try:
            piece = new_tok.id_to_piece(i)
        except UnicodeDecodeError:
            fail.append(i)
            continue

        if piece in ('<unk>', '<s>', '</s>', '<pad>', '<user>', '<assistant>', '<system>'):
            continue

        text = piece.replace('▁', ' ')
        if not text.strip():
            text = ' '
        try:
            old_ids = old_tok.encode(text, add_special_tokens=False)
        except:
            fail.append(i)
            continue

        if len(old_ids) == 1:
            one_to_one.append(i)
        else:
            multi.append(i)

    print(f"One-to-one (freeze): {len(one_to_one)}")
    print(f"Multi-to-one (train): {len(multi)}")
    print(f"Fallback (train): {len(fail)}")

    with open(args.output, 'w') as f:
        json.dump(one_to_one, f)
    print(f"Saved frozen IDs to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_tokenizer", type=str, required=True)
    parser.add_argument("--old_model_path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--cn_dict", type=str, default=None)
    args = parser.parse_args()
    main(args)

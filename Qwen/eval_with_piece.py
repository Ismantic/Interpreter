"""
lm-evaluation-harness wrapper that uses our PieceTokenizerWrapper instead of
HuggingFace AutoTokenizer. Needed because the piece tokenizer .model file is a
custom text format that AutoTokenizer can't load.

Registers as model name 'hf_piece' so it's invokable via:
    python -m lm_eval --model hf_piece --model_args pretrained=PATH ...

CLI usage (one task at a time, matching paper Table 2 shot counts):
    python eval_with_piece.py --model_path ./Qwen3-0.6B-Base-new-tok \
        --task piqa --num_fewshot 5 --output_path ./eval_results/new-tok/piqa
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

from tokenizer_wrapper import PieceTokenizerWrapper


class _TokenizerStub:
    """Minimal HF-tokenizer-like surface that satisfies all attribute reads
    HFLM and stop_sequences_criteria perform. Encoding/decoding flows through
    the real PieceTokenizerWrapper held on this stub."""

    def __init__(self, wrapper: PieceTokenizerWrapper, model_path: str):
        self._w = wrapper
        self.bos_token_id = wrapper.bos_token_id
        self.eos_token_id = wrapper.eos_token_id
        self.pad_token_id = wrapper.pad_token_id
        self.unk_token_id = 0
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.padding_side = "left"
        self.model_max_length = 32768
        self.name_or_path = model_path
        self.vocab_size = wrapper.vocab_size

    def __len__(self):
        return self.vocab_size

    def encode(self, text, add_special_tokens=False, **kwargs):
        ids = self._w.encode(text, add_special_tokens=False)
        if add_special_tokens:
            ids = [self._w.bos_token_id] + ids
        return ids

    def decode(self, ids, skip_special_tokens=True, **kwargs):
        if isinstance(ids, int):
            ids = [ids]
        elif hasattr(ids, "tolist"):
            ids = ids.tolist()
        return self._w.decode(ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(self, batch_ids, skip_special_tokens=True, **kwargs):
        if hasattr(batch_ids, "tolist"):
            batch_ids = batch_ids.tolist()
        return [self.decode(row, skip_special_tokens=skip_special_tokens) for row in batch_ids]


@register_model("hf_piece")
class HFLMPiece(HFLM):
    def _create_tokenizer(self, pretrained, tokenizer, revision="main",
                          trust_remote_code=False, use_fast_tokenizer=True,
                          gguf_file=None, add_bos_token=None, subfolder=""):
        path = pretrained if isinstance(pretrained, str) else self.model.name_or_path
        self._piece = PieceTokenizerWrapper(path)
        self.tokenizer = _TokenizerStub(self._piece, path)

    def tok_encode(self, string, left_truncate_len=None, add_special_tokens=None):
        ids = self._piece.encode(string, add_special_tokens=False)
        if add_special_tokens is True:
            ids = [self._piece.bos_token_id] + ids
        if left_truncate_len:
            ids = ids[-left_truncate_len:]
        return ids

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def tok_batch_encode(self, strings, padding_side="left",
                         left_truncate_len=None, truncation=False):
        pad_id = self._piece.pad_token_id
        all_ids = [self._piece.encode(s, add_special_tokens=False) for s in strings]
        if left_truncate_len:
            all_ids = [ids[-left_truncate_len:] for ids in all_ids]
        max_len = max(len(ids) for ids in all_ids) if all_ids else 0

        input_ids, attn = [], []
        for ids in all_ids:
            n_pad = max_len - len(ids)
            if padding_side == "left":
                input_ids.append([pad_id] * n_pad + ids)
                attn.append([0] * n_pad + [1] * len(ids))
            else:
                input_ids.append(ids + [pad_id] * n_pad)
                attn.append([1] * len(ids) + [0] * n_pad)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long)

    def apply_chat_template(self, chat_history, add_generation_prompt=True, **kwargs):
        return self._piece.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=add_generation_prompt
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--num_fewshot", type=int, default=None)
    parser.add_argument("--batch_size", default="auto")
    parser.add_argument("--output_path", default=None,
                        help="Write results JSON to this path")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of evaluation examples (smoke test)")
    args = parser.parse_args()

    from lm_eval import simple_evaluate

    results = simple_evaluate(
        model="hf_piece",
        model_args=f"pretrained={args.model_path},dtype=bfloat16",
        tasks=[args.task],
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
    )

    out = {"task": args.task, "num_fewshot": args.num_fewshot,
           "results": results["results"]}
    print(json.dumps(out, indent=2, default=str))

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nSaved to {args.output_path}")


if __name__ == "__main__":
    main()

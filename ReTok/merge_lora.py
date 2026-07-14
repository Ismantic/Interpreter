"""Merge a LoRA adapter into the base SFT model (PieceTokenizer variant).

Tokenizer-only fork of ../Qwen/merge_lora_qwen3.py — the merge math
(peft.PeftModel.from_pretrained → merge_and_unload → save_pretrained) is verbatim.
Only difference: tokenizer artifacts copied as piece's 5-file set instead of
AutoTokenizer.save_pretrained.
"""
import os
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import _copy_tokenizer_artifacts  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--adapter_model_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--save_dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.save_dtype]

    print(f"Loading base: {args.base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=dtype)
    print(f"Loading adapter: {args.adapter_model_path}")
    model = PeftModel.from_pretrained(model, args.adapter_model_path)
    print("Merging...")
    model = model.merge_and_unload()
    print(f"Saving to {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)
    model.save_pretrained(args.output_path)

    # Copy the 5 piece tokenizer artifacts from base
    _copy_tokenizer_artifacts(args.base_model_path, args.output_path)
    print("Done.")


if __name__ == "__main__":
    main()

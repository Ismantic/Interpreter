"""Merge LoRA adapter into base SFT model (Qwen3)."""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


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
    model.save_pretrained(args.output_path)

    # Also save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    tokenizer.save_pretrained(args.output_path)
    print("Done.")


if __name__ == "__main__":
    main()

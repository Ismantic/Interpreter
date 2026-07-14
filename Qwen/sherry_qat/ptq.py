"""PTQ a model with our SEQ quantizer (no training), bake, save plain Qwen3.

Usage: python ptq.py <fp_model> <w_bits> <out_dir>
Lets us measure N-bit quality on the SAME eval protocol (eval_multi.py) as the
FP / 2-bit models -- no llama.cpp, no chat-template mismatch.
"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantize import quantize_qwen3, bake_quantized

src, w_bits, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
m = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16)
m, n = quantize_qwen3(m, method="seq", w_bits=w_bits, group_size=128)
bake_quantized(m)
m.config.use_cache = True
m.save_pretrained(out)
AutoTokenizer.from_pretrained(src).save_pretrained(out)
print(f"PTQ seq {w_bits}-bit: {n} layers baked -> {out}", flush=True)

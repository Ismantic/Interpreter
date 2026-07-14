"""Materialize a QAT checkpoint into a plain Qwen3 model for fast vLLM eval.

The fake-quant forward computes, per projection layer, the dequantized weight
NMQuant(W) -- a normal dense bf16 tensor whose values happen to be 3:4-sparse
ternary. Baking that tensor back into a plain nn.Linear gives a standard Qwen3
checkpoint that vLLM CAN load, and `F.linear(x, NMQuant(W))` is bit-identical
to the Arenas eps=0 forward. So the materialized model evaluates exactly the
same as the 1.25-bit fake-quant model, but runs at full vLLM speed.

After training:
    python materialize.py --in ./output_qat_125bit --out ./output_qat_125bit_mat
    python ../eval_vllm.py --model_path ./output_qat_125bit_mat \
        --testset wmt23 --direction both
"""
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from quant import NMQuant
from quantize import ATTN_PROJ, MLP_PROJ


def materialize(in_path, out_path, group_size=128, N=3, M=4):
    model = AutoModelForCausalLM.from_pretrained(in_path, dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(in_path)
    n, zeros, total = 0, 0, 0
    with torch.no_grad():
        for layer in model.model.layers:
            for holder, names in ((layer.self_attn, ATTN_PROJ), (layer.mlp, MLP_PROJ)):
                for name in names:
                    lin = getattr(holder, name)
                    qw = NMQuant.apply(lin.weight.data.float(), "per_group",
                                       group_size, N, M)
                    lin.weight.data = qw.to(lin.weight.dtype)
                    n += 1
                    zeros += (qw == 0).sum().item()
                    total += qw.numel()
    print(f"materialized {n} layers | zero_frac={zeros/total:.4f} "
          f"(expect 0.2500)", flush=True)
    model.config.use_cache = True  # plain Qwen3 now; re-enable for fast inference
    model.save_pretrained(out_path)
    tok.save_pretrained(out_path)
    print(f"saved plain Qwen3 (1.25-bit weights baked in) -> {out_path}\n"
          f"eval fast with: python ../eval_vllm.py --model_path {out_path} "
          f"--testset wmt23 --direction both", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", default="./output_qat_125bit")
    p.add_argument("--out", dest="out_path", default="./output_qat_125bit_mat")
    p.add_argument("--group_size", type=int, default=128)
    materialize(p.parse_args().in_path, p.parse_args().out_path,
                p.parse_args().group_size)

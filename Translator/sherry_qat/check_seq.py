"""Measure the untrained 2-bit SEQ loss (RTN starting point) vs FP."""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantize import quantize_qwen3, quant_stats
from quant import set_arenas_eps

M = "/home/tfbao/Shiyu/Interpreter/Translator/output_1.7b_grpo_full"
KD = "/home/tfbao/Shiyu/Interpreter/Translator/sherry_qat/qat_kd_10pct.jsonl"
dev = "cuda"

tok = AutoTokenizer.from_pretrained(M)
data = [json.loads(l) for l in open(KD)][:32]

def batch_loss(model):
    losses = []
    for d in data:
        m = d["messages"]
        pre = tok.encode(f"<|im_start|>user\n{m[0]['content']}<|im_end|>\n<|im_start|>assistant\n",
                         add_special_tokens=False)
        resp = tok.encode(m[1]["content"], add_special_tokens=False) + [tok.convert_tokens_to_ids("<|im_end|>")]
        ids = torch.tensor([pre + resp], device=dev)
        labels = ids.clone(); labels[0, :len(pre)] = -100
        with torch.no_grad():
            losses.append(model(input_ids=ids, labels=labels).loss.item())
    return sum(losses) / len(losses)

model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to(dev).eval()
print(f"FP loss on KD data: {batch_loss(model):.4f}", flush=True)

model, n = quantize_qwen3(model, method="seq", w_bits=2, group_size=128)
set_arenas_eps(model, 0.0)
model.eval()
st = quant_stats(model)
print(f"SEQ 2-bit: {n} layers, zero_frac={st['zero_frac']:.4f}, "
      f"layer0 unique={st['unique_vals_layer0']}", flush=True)
print(f"untrained 2-bit loss on KD data: {batch_loss(model):.4f}", flush=True)

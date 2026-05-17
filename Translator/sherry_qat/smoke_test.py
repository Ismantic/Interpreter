"""Phase 1 smoke test: quantize Qwen3, check sparsity, forward/backward, gen."""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from quant import Arenas, set_arenas_eps
from quantize import quantize_qwen3, quant_stats

MODEL = "/home/tfbao/Shiyu/Interpreter/Translator/output_1.7b_grpo_full"
dev = "cuda"

print("loading FP model ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)

model, n = quantize_qwen3(model, granularity="per_group", group_size=128, N=3, M=4)
print(f"quantized {n} projection layers", flush=True)
model = model.to(dev)

# --- 1. sparsity / ternary check ---
st = quant_stats(model)
print(f"[sparsity] quant_layers={st['quant_layers']} zero_frac={st['zero_frac']:.4f} "
      f"(expect ~0.2500)  quant_params={st['quant_params']/1e9:.2f}B", flush=True)
# ternary check on one layer
q0 = next(m for m in model.modules() if isinstance(m, Arenas))
qw = q0.quantized_weight()
uniq = qw.unique()
print(f"[ternary] one layer: {qw.numel()} weights, {uniq.numel()} unique values "
      f"(0 + ternary signs per group)", flush=True)

# --- 2. forward + loss ---
prompt = ("<|im_start|>user\nTranslate the following text from English to Chinese.\n"
          "English: The weather is nice today.\nChinese:<|im_end|>\n"
          "<|im_start|>assistant\n今天天气很好。<|im_end|>")
enc = tok(prompt, return_tensors="pt").to(dev)
labels = enc.input_ids.clone()
model.train()
set_arenas_eps(model, 0.5)  # mid-anneal
out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=labels)
print(f"[forward] loss={out.loss.item():.4f} finite={torch.isfinite(out.loss).item()}", flush=True)

# --- 3. backward / STE gradient check ---
out.loss.backward()
g = q0.weight.grad
print(f"[backward] quant-layer grad: finite={torch.isfinite(g).all().item()} "
      f"nonzero={(g != 0).any().item()} norm={g.norm().item():.4f}", flush=True)

# --- 4. generation sanity (pre-QAT: quality WILL be poor, just check it runs) ---
model.eval()
set_arenas_eps(model, 0.0)
gp = ("<|im_start|>user\nTranslate the following text from English to Chinese.\n"
      "English: The weather is nice today.\nChinese:<|im_end|>\n<|im_start|>assistant\n")
genc = tok(gp, return_tensors="pt").to(dev)
with torch.no_grad():
    gen = model.generate(**genc, max_new_tokens=40, do_sample=False)
txt = tok.decode(gen[0][genc.input_ids.shape[1]:], skip_special_tokens=True)
print(f"[generate] pre-QAT 1.25-bit output: {txt!r}", flush=True)
print("SMOKE TEST PASSED" if torch.isfinite(out.loss) else "SMOKE TEST FAILED", flush=True)

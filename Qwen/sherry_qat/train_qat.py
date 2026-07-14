"""Sherry 1.25-bit QAT for the Qwen3 translation model.

Full-parameter QAT: load the FP teacher (output_1.7b_grpo_full), swap the
projection layers for Arenas (3:4-sparse ternary fake-quant), and fine-tune on
the KD dataset (teacher-distilled translations) with masked CE loss. The Arenas
annealing residual (eps: 0 -> 1 -> 0) keeps training stable through the
ternary collapse.

Single-4090 config: bf16, gradient checkpointing, Adafactor (tiny optimizer
state), small batch + accumulation.

Saved checkpoints are ordinary Qwen3 dirs (Arenas master weights == nn.Linear
weights), re-quantize at eval time with quantize_qwen3.

Usage:
    python train_qat.py --max_steps 200          # smoke run
    python train_qat.py --num_epochs 3           # full run
"""
import os, sys, json, math, time, argparse, signal
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, Adafactor

from quantize import quantize_qwen3, quant_stats, bake_quantized
from quant import set_arenas_eps, anneal_eps

IGNORE = -100
T = "/home/tfbao/Shiyu/Interpreter/Qwen"


class KDDataset(Dataset):
    """ChatML translation pairs; loss masked to the assistant turn."""

    def __init__(self, path, tok, max_len=512):
        self.tok = tok
        self.max_len = max_len
        self.im_end = tok.convert_tokens_to_ids("<|im_end|>")
        with open(path) as f:
            self.data = [json.loads(l) for l in f if l.strip()]
        print(f"loaded {len(self.data)} KD pairs from {path}", flush=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        m = self.data[i]["messages"]
        prefix = f"<|im_start|>user\n{m[0]['content']}<|im_end|>\n<|im_start|>assistant\n"
        pre = self.tok.encode(prefix, add_special_tokens=False)
        resp = self.tok.encode(m[1]["content"], add_special_tokens=False) + [self.im_end]
        ids = (pre + resp)[:self.max_len]
        ids = torch.tensor(ids, dtype=torch.long)
        labels = torch.full_like(ids, IGNORE)
        labels[len(pre):] = ids[len(pre):]
        return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id):
    import torch.nn.utils.rnn as rnn
    ids = rnn.pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id)
    labels = rnn.pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=IGNORE)
    return {"input_ids": ids, "labels": labels, "attention_mask": ids.ne(pad_id)}


def kd_loss(student_logits, teacher_logits, labels, temp):
    """Forward-KL logit distillation, computed only on the assistant tokens."""
    mask = labels[:, 1:].ne(IGNORE)               # [B, S-1]
    s = student_logits[:, :-1, :][mask].float()   # [N, vocab]
    t = teacher_logits[:, :-1, :][mask].float()
    log_s = F.log_softmax(s / temp, dim=-1)
    soft_t = F.softmax(t / temp, dim=-1)
    return F.kl_div(log_s, soft_t, reduction="batchmean") * (temp * temp)


def main(a):
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model_path)
    pad_id = tok.pad_token_id

    print("loading + quantizing model ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.model_path, dtype=torch.bfloat16)
    model, n = quantize_qwen3(model, method=a.method, w_bits=a.w_bits,
                              granularity="per_group", group_size=a.group_size, N=3, M=4)
    model = model.to(dev)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    st = quant_stats(model)
    print(f"method={a.method} | quantized {n} layers | zero_frac={st['zero_frac']:.4f} "
          f"| layer0 unique vals={st['unique_vals_layer0']} "
          f"| quant_params={st['quant_params']/1e9:.2f}B", flush=True)

    teacher = None
    if a.kd == "logit":
        print("loading FP teacher for logit-KD ...", flush=True)
        teacher = AutoModelForCausalLM.from_pretrained(
            a.teacher_path or a.model_path, dtype=torch.bfloat16)
        teacher.config.use_cache = False
        teacher = teacher.to(dev).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"logit-KD: alpha={a.kd_alpha} temp={a.kd_temp}", flush=True)

    ds = KDDataset(a.data_path, tok, a.max_len)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=2,
                        collate_fn=lambda b: collate(b, pad_id))

    steps_per_epoch = len(loader) // a.grad_accum
    max_steps = a.max_steps if a.max_steps > 0 else steps_per_epoch * a.num_epochs
    warmup = max(1, int(max_steps * a.warmup_ratio))
    print(f"training: {max_steps} steps ({steps_per_epoch}/epoch), warmup={warmup}", flush=True)

    opt = Adafactor(model.parameters(), lr=a.lr, relative_step=False,
                    scale_parameter=False, warmup_init=False, weight_decay=0.0)

    def lr_at(step):
        if step < warmup:
            return a.lr * step / warmup
        return a.lr  # constant after warmup (Sherry recipe)

    interrupted = {"v": False}
    def _sig(s, f):
        if interrupted["v"]:
            sys.exit(1)
        interrupted["v"] = True
        print(f"\nSIGINT -- will save and stop", flush=True)
    signal.signal(signal.SIGINT, _sig)

    model.train()
    step, micro, t0 = 0, 0, time.time()
    opt.zero_grad()
    done = False
    while step < max_steps and not done:
        for batch in loader:
            batch = {k: v.to(dev) for k, v in batch.items()}
            out = model(**batch)
            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(input_ids=batch["input_ids"],
                                       attention_mask=batch["attention_mask"]).logits
                kd = kd_loss(out.logits, t_logits, batch["labels"], a.kd_temp)
                loss_full = a.kd_alpha * kd + (1.0 - a.kd_alpha) * out.loss
            else:
                loss_full = out.loss
            loss = loss_full / a.grad_accum
            loss.backward()
            micro += 1
            if micro % a.grad_accum == 0:
                # Arenas annealing residual + lr schedule
                set_arenas_eps(model, anneal_eps(step, max_steps))
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
                if step % a.log_steps == 0:
                    el = time.time() - t0
                    print(f"step {step}/{max_steps} | loss {loss_full.item():.4f} "
                          f"| eps {anneal_eps(step, max_steps):.3f} | lr {lr_at(step):.2e} "
                          f"| {el:.0f}s | {step/el:.2f} it/s", flush=True)
                if a.save_steps > 0 and step % a.save_steps == 0:
                    sp = os.path.join(a.output_dir, f"checkpoint-{step}")
                    model.save_pretrained(sp); tok.save_pretrained(sp)
                    print(f"saved {sp}", flush=True)
                if step >= max_steps or interrupted["v"]:
                    done = True
                    break

    # bake the dequantized weights -> plain Qwen3 (directly vLLM-loadable)
    set_arenas_eps(model, 0.0)
    bake_quantized(model)
    model.config.use_cache = True
    os.makedirs(a.output_dir, exist_ok=True)
    model.save_pretrained(a.output_dir); tok.save_pretrained(a.output_dir)
    print(f"done: {step} steps in {time.time()-t0:.0f}s -> {a.output_dir} "
          f"(baked, plain Qwen3 -- eval directly with eval_vllm.py)", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=f"{T}/output_1.7b_grpo_full")
    p.add_argument("--data_path", default=f"{T}/sherry_qat/qat_kd.jsonl")
    p.add_argument("--output_dir", default=f"{T}/sherry_qat/output_qat_2bit")
    p.add_argument("--method", default="seq", choices=["sherry", "seq"],
                   help="sherry=1.25-bit 3:4 ternary, seq=2-bit stretched-elastic")
    p.add_argument("--w_bits", type=int, default=2, help="bit width for seq")
    p.add_argument("--group_size", type=int, default=128)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--kd", default="seq", choices=["seq", "logit"],
                   help="seq=CE on teacher text; logit=KL to teacher logits")
    p.add_argument("--kd_alpha", type=float, default=0.9, help="logit-KD: weight of KL vs CE")
    p.add_argument("--kd_temp", type=float, default=2.0, help="logit-KD softmax temperature")
    p.add_argument("--teacher_path", default="", help="logit-KD teacher (default: model_path)")
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    main(p.parse_args())

"""Evaluate the 1.25-bit QAT model with COMET.

vLLM cannot run the custom Arenas modules, so we load the saved checkpoint as a
standard Qwen3, re-apply quantize_qwen3 (copies weights into Arenas, fake-quant
forward), and translate with HF generate. COMET vs the FP baseline 0.8054/0.8542.

Usage: python eval_qat.py --model_path ./output_qat_125bit [--testsets wmt23]
"""
import os, argparse, time, torch
import sacrebleu
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantize import quantize_qwen3, quant_stats
from quant import set_arenas_eps

PROMPT = {"zh": "Translate the following text from Chinese to English.\nChinese: {s}\nEnglish:",
          "en": "Translate the following text from English to Chinese.\nEnglish: {s}\nChinese:"}
FLORES = "/home/tfbao/Shiyu/Interpreter/Qwen/flores200_dataset/devtest"
ALL_TS = [("wmt23", "zh-en"), ("wmt23", "en-zh"), ("wmt24", "en-zh"),
          ("flores", "zh-en"), ("flores", "en-zh")]
COMET_CKPT = os.path.expanduser("~/.cache/comet/models--Unbabel--wmt22-comet-da/"
    "snapshots/2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt")


def load_testset(ts, direction):
    s, t = direction.split("-")
    if ts == "flores":
        fn = {"zh": "zho_Hans.devtest", "en": "eng_Latn.devtest"}
        src = [l.strip() for l in open(f"{FLORES}/{fn[s]}")]
        ref = [l.strip() for l in open(f"{FLORES}/{fn[t]}")]
        return src, ref
    src = [l.strip() for l in open(sacrebleu.get_source_file(ts, direction))]
    ref = [l.strip() for l in open(sacrebleu.get_reference_files(ts, direction)[0])]
    return src, ref


@torch.no_grad()
def translate(model, tok, srcs, src_lang, bs=32, max_new=256):
    out = []
    for i in range(0, len(srcs), bs):
        chunk = srcs[i:i+bs]
        prompts = [f"<|im_start|>user\n{PROMPT[src_lang].format(s=s)}<|im_end|>\n"
                   f"<|im_start|>assistant\n" for s in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            out.append(tok.decode(gen[j][enc.input_ids.shape[1]:],
                                  skip_special_tokens=True).strip())
    return out


def main(a):
    tok = AutoTokenizer.from_pretrained(a.model_path)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model_path, dtype=torch.bfloat16)
    model, n = quantize_qwen3(model, group_size=a.group_size, N=3, M=4)
    set_arenas_eps(model, 0.0)
    model.config.use_cache = True  # KV cache: Arenas is a plain linear, safe + faster
    model = model.to("cuda").eval()
    st = quant_stats(model)
    print(f"1.25-bit model: {n} quant layers, zero_frac={st['zero_frac']:.4f}", flush=True)

    sel = ALL_TS if a.testsets == "all" else [t for t in ALL_TS if t[0] == a.testsets]
    results = {}
    for ts, direction in sel:
        src_lang = direction.split("-")[0]
        srcs, refs = load_testset(ts, direction)
        if a.max_samples > 0:
            srcs, refs = srcs[:a.max_samples], refs[:a.max_samples]
        t0 = time.time()
        hyps = translate(model, tok, srcs, src_lang)
        results[(ts, direction)] = (srcs, hyps, refs)
        bleu = sacrebleu.corpus_bleu(hyps, [refs],
                                     tokenize="zh" if direction.endswith("zh") else "13a")
        print(f"{ts} {direction}: BLEU {bleu.score:.2f} ({time.time()-t0:.0f}s)", flush=True)

    print("loading COMET ...", flush=True)
    from comet import load_from_checkpoint
    comet = load_from_checkpoint(COMET_CKPT)
    print(f"\n{'='*64}\n1.25-bit QAT eval: {a.model_path}\n{'='*64}")
    print(f"{'testset':<18}{'COMET':>10}{'BLEU':>10}")
    for (ts, direction), (srcs, hyps, refs) in results.items():
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
        cm = comet.predict(data, batch_size=64, gpus=1, progress_bar=False)
        comet_score = sum(cm.scores) / len(cm.scores)
        bleu = sacrebleu.corpus_bleu(hyps, [refs],
                                     tokenize="zh" if direction.endswith("zh") else "13a")
        print(f"{ts+' '+direction:<18}{comet_score:>10.4f}{bleu.score:>10.2f}", flush=True)
    print("FP baseline (output_1.7b_grpo_full): wmt23 zh-en 0.8054 / en-zh 0.8542")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="./output_qat_125bit")
    p.add_argument("--testsets", default="wmt23", help="wmt23 | wmt24 | flores | all")
    p.add_argument("--group_size", type=int, default=128)
    p.add_argument("--max_samples", type=int, default=0)
    main(p.parse_args())

"""Evaluate the deployed 4-bit Q4_K_M model (via the running llama-server) on
WMT23, vs the FP baseline. CPU translation -> runs in parallel with GPU training.

Translates and saves q4_translations.json + reports BLEU immediately.
COMET is scored separately (needs GPU) once training frees it:
    python eval_q4_cpu.py --comet
"""
import json, time, sys, os, random, urllib.request
import sacrebleu

MAX = int(os.environ.get("MAX", "0"))  # 0 = full test set; else N random sentences/dir

EP = os.environ.get("EP", "http://127.0.0.1:8080/completion")  # raw endpoint
OUT = "q4_translations.json"
PROMPT = {"zh": "Translate the following text from Chinese to English.\nChinese: {s}\nEnglish:",
          "en": "Translate the following text from English to Chinese.\nEnglish: {s}\nChinese:"}
TESTS = [("wmt23", "zh-en"), ("wmt23", "en-zh")]
COMET_CKPT = ("/home/tfbao/.cache/comet/models--Unbabel--wmt22-comet-da/snapshots/"
              "2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt")


def translate(text, srclang):
    # bare ChatML prompt -- exactly the model's training format (matches eval_multi.py)
    prompt = (f"<|im_start|>user\n{PROMPT[srclang].format(s=text)}<|im_end|>\n"
              f"<|im_start|>assistant\n")
    body = json.dumps({"prompt": prompt, "n_predict": 256,
                       "temperature": 0, "stop": ["<|im_end|>"]}).encode()
    for attempt in range(3):
        try:
            req = urllib.request.Request(EP, body, {"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=180))
            return r["content"].strip()
        except Exception as e:
            if attempt == 2:
                print(f"  [translate failed after retries: {e}]", flush=True)
                return ""
            time.sleep(3)


def do_translate():
    results = {}
    for ts, direction in TESTS:
        src = [l.strip() for l in open(sacrebleu.get_source_file(ts, direction))]
        ref = [l.strip() for l in open(sacrebleu.get_reference_files(ts, direction)[0])]
        if MAX and MAX < len(src):
            random.seed(42)  # fixed seed -> reproducible random sample
            idx = sorted(random.sample(range(len(src)), MAX))
            src = [src[i] for i in idx]
            ref = [ref[i] for i in idx]
        srclang = direction.split("-")[0]
        hyps, t0 = [], time.time()
        for i, s in enumerate(src):
            hyps.append(translate(s, srclang))
            if (i + 1) % 200 == 0:
                print(f"{ts} {direction}: {i+1}/{len(src)} ({time.time()-t0:.0f}s)", flush=True)
        bleu = sacrebleu.corpus_bleu(hyps, [ref],
                                     tokenize="zh" if direction.endswith("zh") else "13a")
        results[f"{ts}_{direction}"] = {"src": src, "hyp": hyps, "ref": ref,
                                        "bleu": bleu.score}
        json.dump(results, open(OUT, "w"), ensure_ascii=False)
        print(f"=== {ts} {direction}: BLEU {bleu.score:.2f} ===", flush=True)
    print(f"translations saved -> {OUT}", flush=True)


def do_comet():
    from comet import load_from_checkpoint
    results = json.load(open(OUT))
    comet = load_from_checkpoint(COMET_CKPT)
    print(f"\n{'='*52}\n4-bit Q4_K_M vs FP baseline\n{'='*52}")
    print(f"{'testset':<16}{'COMET':>9}{'BLEU':>8}   {'FP COMET':>9}")
    fp = {"wmt23_zh-en": 0.8054, "wmt23_en-zh": 0.8542}
    for key, d in results.items():
        data = [{"src": s, "mt": h, "ref": r}
                for s, h, r in zip(d["src"], d["hyp"], d["ref"])]
        cm = comet.predict(data, batch_size=64, gpus=1, progress_bar=False)
        c = sum(cm.scores) / len(cm.scores)
        print(f"{key:<16}{c:>9.4f}{d['bleu']:>8.2f}   {fp.get(key,0):>9.4f}"
              f"   (Δ {c-fp.get(key,0):+.4f})", flush=True)


if __name__ == "__main__":
    do_comet() if "--comet" in sys.argv else do_translate()

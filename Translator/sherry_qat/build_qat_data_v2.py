"""Stage D: build a much larger KD/QAT dataset from the private/ source pools.

KD needs only SOURCE sentences (the FP teacher generates the targets). We sweep
private/*.txt, auto-detect zh/en per line, sentence-split paragraphs, filter for
length / language purity / junk, dedup, balance the two directions, then
teacher-distill. Failed teacher outputs are dropped. Merged with the existing
43K -> qat_kd_v2.jsonl.
"""
import json, re, glob, random, time
from vllm import LLM, SamplingParams

T = "/home/tfbao/Shiyu/Interpreter/Translator"
PRIV = "/home/tfbao/Shiyu/Interpreter/private"
TEACHER = f"{T}/output_1.7b_grpo_full"
EXISTING = f"{T}/sherry_qat/qat_kd.jsonl"
OUT = f"{T}/sherry_qat/qat_kd_v2.jsonl"
TARGET_PER_DIR = 48000           # ~48K zh-source + ~48K en-source (+43K existing ~= 130K)

CJK = re.compile(r"[一-鿿]")
URL = re.compile(r"https?://|www\.|\.com|\.cn/")
PROMPT = {"zh": "Translate the following text from Chinese to English.\nChinese: {s}\nEnglish:",
          "en": "Translate the following text from English to Chinese.\nEnglish: {s}\nChinese:"}


def is_zh(s):
    return len(CJK.findall(s)) / max(1, len(s)) > 0.20


def sent_split(line, zh):
    parts = re.split(r"(?<=[。！？；])", line) if zh else re.split(r"(?<=[.!?])\s+", line)
    return [p.strip() for p in parts if p.strip()]


def good(s):
    n = len(s)
    if URL.search(s):
        return False
    zh = is_zh(s)
    if zh:
        if not (8 <= n <= 256):
            return False
        if len(CJK.findall(s)) / n < 0.45:
            return False
    else:
        if not (25 <= n <= 480):
            return False
        if sum(c.isascii() and c.isalpha() for c in s) / n < 0.55:
            return False
    if sum(not c.isalnum() and not c.isspace() for c in s) / n > 0.30:
        return False
    return True


def collect():
    seen = set()
    zh, en = [], []
    for path in sorted(glob.glob(f"{PRIV}/*.txt")):
        for raw in open(path, errors="ignore"):
            raw = raw.strip()
            if not raw:
                continue
            z0 = is_zh(raw)
            units = sent_split(raw, z0) if len(raw) > (256 if z0 else 480) else [raw]
            for s in units:
                if not good(s):
                    continue
                key = s[:120]
                if key in seen:
                    continue
                seen.add(key)
                (zh if is_zh(s) else en).append(s)
    random.seed(42)
    random.shuffle(zh)
    random.shuffle(en)
    return zh[:TARGET_PER_DIR], en[:TARGET_PER_DIR]


def teacher_ok(src, tgt):
    if not tgt:
        return False
    r = len(tgt) / max(1, len(src))
    if r < 0.2 or r > 5.0:
        return False
    toks = tgt.split()
    if len(toks) > 12 and len(set(toks)) / len(toks) < 0.35:   # repetition
        return False
    return True


def main():
    zh, en = collect()
    print(f"collected after filter: {len(zh)} zh-source, {len(en)} en-source", flush=True)
    items = [("zh", s) for s in zh] + [("en", s) for s in en]

    llm = LLM(model=TEACHER, dtype="bfloat16", max_model_len=768,
              gpu_memory_utilization=0.85)
    wrapped = [f"<|im_start|>user\n{PROMPT[d].format(s=s)}<|im_end|>\n<|im_start|>assistant\n"
               for d, s in items]
    params = SamplingParams(max_tokens=256, temperature=0.0, stop=["<|im_end|>"])
    t0 = time.time()
    outs = llm.generate(wrapped, params)
    print(f"teacher gen: {len(outs)} in {time.time()-t0:.0f}s", flush=True)

    # merge: existing 43K + new (deduped by prompt), drop failed teacher outputs
    seen_prompts, recs = set(), []
    with open(EXISTING) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs.append(r)
                seen_prompts.add(r["messages"][0]["content"])
    kept = 0
    for (d, s), o in zip(items, outs):
        tgt = o.outputs[0].text.strip()
        if not teacher_ok(s, tgt):
            continue
        prompt = PROMPT[d].format(s=s)
        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)
        recs.append({"messages": [{"role": "user", "content": prompt},
                                  {"role": "assistant", "content": tgt}]})
        kept += 1
    random.shuffle(recs)
    with open(OUT, "w", encoding="utf8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"new kept: {kept} | total qat_kd_v2: {len(recs)} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

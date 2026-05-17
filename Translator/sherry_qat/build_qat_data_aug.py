"""Augment the KD/QAT dataset with alma_sources.jsonl.

The first pass (build_qat_data.py) deduped SFT+CPO+GRPO prompts down to ~36K --
heavy overlap. This adds alma_sources (source + direction), teacher-distills it,
and appends new (deduped) pairs to qat_kd.jsonl for broader source coverage.
"""
import json, time
from vllm import LLM, SamplingParams

T = "/home/tfbao/Shiyu/Interpreter/Translator"
TEACHER = f"{T}/output_1.7b_grpo_full"
KD = f"{T}/sherry_qat/qat_kd.jsonl"

PROMPT = {"zh-en": "Translate the following text from Chinese to English.\nChinese: {s}\nEnglish:",
          "en-zh": "Translate the following text from English to Chinese.\nEnglish: {s}\nChinese:"}


def main():
    existing = set()
    with open(KD) as f:
        for line in f:
            if line.strip():
                existing.add(json.loads(line)["messages"][0]["content"])

    new_prompts = set()
    with open(f"{T}/alma_sources.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            d = r.get("direction")
            if d not in PROMPT:
                continue
            p = PROMPT[d].format(s=r["source"])
            if len(p) < 1200 and p not in existing:
                new_prompts.add(p)
    new_prompts = list(new_prompts)
    print(f"{len(new_prompts)} new prompts to distill (after dedup)", flush=True)
    if not new_prompts:
        return

    llm = LLM(model=TEACHER, dtype="bfloat16", max_model_len=768,
              gpu_memory_utilization=0.85)
    wrapped = [f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n" for p in new_prompts]
    params = SamplingParams(max_tokens=256, temperature=0.0, stop=["<|im_end|>"])
    t0 = time.time()
    outs = llm.generate(wrapped, params)
    print(f"teacher gen: {len(outs)} in {time.time()-t0:.0f}s", flush=True)

    n = 0
    with open(KD, "a", encoding="utf8") as f:
        for p, o in zip(new_prompts, outs):
            tgt = o.outputs[0].text.strip()
            if not tgt:
                continue
            f.write(json.dumps({"messages": [
                {"role": "user", "content": p},
                {"role": "assistant", "content": tgt}]}, ensure_ascii=False) + "\n")
            n += 1
    print(f"appended {n} pairs -> {KD}", flush=True)


if __name__ == "__main__":
    main()

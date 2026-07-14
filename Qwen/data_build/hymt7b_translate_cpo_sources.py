"""Translate cpo_preference.jsonl sources with HY-MT 7B for augmenting candidate pool."""
import json
import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

PROMPT_ZH2EN = "Translate the following text from Chinese to English.\nChinese: {src}\nEnglish:"
PROMPT_EN2ZH = "Translate the following text from English to Chinese.\nEnglish: {src}\nChinese:"


def extract_src(prompt, direction):
    if direction == "zh-en":
        return prompt.split("Chinese: ")[1].split("\nEnglish:")[0].strip()
    return prompt.split("English: ")[1].split("\nChinese:")[0].strip()


def main():
    # Read unique sources from cpo_preference
    sources_by_dir = {"zh-en": [], "en-zh": []}
    seen = {"zh-en": set(), "en-zh": set()}
    with open("data/cpo_preference.jsonl") as f:
        for line in f:
            d = json.loads(line)
            direction = d["direction"]
            src = extract_src(d["prompt"], direction)
            if src not in seen[direction]:
                seen[direction].add(src)
                sources_by_dir[direction].append(src)
    print(f"Unique sources: zh-en={len(sources_by_dir['zh-en'])}, en-zh={len(sources_by_dir['en-zh'])}")

    model_path = "./models/HY-MT1.5-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = LLM(model=model_path, dtype="bfloat16",
                trust_remote_code=True, max_model_len=2048,
                gpu_memory_utilization=0.85)

    results = {}
    for direction in ["zh-en", "en-zh"]:
        src_lang = direction.split("-")[0]
        tmpl = PROMPT_ZH2EN if src_lang == "zh" else PROMPT_EN2ZH
        srcs = sources_by_dir[direction]
        print(f"\nTranslating {direction}: {len(srcs)} sentences")
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": tmpl.format(src=s)}],
                tokenize=False, add_generation_prompt=True
            )
            for s in srcs
        ]
        params = SamplingParams(max_tokens=256, temperature=0, stop_token_ids=[127960, 127967])
        t0 = time.time()
        outputs = model.generate(prompts, params)
        trans = [o.outputs[0].text.strip() for o in outputs]
        print(f"Done in {time.time()-t0:.1f}s")
        results[direction] = {"sources": srcs, "translations": trans}

    with open("data/hymt7b_cpov3_translations.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nSaved to hymt7b_cpov3_translations.json")


if __name__ == "__main__":
    main()

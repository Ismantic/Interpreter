"""Translate ALMA-R sources with HY-MT 7B for use in CPO candidate pool."""
import json
import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

PROMPT_ZH2EN = "Translate the following text from Chinese to English.\nChinese: {src}\nEnglish:"
PROMPT_EN2ZH = "Translate the following text from English to Chinese.\nEnglish: {src}\nChinese:"


def main():
    model_path = "./models/HY-MT1.5-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = LLM(model=model_path, dtype="bfloat16",
                trust_remote_code=True, max_model_len=2048,
                gpu_memory_utilization=0.85)

    with open("alma_r_sources.json") as f:
        sources = json.load(f)

    results = {}
    for direction in ["zh-en", "en-zh"]:
        src_lang = direction.split("-")[0]
        prompt_template = PROMPT_ZH2EN if src_lang == "zh" else PROMPT_EN2ZH
        srcs = sources[direction]
        print(f"\nTranslating {direction}: {len(srcs)} sentences")

        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_template.format(src=s)}],
                tokenize=False, add_generation_prompt=True
            )
            for s in srcs
        ]
        params = SamplingParams(max_tokens=256, temperature=0, stop_token_ids=[127960, 127967])
        t0 = time.time()
        outputs = model.generate(prompts, params)
        translations = [o.outputs[0].text.strip() for o in outputs]
        print(f"Done in {time.time()-t0:.1f}s")
        results[direction] = {"sources": srcs, "translations": translations}
        for k in range(2):
            print(f"  src: {srcs[k][:80]}")
            print(f"  trans: {translations[k][:80]}")

    with open("hymt7b_almar_translations.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to hymt7b_almar_translations.json")


if __name__ == "__main__":
    main()

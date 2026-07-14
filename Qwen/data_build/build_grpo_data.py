"""Build GRPO dataset: translation prompts + source + reference for COMET reward."""
import json
import random
import sacrebleu

random.seed(42)

PROMPT_ZH2EN = "Translate the following text from Chinese to English.\nChinese: {src}\nEnglish:"
PROMPT_EN2ZH = "Translate the following text from English to Chinese.\nEnglish: {src}\nChinese:"


def main():
    rows = []
    for year in ["wmt17", "wmt18", "wmt19", "wmt20", "wmt21"]:
        for direction in ["zh-en", "en-zh"]:
            try:
                src_file = sacrebleu.get_source_file(year, direction)
                ref_files = sacrebleu.get_reference_files(year, direction)
                with open(src_file) as f:
                    sources = [l.strip() for l in f]
                with open(ref_files[0]) as f:
                    refs = [l.strip() for l in f]
                for s, r in zip(sources, refs):
                    if not s or not r:
                        continue
                    if len(s) > 400:  # skip very long
                        continue
                    tmpl = PROMPT_ZH2EN if direction == "zh-en" else PROMPT_EN2ZH
                    rows.append({
                        "prompt": [{"role": "user", "content": tmpl.format(src=s)}],
                        "source": s,
                        "reference": r,
                        "direction": direction,
                    })
            except Exception as e:
                print(f"skip {year} {direction}: {e}")

    # Dedup by source
    seen = set()
    unique = []
    for row in rows:
        key = (row["direction"], row["source"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)

    random.shuffle(unique)
    print(f"Total unique GRPO prompts: {len(unique)}")
    zh_en = sum(1 for r in unique if r["direction"] == "zh-en")
    print(f"  zh-en: {zh_en}, en-zh: {len(unique)-zh_en}")

    with open("data/grpo_data.jsonl", "w") as f:
        for row in unique:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("Saved to grpo_data.jsonl")


if __name__ == "__main__":
    main()

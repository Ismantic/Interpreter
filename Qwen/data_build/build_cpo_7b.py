"""Build CPO data: HY-MT 7B chosen vs our greedy rejected.

Approach: always use HY-MT as chosen, our greedy as rejected.
Rationale: HY-MT 7B beats our model by ~0.012 COMET on WMT23, so as chosen
it should work like Exp B (GPT-4 chosen). No COMET scoring needed -
HY-MT is statistically better on average.
"""
import json


def main():
    with open("data/hymt7b_almar_translations.json") as f:
        hymt = json.load(f)

    # Load our greedy translations from cpo_gpt4_vs_ours.jsonl
    our_greedy = {"zh-en": {}, "en-zh": {}}
    with open("data/cpo_gpt4_vs_ours.jsonl") as f:
        for line in f:
            d = json.loads(line)
            prompt = d["prompt"]
            direction = d["direction"]
            if direction == "zh-en":
                src = prompt.split("Chinese: ")[1].split("\nEnglish:")[0].strip()
            else:
                src = prompt.split("English: ")[1].split("\nChinese:")[0].strip()
            our_greedy[direction][src] = d["rejected"]

    pairs = []
    for direction in ["zh-en", "en-zh"]:
        srcs = hymt[direction]["sources"]
        hymt_trans = hymt[direction]["translations"]

        matched = 0
        skipped_identical = 0
        for i, s in enumerate(srcs):
            ours = our_greedy[direction].get(s)
            if ours is None:
                continue
            hymt_t = hymt_trans[i].strip()
            if not hymt_t:
                continue
            if hymt_t == ours.strip():
                skipped_identical += 1
                continue
            matched += 1
            if direction == "zh-en":
                prompt = f"Translate the following text from Chinese to English.\nChinese: {s}\nEnglish:"
            else:
                prompt = f"Translate the following text from English to Chinese.\nEnglish: {s}\nChinese:"
            pairs.append({
                "prompt": prompt,
                "chosen": hymt_t,
                "rejected": ours,
                "direction": direction
            })
        print(f"{direction}: {matched} matched, {skipped_identical} identical (skipped)")

    print(f"\nTotal: {len(pairs)} preference pairs")
    with open("data/cpo_hymt7b_vs_ours.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("Saved to cpo_hymt7b_vs_ours.jsonl")


if __name__ == "__main__":
    main()

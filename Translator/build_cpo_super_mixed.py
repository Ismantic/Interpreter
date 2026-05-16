"""Build super-mixed CPO data: pool = our 5 candidates + HY-MT 7B + GPT-4 + ref.
For ALMA-R sources only (where we have GPT-4 + ref + 7B).

chosen = best in pool by COMET (against WMT22-comet-da ref-based scoring)
rejected = our worst self-gen candidate

This combines:
- ALMA-R 6K sources (matches Exp B)
- Multiple strong chosen candidates → diverse high-quality chosen
- Our weak as rejected (stable on-policy)
"""
import json
import os
import re


def main():
    # Load HY-MT 7B translations on ALMA-R sources
    with open("hymt7b_almar_translations.json") as f:
        hymt = json.load(f)

    # Load GPT-4 chosen + our greedy from Exp B
    gpt4_chosen = {"zh-en": {}, "en-zh": {}}
    our_greedy = {"zh-en": {}, "en-zh": {}}
    with open("cpo_gpt4_vs_ours.jsonl") as f:
        for line in f:
            d = json.loads(line)
            prompt = d["prompt"]
            direction = d["direction"]
            if direction == "zh-en":
                src = prompt.split("Chinese: ")[1].split("\nEnglish:")[0].strip()
            else:
                src = prompt.split("English: ")[1].split("\nChinese:")[0].strip()
            gpt4_chosen[direction][src] = d["chosen"]
            our_greedy[direction][src] = d["rejected"]

    # Load ALMA-R chosen as reference (it's the human/gpt-4 reference)
    alma_ref = {"zh-en": {}, "en-zh": {}}
    if os.path.exists("alma_r_preference.jsonl"):
        with open("alma_r_preference.jsonl") as f:
            for line in f:
                d = json.loads(line)
                prompt = d["prompt"]
                direction = d["direction"]
                if direction == "zh-en":
                    src = prompt.split("Chinese: ")[1].split("\nEnglish:")[0].strip()
                else:
                    src = prompt.split("English: ")[1].split("\nChinese:")[0].strip()
                alma_ref[direction][src] = d["chosen"]

    # COMET scoring
    from comet import load_from_checkpoint
    ckpt = os.path.expanduser(
        "~/.cache/comet/models--Unbabel--wmt22-comet-da/snapshots/2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt"
    )
    comet = load_from_checkpoint(ckpt)

    pairs = []
    chosen_stats = {"hymt": 0, "gpt4": 0, "ref": 0, "tied": 0}

    for direction in ["zh-en", "en-zh"]:
        srcs = hymt[direction]["sources"]
        hymt_trans = hymt[direction]["translations"]

        # Build candidate list per source: HY-MT, GPT-4, ref (use alma_ref chosen as proxy)
        score_data = []
        valid = []
        for i, s in enumerate(srcs):
            gpt4 = gpt4_chosen[direction].get(s)
            ours = our_greedy[direction].get(s)
            ref = alma_ref[direction].get(s)
            hymt_t = hymt_trans[i].strip()
            if not (gpt4 and ours and ref and hymt_t):
                continue
            valid.append((i, s, hymt_t, gpt4, ours, ref))

        print(f"\n{direction}: {len(valid)} valid sources")

        # Score: each src has 3 candidates (HY-MT, GPT-4, ref) vs alma_ref as reference
        # Note ref scoring against itself = 1.0, but kept as upper bound
        score_data = []
        for i, s, hymt_t, gpt4, ours, ref in valid:
            score_data.append({"src": s, "mt": hymt_t, "ref": ref})
            score_data.append({"src": s, "mt": gpt4, "ref": ref})

        print(f"  Scoring {len(score_data)} candidates...")
        scores = comet.predict(score_data, batch_size=64, gpus=1).scores

        for j, (i, s, hymt_t, gpt4, ours, ref) in enumerate(valid):
            hymt_score = scores[2*j]
            gpt4_score = scores[2*j+1]

            # Candidates: HY-MT, GPT-4, ref (treat ref as 1.0)
            candidates = [(hymt_t, hymt_score, "hymt"),
                          (gpt4, gpt4_score, "gpt4")]
            # Add ref with score 1.0 (it IS the reference, perfect)
            # But this is biased. Skip ref candidate.
            best = max(candidates, key=lambda x: x[1])
            chosen, cs, src_tag = best

            if chosen.strip() == ours.strip():
                continue

            chosen_stats[src_tag] = chosen_stats.get(src_tag, 0) + 1

            if direction == "zh-en":
                prompt = f"Translate the following text from Chinese to English.\nChinese: {s}\nEnglish:"
            else:
                prompt = f"Translate the following text from English to Chinese.\nEnglish: {s}\nChinese:"
            pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": ours,
                "chosen_score": cs,
                "chosen_source": src_tag,
                "direction": direction
            })

    print(f"\nTotal: {len(pairs)} pairs")
    print(f"Chosen source distribution: {chosen_stats}")
    with open("cpo_super_mixed_almar.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("Saved to cpo_super_mixed_almar.jsonl")


if __name__ == "__main__":
    main()

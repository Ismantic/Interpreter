"""Build CPO data with mixed chosen sources: HY-MT 7B + GPT-4 + ref.

For each ALMA-R source:
- Candidates: HY-MT 7B translation, GPT-4 (existing chosen in Exp B data)
- COMET-score both against the alma-r reference
- chosen = best by COMET, rejected = our greedy
"""
import json
import os


def main():
    # Load HY-MT 7B translations
    with open("hymt7b_almar_translations.json") as f:
        hymt = json.load(f)

    # Load Exp B data: GPT-4 chosen + our greedy rejected
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

    # Load ALMA-R original preference for additional refs (ref-based COMET scoring)
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
                # use chosen as reference for COMET (ALMA-R chosen is gpt4-mqm or ref)
                alma_ref[direction][src] = d["chosen"]
    print(f"alma_r refs: zh-en={len(alma_ref['zh-en'])}, en-zh={len(alma_ref['en-zh'])}")

    # COMET scoring
    from comet import load_from_checkpoint
    ckpt = os.path.expanduser(
        "~/.cache/comet/models--Unbabel--wmt22-comet-da/snapshots/2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt"
    )
    comet = load_from_checkpoint(ckpt)

    pairs = []
    sources_chosen_stats = {"hymt": 0, "gpt4": 0, "tied": 0}

    for direction in ["zh-en", "en-zh"]:
        srcs = hymt[direction]["sources"]
        hymt_trans = hymt[direction]["translations"]
        # Build scoring data: each src has hymt and gpt4 candidates, score against alma_ref
        cand_data = []
        valid = []
        for i, s in enumerate(srcs):
            ref = alma_ref[direction].get(s)
            gpt4 = gpt4_chosen[direction].get(s)
            ours = our_greedy[direction].get(s)
            hymt_t = hymt_trans[i].strip()
            if not (ref and gpt4 and ours and hymt_t):
                continue
            valid.append(i)
            cand_data.append({"src": s, "mt": hymt_t, "ref": ref})  # hymt score
            cand_data.append({"src": s, "mt": gpt4, "ref": ref})    # gpt4 score
        print(f"\n{direction}: scoring {len(cand_data)} candidates...")
        scores = comet.predict(cand_data, batch_size=64, gpus=1).scores

        for j, i in enumerate(valid):
            s = srcs[i]
            hymt_t = hymt_trans[i].strip()
            gpt4 = gpt4_chosen[direction][s]
            ours = our_greedy[direction][s]
            hymt_score = scores[2*j]
            gpt4_score = scores[2*j+1]

            if hymt_score > gpt4_score:
                chosen, cs, src_tag = hymt_t, hymt_score, "hymt"
                sources_chosen_stats["hymt"] += 1
            elif gpt4_score > hymt_score:
                chosen, cs, src_tag = gpt4, gpt4_score, "gpt4"
                sources_chosen_stats["gpt4"] += 1
            else:
                chosen, cs, src_tag = hymt_t, hymt_score, "tied"
                sources_chosen_stats["tied"] += 1

            if chosen.strip() == ours.strip():
                continue

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

    print(f"\nTotal: {len(pairs)} preference pairs")
    print(f"Chosen sources: {sources_chosen_stats}")
    with open("cpo_mixed_chosen.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("Saved to cpo_mixed_chosen.jsonl")


if __name__ == "__main__":
    main()

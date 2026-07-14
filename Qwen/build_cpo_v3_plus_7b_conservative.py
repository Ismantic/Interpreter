"""Conservative version: only replace chosen with HY-MT 7B when it beats by margin.

The aggressive version (any improvement) caused unstable training.
This version only replaces when 7B COMET score exceeds our chosen by >0.03,
filtering out marginal/noisy replacements that may have style mismatch.
"""
import json
import os
import sacrebleu


def extract_src(prompt, direction):
    if direction == "zh-en":
        return prompt.split("Chinese: ")[1].split("\nEnglish:")[0].strip()
    return prompt.split("English: ")[1].split("\nChinese:")[0].strip()


def load_wmt_refs():
    src_to_ref = {"zh-en": {}, "en-zh": {}}
    for year in ["wmt17", "wmt18", "wmt19", "wmt20", "wmt21", "wmt22"]:
        for direction in ["zh-en", "en-zh"]:
            try:
                src_file = sacrebleu.get_source_file(year, direction)
                ref_files = sacrebleu.get_reference_files(year, direction)
                with open(src_file) as f:
                    sources = [line.strip() for line in f]
                with open(ref_files[0]) as f:
                    refs = [line.strip() for line in f]
                for s, r in zip(sources, refs):
                    src_to_ref[direction][s] = r
            except Exception:
                pass
    return src_to_ref


def main():
    MARGIN = 0.03

    with open("hymt7b_cpov3_translations.json") as f:
        hymt = json.load(f)
    hymt_dict = {}
    for direction in ["zh-en", "en-zh"]:
        hymt_dict[direction] = dict(zip(hymt[direction]["sources"], hymt[direction]["translations"]))

    src_to_ref = load_wmt_refs()

    pairs = []
    with open("cpo_preference.jsonl") as f:
        for line in f:
            pairs.append(json.loads(line))

    augmentable = []
    for p in pairs:
        direction = p["direction"]
        src = extract_src(p["prompt"], direction)
        if src in hymt_dict[direction] and src in src_to_ref[direction]:
            augmentable.append((p, src, direction))
    print(f"Augmentable: {len(augmentable)}")

    from comet import load_from_checkpoint
    ckpt = os.path.expanduser(
        "~/.cache/comet/models--Unbabel--wmt22-comet-da/snapshots/2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt"
    )
    comet = load_from_checkpoint(ckpt)

    score_data = []
    for p, src, direction in augmentable:
        ref = src_to_ref[direction][src]
        score_data.append({"src": src, "mt": p["chosen"], "ref": ref})
        score_data.append({"src": src, "mt": hymt_dict[direction][src], "ref": ref})

    print(f"Scoring {len(score_data)}...")
    scores = comet.predict(score_data, batch_size=64, gpus=1).scores

    new_pairs = []
    replaced = 0
    for i, (p, src, direction) in enumerate(augmentable):
        our_score = scores[2*i]
        hymt_score = scores[2*i+1]
        hymt_t = hymt_dict[direction][src]
        if hymt_score - our_score > MARGIN and hymt_t != p["rejected"]:
            new_pairs.append({
                "prompt": p["prompt"], "chosen": hymt_t, "rejected": p["rejected"],
                "chosen_score": hymt_score, "rejected_score": p["rejected_score"],
                "direction": direction
            })
            replaced += 1
        else:
            new_pairs.append(p)

    print(f"Replaced (margin>{MARGIN}): {replaced} / {len(augmentable)}")
    print(f"Total: {len(new_pairs)}")
    with open("cpo_v3_plus_7b_conservative.jsonl", "w") as f:
        for p in new_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("Saved to cpo_v3_plus_7b_conservative.jsonl")


if __name__ == "__main__":
    main()

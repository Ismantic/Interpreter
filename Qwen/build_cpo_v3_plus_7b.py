"""Augment CPO v3 candidate pool with HY-MT 7B translations.

Approach:
- Original cpo_preference.jsonl has chosen/rejected from our 5 self-gen candidates (COMET scored).
- We add HY-MT 7B translation as a 6th candidate.
- COMET-score the 7B translation against the same ref.
- If 7B > current chosen: replace chosen
- Keep rejected as-is (still our worst)

Need: WMT references for each source. Since cpo_preference comes from WMT17-21,
we'd need to look those up. As proxy: use the chosen translation (which is our best)
as reference for HY-MT scoring - this is biased but the original CPO v3 chosen
already passed quality filtering.

Better: score 7B against the ORIGINAL WMT reference. Get refs from sacrebleu.
"""
import json
import os
import sacrebleu


def extract_src(prompt, direction):
    if direction == "zh-en":
        return prompt.split("Chinese: ")[1].split("\nEnglish:")[0].strip()
    return prompt.split("English: ")[1].split("\nChinese:")[0].strip()


def load_wmt_refs():
    """Load all WMT17-22 zh-en/en-zh sources and references into a dict."""
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
            except Exception as e:
                print(f"  Skip {year} {direction}: {e}")
    print(f"Loaded refs: zh-en={len(src_to_ref['zh-en'])}, en-zh={len(src_to_ref['en-zh'])}")
    return src_to_ref


def main():
    # Load 7B translations
    with open("hymt7b_cpov3_translations.json") as f:
        hymt = json.load(f)
    hymt_dict = {}
    for direction in ["zh-en", "en-zh"]:
        hymt_dict[direction] = dict(zip(hymt[direction]["sources"], hymt[direction]["translations"]))

    # Load WMT refs
    src_to_ref = load_wmt_refs()

    # Load original CPO v3 pairs
    pairs = []
    with open("cpo_preference.jsonl") as f:
        for line in f:
            pairs.append(json.loads(line))
    print(f"Loaded {len(pairs)} CPO v3 pairs")

    # Find which sources have HY-MT translation AND WMT ref
    augmentable = []
    for p in pairs:
        direction = p["direction"]
        src = extract_src(p["prompt"], direction)
        if src in hymt_dict[direction] and src in src_to_ref[direction]:
            augmentable.append((p, src, direction))
    print(f"Augmentable pairs (have 7B + WMT ref): {len(augmentable)}")

    # COMET score HY-MT vs WMT ref
    from comet import load_from_checkpoint
    ckpt = os.path.expanduser(
        "~/.cache/comet/models--Unbabel--wmt22-comet-da/snapshots/2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt"
    )
    comet = load_from_checkpoint(ckpt)

    # Score chosen (our best) and HY-MT against ref
    score_data = []
    for p, src, direction in augmentable:
        ref = src_to_ref[direction][src]
        score_data.append({"src": src, "mt": p["chosen"], "ref": ref})   # our chosen
        score_data.append({"src": src, "mt": hymt_dict[direction][src], "ref": ref})  # hymt

    print(f"\nScoring {len(score_data)} translations...")
    scores = comet.predict(score_data, batch_size=64, gpus=1).scores

    new_pairs = []
    replaced = 0
    kept = 0
    for i, (p, src, direction) in enumerate(augmentable):
        our_chosen_score = scores[2*i]
        hymt_score = scores[2*i+1]
        hymt_t = hymt_dict[direction][src]
        if hymt_score > our_chosen_score and hymt_t != p["rejected"]:
            # Replace chosen with HY-MT
            new_pairs.append({
                "prompt": p["prompt"],
                "chosen": hymt_t,
                "rejected": p["rejected"],
                "chosen_score": hymt_score,
                "rejected_score": p["rejected_score"],
                "direction": direction
            })
            replaced += 1
        else:
            new_pairs.append(p)
            kept += 1

    # Add non-augmentable pairs as-is
    augmentable_keys = set((id(p), src, direction) for p, src, direction in augmentable)
    non_augmentable = 0
    for p in pairs:
        direction = p["direction"]
        src = extract_src(p["prompt"], direction)
        if (id(p), src, direction) not in augmentable_keys:
            new_pairs.append(p)
            non_augmentable += 1

    print(f"\nResults:")
    print(f"  Replaced with 7B chosen: {replaced}")
    print(f"  Kept original chosen: {kept}")
    print(f"  Non-augmentable (kept as-is): {non_augmentable}")
    print(f"  Total: {len(new_pairs)}")

    with open("cpo_v3_plus_7b.jsonl", "w") as f:
        for p in new_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("Saved to cpo_v3_plus_7b.jsonl")


if __name__ == "__main__":
    main()

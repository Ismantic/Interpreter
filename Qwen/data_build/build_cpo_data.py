"""
Score candidates with COMET and build CPO preference pairs.

Usage:
    python build_cpo_data.py --input ./cpo_candidates.jsonl --output ./cpo_preference.jsonl
"""
import os
import sys
import json
import argparse
import numpy as np


def main(args):
    # Load candidates
    examples = []
    with open(args.input, 'r', encoding='utf8') as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"Loaded {len(examples)} examples")

    # Prepare COMET scoring data
    comet_data = []
    example_indices = []
    for i, ex in enumerate(examples):
        for j, candidate in enumerate(ex['candidates']):
            comet_data.append({
                "src": ex['source'],
                "mt": candidate,
                "ref": ex['reference'],
            })
            example_indices.append((i, j))

    print(f"Scoring {len(comet_data)} candidates with COMET...")

    # Load COMET
    from comet import load_from_checkpoint
    comet_path = None
    for p in [
        os.path.expanduser("~/.cache/comet/models--Unbabel--wmt22-comet-da/snapshots/2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt"),
    ]:
        if os.path.exists(p):
            comet_path = p
            break
    model = load_from_checkpoint(comet_path)
    print("COMET loaded")

    output = model.predict(comet_data, batch_size=args.batch_size, gpus=1)
    scores = output.scores

    # Group scores by example
    example_scores = {}
    for (i, j), score in zip(example_indices, scores):
        if i not in example_scores:
            example_scores[i] = []
        example_scores[i].append((j, score))

    # Build preference pairs: best vs worst candidate
    preference_data = []
    for i, ex in enumerate(examples):
        if i not in example_scores:
            continue
        scored = sorted(example_scores[i], key=lambda x: x[1], reverse=True)
        best_idx, best_score = scored[0]
        worst_idx, worst_score = scored[-1]

        # Skip if difference too small
        if best_score - worst_score < 0.01:
            continue

        direction = ex['direction']
        src = ex['source']
        if direction == "zh-en":
            prompt = f"Translate the following text from Chinese to English.\nChinese: {src}\nEnglish:"
        else:
            prompt = f"Translate the following text from English to Chinese.\nEnglish: {src}\nChinese:"

        preference_data.append({
            "prompt": prompt,
            "chosen": ex['candidates'][best_idx],
            "rejected": ex['candidates'][worst_idx],
            "chosen_score": best_score,
            "rejected_score": worst_score,
            "direction": direction,
        })

    # Also add reference as chosen when it beats all candidates
    for i, ex in enumerate(examples):
        if i not in example_scores:
            continue
        scored = sorted(example_scores[i], key=lambda x: x[1])
        worst_idx, worst_score = scored[0]

        direction = ex['direction']
        src = ex['source']
        ref = ex['reference']
        if direction == "zh-en":
            prompt = f"Translate the following text from Chinese to English.\nChinese: {src}\nEnglish:"
        else:
            prompt = f"Translate the following text from English to Chinese.\nEnglish: {src}\nChinese:"

        # Use reference as chosen, worst candidate as rejected
        preference_data.append({
            "prompt": prompt,
            "chosen": ref,
            "rejected": ex['candidates'][worst_idx],
            "chosen_score": 1.0,  # reference assumed perfect
            "rejected_score": worst_score,
            "direction": direction,
        })

    import random
    random.seed(42)
    random.shuffle(preference_data)

    with open(args.output, 'w', encoding='utf8') as f:
        for p in preference_data:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    print(f"\nBuilt {len(preference_data)} preference pairs")
    zh2en = sum(1 for p in preference_data if p['direction'] == 'zh-en')
    en2zh = len(preference_data) - zh2en
    print(f"  zh→en: {zh2en}, en→zh: {en2zh}")

    # Score distribution
    chosen_scores = [p['chosen_score'] for p in preference_data if p['chosen_score'] < 1.0]
    rejected_scores = [p['rejected_score'] for p in preference_data]
    if chosen_scores:
        print(f"  Chosen scores: mean={np.mean(chosen_scores):.4f}")
    print(f"  Rejected scores: mean={np.mean(rejected_scores):.4f}")
    print(f"  Avg gap: {np.mean([p['chosen_score']-p['rejected_score'] for p in preference_data]):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./data/cpo_candidates.jsonl")
    parser.add_argument("--output", type=str, default="./data/cpo_preference.jsonl")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    main(args)

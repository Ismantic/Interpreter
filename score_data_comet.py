"""
Score translation pairs with COMET for data filtering.
Reads a pre-tokenized .pt file, decodes pairs, scores with COMET,
and outputs a filtered .pt file with only high-scoring pairs.

Usage:
    python score_data_comet.py \
        --data ./private/phase1_v18_ft.pt \
        --tokenizer_path ./HY-MT1.5-1.8B-new-tok \
        --output ./private/phase1_comet_filtered.pt \
        --threshold 0.80 \
        --batch_size 64
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import torch
import numpy as np


def main(args):
    from tokenizer_wrapper import PieceTokenizerWrapper
    tok = PieceTokenizerWrapper(args.tokenizer_path)

    # Load data
    data = torch.load(args.data, weights_only=True).long()
    print(f"Loaded {data.shape[0]} chunks, seq_len={data.shape[1]}")

    assistant_id = tok.assistant_token_id
    eos_id = tok.eos_token_id
    user_id = 65004

    # Decode each chunk into source/translation pairs
    pairs = []  # (chunk_idx, src_text, tgt_text, direction)
    skipped = 0

    for i in range(data.shape[0]):
        tokens = data[i].tolist()
        if assistant_id not in tokens:
            skipped += 1
            continue

        asst_pos = tokens.index(assistant_id)
        # Find user token to extract prompt
        user_positions = [j for j, t in enumerate(tokens) if t == user_id]
        if not user_positions:
            skipped += 1
            continue

        # Decode prompt and response
        prompt_tokens = tokens[user_positions[0]+1:asst_pos]
        # Find eos after assistant
        eos_positions = [j for j in range(asst_pos, len(tokens)) if tokens[j] == eos_id]
        if not eos_positions:
            response_tokens = tokens[asst_pos+1:]
        else:
            response_tokens = tokens[asst_pos+1:eos_positions[0]]

        try:
            prompt_text = tok.decode(prompt_tokens, skip_special_tokens=True).strip()
            response_text = tok.decode(response_tokens, skip_special_tokens=True).strip()
        except:
            skipped += 1
            continue

        if not prompt_text or not response_text:
            skipped += 1
            continue

        # Determine direction from prompt
        import re
        if '翻译为英' in prompt_text:
            # zh→en: extract source (Chinese text after prompt instruction)
            src = re.sub(r'^.*?[：:]\s*', '', prompt_text, count=1).strip()
            if '\n\n' in prompt_text:
                src = prompt_text.split('\n\n', 1)[1].strip()
            pairs.append((i, src, response_text, 'zh-en'))
        elif 'Translate' in prompt_text:
            # en→zh: extract source (English text after prompt instruction)
            src = prompt_text.split('\n\n', 1)[1].strip() if '\n\n' in prompt_text else prompt_text
            pairs.append((i, src, response_text, 'en-zh'))
        else:
            skipped += 1
            continue

        if len(pairs) % 10000 == 0 and len(pairs) > 0:
            print(f"  decoded {len(pairs)} pairs...")

    print(f"Decoded {len(pairs)} pairs, skipped {skipped}")
    print(f"  zh-en: {sum(1 for p in pairs if p[3]=='zh-en')}")
    print(f"  en-zh: {sum(1 for p in pairs if p[3]=='en-zh')}")

    # Score with COMET
    from comet import load_from_checkpoint
    comet_path = None
    for p in [
        os.path.expanduser("~/.cache/comet/models--Unbabel--wmt22-comet-da/snapshots/2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt"),
        os.path.expanduser("~/.cache/comet/wmt22-comet-da/checkpoints/model.ckpt"),
    ]:
        if os.path.exists(p):
            comet_path = p
            break

    print(f"Loading COMET from {comet_path}")
    model = load_from_checkpoint(comet_path)

    # COMET needs: src, mt (hypothesis), ref (reference)
    # We don't have references, so use reference-free mode or use src=mt=translation
    # Actually, wmt22-comet-da needs ref. Use "referenceless" scoring: src + mt only
    # Or just use src and mt, leave ref empty
    comet_data = []
    for idx, src, tgt, direction in pairs:
        comet_data.append({"src": src, "mt": tgt, "ref": tgt})  # self-reference as proxy

    print(f"Scoring {len(comet_data)} pairs with COMET...")
    output = model.predict(comet_data, batch_size=args.batch_size, gpus=1)
    scores = output.scores

    # Analyze score distribution
    scores_arr = np.array(scores)
    print(f"\nCOMET score distribution:")
    print(f"  mean: {scores_arr.mean():.4f}")
    print(f"  median: {np.median(scores_arr):.4f}")
    print(f"  p10: {np.percentile(scores_arr, 10):.4f}")
    print(f"  p25: {np.percentile(scores_arr, 25):.4f}")
    print(f"  p75: {np.percentile(scores_arr, 75):.4f}")

    # Filter
    keep_indices = []
    for j, (idx, src, tgt, direction) in enumerate(pairs):
        if scores[j] >= args.threshold:
            keep_indices.append(idx)

    print(f"\nFiltered: {len(keep_indices)}/{len(pairs)} pairs above threshold {args.threshold}")

    # Also keep chunks without assistant token (boundary chunks)
    no_asst_indices = [i for i in range(data.shape[0]) if assistant_id not in data[i].tolist()]
    keep_indices.extend(no_asst_indices)
    keep_indices = sorted(set(keep_indices))

    # Save filtered data
    filtered_data = data[keep_indices]
    torch.save(filtered_data.to(torch.int32), args.output)
    print(f"Saved {filtered_data.shape[0]} chunks to {args.output}")
    print(f"  ({filtered_data.shape[0] * data.shape[1] / 1e6:.1f}M tokens)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default="./HY-MT1.5-1.8B-new-tok")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    main(args)

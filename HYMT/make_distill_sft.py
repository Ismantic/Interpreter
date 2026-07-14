"""
Combine distillation source/target pairs into SFT JSONL format.
Output: one JSONL file with {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}

Usage:
    python make_distill_sft.py --output ./private/sft_distill.jsonl
"""
import json
import os
import argparse

ZH2EN_PROMPT = "将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：\n\n{src}"
EN2ZH_PROMPT = "Translate the following segment into Chinese, without additional explanation.\n\n{src}"


def load_pairs(src_file, tgt_file):
    with open(src_file, 'r', encoding='utf8') as f:
        sources = [l.strip() for l in f if l.strip()]
    with open(tgt_file, 'r', encoding='utf8') as f:
        targets = [l.strip() for l in f if l.strip()]
    n = min(len(sources), len(targets))
    return sources[:n], targets[:n]


def main(args):
    base = os.path.dirname(os.path.abspath(__file__))
    priv = os.path.join(base, "private")

    all_pairs = []

    # zh→en distillation pairs
    zh2en_files = [
        ("distill_skypile_zh.txt", "distill_skypile_en.txt"),
        ("distill_fineweb_zh.txt", "distill_fineweb_en.txt"),
        ("distill_zh.txt", "distill_zh2en.txt"),
    ]
    # Newly distilled
    if os.path.exists(os.path.join(priv, "distill_news_zh.txt")):
        zh2en_files.append(("distill_news_zh.txt", "distill_news_en.txt"))
    if os.path.exists(os.path.join(priv, "distill_hn_zh.txt")):
        zh2en_files.append(("distill_hn_zh.txt", "distill_hn_en.txt"))

    for src_name, tgt_name in zh2en_files:
        src_path = os.path.join(priv, src_name)
        tgt_path = os.path.join(priv, tgt_name)
        if not os.path.exists(src_path) or not os.path.exists(tgt_path):
            print(f"  Skipping {src_name} (not found)")
            continue
        sources, targets = load_pairs(src_path, tgt_path)
        for s, t in zip(sources, targets):
            if len(s) < 5 or len(t) < 5:
                continue
            all_pairs.append({
                "messages": [
                    {"role": "user", "content": ZH2EN_PROMPT.format(src=s)},
                    {"role": "assistant", "content": t}
                ]
            })
        print(f"  zh→en: {src_name} → {len(sources)} pairs")

    # en→zh distillation pairs
    en2zh_files = [
        ("distill_en_src.txt", "distill_en2zh.txt"),
        ("distill_fwedu_en.txt", "distill_fwedu_zh.txt"),
    ]
    if os.path.exists(os.path.join(priv, "distill_cnn_en.txt")):
        en2zh_files.append(("distill_cnn_en.txt", "distill_cnn_zh.txt"))
    if os.path.exists(os.path.join(priv, "distill_fineweb2_en.txt")):
        en2zh_files.append(("distill_fineweb2_en.txt", "distill_fineweb2_zh.txt"))

    for src_name, tgt_name in en2zh_files:
        src_path = os.path.join(priv, src_name)
        tgt_path = os.path.join(priv, tgt_name)
        if not os.path.exists(src_path) or not os.path.exists(tgt_path):
            print(f"  Skipping {src_name} (not found)")
            continue
        sources, targets = load_pairs(src_path, tgt_path)
        for s, t in zip(sources, targets):
            if len(s) < 5 or len(t) < 5:
                continue
            all_pairs.append({
                "messages": [
                    {"role": "user", "content": EN2ZH_PROMPT.format(src=s)},
                    {"role": "assistant", "content": t}
                ]
            })
        print(f"  en→zh: {src_name} → {len(sources)} pairs")

    # Shuffle
    import random
    random.seed(42)
    random.shuffle(all_pairs)

    # Save
    with open(args.output, 'w', encoding='utf8') as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    print(f"\nTotal: {len(all_pairs)} pairs saved to {args.output}")

    # Count by direction
    zh2en = sum(1 for p in all_pairs if '翻译为英' in p['messages'][0]['content'])
    en2zh = sum(1 for p in all_pairs if 'Translate' in p['messages'][0]['content'])
    print(f"  zh→en: {zh2en}")
    print(f"  en→zh: {en2zh}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="./private/sft_distill.jsonl")
    args = parser.parse_args()
    main(args)

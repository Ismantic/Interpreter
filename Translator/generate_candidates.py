"""
Generate multiple translation candidates for CPO preference data.
Uses temperature sampling to get diverse candidates.

Usage:
    python generate_candidates.py \
        --model_path ./output_1.7b_base_v2 \
        --output ./cpo_candidates.jsonl \
        --n_candidates 5 \
        --temperature 0.7
"""
import os
import sys
import json
import argparse
import torch
import sacrebleu
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_ZH2EN = "Translate the following text from Chinese to English.\nChinese: {src}\nEnglish:"
PROMPT_EN2ZH = "Translate the following text from English to Chinese.\nEnglish: {src}\nChinese:"


def generate_candidates(model, tokenizer, sources, prompt_template, n_candidates, temperature, batch_size=8, max_new_tokens=256):
    device = next(model.parameters()).device
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    all_candidates = []

    for i in range(0, len(sources), batch_size):
        batch_src = sources[i:i + batch_size]
        prompts = [prompt_template.format(src=s) for s in batch_src]

        all_ids = []
        for p in prompts:
            text = f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_ids.append(ids)

        max_len = max(len(ids) for ids in all_ids)
        pad_id = tokenizer.pad_token_id
        input_ids = torch.tensor([
            [pad_id] * (max_len - len(ids)) + ids for ids in all_ids
        ], device=device)
        attention_mask = input_ids.ne(pad_id)

        batch_candidates = [[] for _ in range(len(batch_src))]

        for k in range(n_candidates):
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True if k > 0 else False,  # first is greedy
                    temperature=temperature if k > 0 else 1.0,
                    top_p=0.9 if k > 0 else 1.0,
                    eos_token_id=im_end_id,
                )

            for j, output in enumerate(outputs):
                gen_ids = output[input_ids.shape[1]:]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                if "<think>" in text:
                    text = text.split("</think>")[-1].strip()
                batch_candidates[j].append(text)

        all_candidates.extend(batch_candidates)

        done = min(i + batch_size, len(sources))
        if done % 100 == 0 or done == len(sources):
            print(f"  generated {done}/{len(sources)}")

    return all_candidates


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
    ).cuda()
    model.eval()

    # Use WMT test sets as source for candidate generation
    zh_sentences, en_refs = [], []
    en_sentences, zh_refs = [], []
    for testset in ["wmt17", "wmt18", "wmt19", "wmt20", "wmt21"]:
        try:
            src_file = sacrebleu.get_source_file(testset, "zh-en")
            ref_files = sacrebleu.get_reference_files(testset, "zh-en")
            with open(src_file) as f:
                zh = [l.strip() for l in f]
            with open(ref_files[0]) as f:
                en = [l.strip() for l in f]
            zh_sentences.extend(zh)
            en_refs.extend(en)
        except:
            pass
        try:
            src_file = sacrebleu.get_source_file(testset, "en-zh")
            ref_files = sacrebleu.get_reference_files(testset, "en-zh")
            with open(src_file) as f:
                en = [l.strip() for l in f]
            with open(ref_files[0]) as f:
                zh = [l.strip() for l in f]
            en_sentences.extend(en)
            zh_refs.extend(zh)
        except:
            pass
    print(f"Source data: {len(zh_sentences)} zh→en, {len(en_sentences)} en→zh")

    results = []

    # zh→en candidates
    print("\n=== Generating zh→en candidates ===")
    zh2en_candidates = generate_candidates(
        model, tokenizer, zh_sentences,
        PROMPT_ZH2EN, args.n_candidates, args.temperature,
        batch_size=args.batch_size,
    )
    for src, ref, candidates in zip(zh_sentences, en_refs, zh2en_candidates):
        results.append({
            "direction": "zh-en",
            "source": src,
            "reference": ref,
            "candidates": candidates,
        })

    # en→zh candidates
    print("\n=== Generating en→zh candidates ===")
    en2zh_candidates = generate_candidates(
        model, tokenizer, en_sentences,
        PROMPT_EN2ZH, args.n_candidates, args.temperature,
        batch_size=args.batch_size,
    )
    for src, ref, candidates in zip(en_sentences, zh_refs, en2zh_candidates):
        results.append({
            "direction": "en-zh",
            "source": src,
            "reference": ref,
            "candidates": candidates,
        })

    # Save
    with open(args.output, 'w', encoding='utf8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"\nSaved {len(results)} examples to {args.output}")
    print(f"  zh→en: {len(zh_sentences)}, en→zh: {len(en_sentences)}")
    print(f"  {args.n_candidates} candidates each")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="./cpo_candidates.jsonl")
    parser.add_argument("--n_candidates", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    main(args)

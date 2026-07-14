"""Eval HY-MT 7B on WMT23 using vLLM with its native chat template."""
import os
import argparse
import time
import sacrebleu
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

PROMPT_ZH2EN = "Translate the following text from Chinese to English.\nChinese: {src}\nEnglish:"
PROMPT_EN2ZH = "Translate the following text from English to Chinese.\nEnglish: {src}\nChinese:"


def load_testset(testset, direction):
    src_file = sacrebleu.get_source_file(testset, direction)
    ref_files = sacrebleu.get_reference_files(testset, direction)
    with open(src_file) as f:
        sources = [line.strip() for line in f]
    with open(ref_files[0]) as f:
        references = [line.strip() for line in f]
    return sources, references


def translate(model, tokenizer, sources, prompt_template):
    prompts = []
    for s in sources:
        user_text = prompt_template.format(src=s)
        # Use HY-MT chat template
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)
    params = SamplingParams(max_tokens=256, temperature=0, stop_token_ids=[127960, 127967])
    outputs = model.generate(prompts, params)
    return [out.outputs[0].text.strip() for out in outputs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--testset", type=str, default="wmt23")
    parser.add_argument("--direction", type=str, default="both", choices=["zh-en", "en-zh", "both"])
    args = parser.parse_args()

    print(f"Loading model {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = LLM(model=args.model_path, dtype="bfloat16",
                trust_remote_code=True, max_model_len=2048,
                gpu_memory_utilization=0.85)

    directions = ["zh-en", "en-zh"] if args.direction == "both" else [args.direction]
    all_results = {}
    all_translations = {}
    all_sources = {}
    all_references = {}

    for direction in directions:
        src_lang, tgt_lang = direction.split("-")
        prompt_template = PROMPT_ZH2EN if src_lang == "zh" else PROMPT_EN2ZH
        sources, references = load_testset(args.testset, direction)
        print(f"\nTranslating {args.testset} {direction} ({len(sources)} sentences)...")
        t0 = time.time()
        translations = translate(model, tokenizer, sources, prompt_template)
        elapsed = time.time() - t0
        print(f"Done in {elapsed:.1f}s ({len(sources)/elapsed:.1f} sent/s)")
        all_translations[direction] = translations
        all_sources[direction] = sources
        all_references[direction] = references
        bleu = sacrebleu.corpus_bleu(translations, [references],
                                      tokenize="zh" if tgt_lang == "zh" else "13a")
        all_results[direction] = {"bleu": bleu.score}
        print(f"BLEU: {bleu.score:.2f}")
        # Sample
        for k in range(min(2, len(sources))):
            print(f"  src: {sources[k][:80]}")
            print(f"  hyp: {translations[k][:80]}")

    # Free vLLM
    del model
    import torch; torch.cuda.empty_cache()
    import gc; gc.collect()

    # COMET
    print("\nLoading COMET...")
    from comet import load_from_checkpoint
    ckpt = os.path.expanduser("~/.cache/comet/models--Unbabel--wmt22-comet-da/snapshots/2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt")
    comet_model = load_from_checkpoint(ckpt)

    for direction in directions:
        print(f"\nComputing COMET for {direction}...")
        comet_data = [{"src": s, "mt": t, "ref": r}
                      for s, t, r in zip(all_sources[direction], all_translations[direction], all_references[direction])]
        comet_output = comet_model.predict(comet_data, batch_size=64, gpus=1)
        all_results[direction]["comet"] = sum(comet_output.scores) / len(comet_output.scores)
        print(f"COMET: {all_results[direction]['comet']:.4f}")

    print(f"\n{'='*60}")
    print(f"Summary HY-MT 7B on {args.testset}")
    print(f"{'='*60}")
    for d, r in all_results.items():
        print(f"  {d}: BLEU = {r['bleu']:.2f} | COMET = {r['comet']:.4f}")

    # Save translations for later use
    import json
    out_path = f"hymt7b_{args.testset}_translations.json"
    with open(out_path, "w") as f:
        json.dump({d: {"sources": all_sources[d], "translations": all_translations[d],
                       "references": all_references[d], **all_results[d]}
                   for d in directions}, f, ensure_ascii=False, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

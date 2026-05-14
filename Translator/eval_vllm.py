"""
Fast evaluation using vLLM for translation.
Usage: python eval_vllm.py --model_path ./output_xxx --testset wmt23 --direction both
"""
import os
import argparse
import time
import sacrebleu
from vllm import LLM, SamplingParams

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


def translate(model, sources, prompt_template):
    prompts = [f"<|im_start|>user\n{prompt_template.format(src=s)}<|im_end|>\n<|im_start|>assistant\n" for s in sources]
    params = SamplingParams(max_tokens=256, temperature=0, stop=["<|im_end|>"])
    outputs = model.generate(prompts, params)
    return [out.outputs[0].text.strip() for out in outputs]


def evaluate(model, testset, direction, comet_model=None):
    src_lang, tgt_lang = direction.split("-")
    prompt_template = PROMPT_ZH2EN if src_lang == "zh" else PROMPT_EN2ZH
    sources, references = load_testset(testset, direction)

    print(f"\n{'='*60}")
    print(f"Evaluating {testset} {direction} ({len(sources)} sentences)")
    print(f"{'='*60}")

    t0 = time.time()
    translations = translate(model, sources, prompt_template)
    elapsed = time.time() - t0

    bleu = sacrebleu.corpus_bleu(translations, [references],
                                  tokenize="zh" if tgt_lang == "zh" else "13a")
    print(f"Time: {elapsed:.1f}s ({len(sources)/elapsed:.1f} sent/s)")
    print(f"BLEU: {bleu.score:.2f}")

    results = {"bleu": bleu.score}

    if comet_model is not None:
        print("Computing COMET...")
        comet_data = [{"src": s, "mt": t, "ref": r}
                      for s, t, r in zip(sources, translations, references)]
        comet_output = comet_model.predict(comet_data, batch_size=64, gpus=1)
        results["comet"] = sum(comet_output.scores) / len(comet_output.scores)
        print(f"COMET: {results['comet']:.4f}")

    print(f"\nSamples:")
    for k in range(min(3, len(sources))):
        print(f"  src: {sources[k][:80]}")
        print(f"  hyp: {translations[k][:80]}")
        print(f"  ref: {references[k][:80]}")
        print()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--testset", type=str, default="wmt23")
    parser.add_argument("--direction", type=str, default="both", choices=["zh-en", "en-zh", "both"])
    parser.add_argument("--no_comet", action="store_true")
    args = parser.parse_args()

    # Load COMET
    comet_model = None
    if not args.no_comet:
        print("Loading COMET model...")
        from comet import load_from_checkpoint
        ckpt = os.path.expanduser("~/.cache/comet/models--Unbabel--wmt22-comet-da/snapshots/2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt")
        if os.path.exists(ckpt):
            comet_model = load_from_checkpoint(ckpt)
            print("COMET loaded.")

    # Load model with vLLM
    print(f"Loading model {args.model_path} with vLLM...")
    model = LLM(model=args.model_path, dtype="bfloat16", max_model_len=512)

    directions = ["zh-en", "en-zh"] if args.direction == "both" else [args.direction]
    all_results = {}

    # First pass: translate with vLLM (no COMET yet)
    all_translations = {}
    all_sources = {}
    all_references = {}
    for direction in directions:
        src_lang, tgt_lang = direction.split("-")
        prompt_template = PROMPT_ZH2EN if src_lang == "zh" else PROMPT_EN2ZH
        sources, references = load_testset(args.testset, direction)
        print(f"\nTranslating {args.testset} {direction} ({len(sources)} sentences)...")
        t0 = time.time()
        translations = translate(model, sources, prompt_template)
        elapsed = time.time() - t0
        print(f"Done in {elapsed:.1f}s ({len(sources)/elapsed:.1f} sent/s)")
        all_translations[direction] = translations
        all_sources[direction] = sources
        all_references[direction] = references

        bleu = sacrebleu.corpus_bleu(translations, [references],
                                      tokenize="zh" if tgt_lang == "zh" else "13a")
        all_results[direction] = {"bleu": bleu.score}
        print(f"BLEU: {bleu.score:.2f}")

    # Free vLLM GPU memory
    del model
    import torch; torch.cuda.empty_cache()
    import gc; gc.collect()

    # Second pass: COMET scoring
    if comet_model is not None:
        for direction in directions:
            print(f"\nComputing COMET for {direction}...")
            comet_data = [{"src": s, "mt": t, "ref": r}
                          for s, t, r in zip(all_sources[direction], all_translations[direction], all_references[direction])]
            comet_output = comet_model.predict(comet_data, batch_size=64, gpus=1)
            all_results[direction]["comet"] = sum(comet_output.scores) / len(comet_output.scores)
            print(f"COMET: {all_results[direction]['comet']:.4f}")

    print(f"\n{'='*60}")
    print(f"Summary ({args.testset})")
    print(f"{'='*60}")
    for d, r in all_results.items():
        comet_str = f" | COMET = {r['comet']:.4f}" if 'comet' in r else ""
        print(f"  {d}: BLEU = {r['bleu']:.2f}{comet_str}")


if __name__ == "__main__":
    main()

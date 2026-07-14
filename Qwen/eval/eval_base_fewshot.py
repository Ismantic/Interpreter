"""
Few-shot baseline eval for a *base* (non-instruction-tuned) model, e.g. Qwen3-1.7B-Base.
Measures the "alignment tax": raw base few-shot vs the SFT/CPO/GRPO aligned models.

Same WMT23 loading, sacrebleu tokenization, and COMET checkpoint as eval_vllm.py, so
numbers are directly comparable. Difference: plain text-completion few-shot prompt
(NO ChatML), with demonstrations drawn from WMT21 (disjoint from the WMT23 test set).

Usage (from Qwen/ root):
    python eval/eval_base_fewshot.py --model_path ./models/Qwen3-1.7B-Base --testset wmt23 --shots 5
"""
import os
import argparse
import time
import sacrebleu
from vllm import LLM, SamplingParams

# Same instruction wording as eval_vllm.py, but here each demo also shows the answer.
BLOCK_ZH2EN = "Translate the following text from Chinese to English.\nChinese: {src}\nEnglish:{tgt}"
BLOCK_EN2ZH = "Translate the following text from English to Chinese.\nEnglish: {src}\nChinese:{tgt}"


def load_pairs(testset, direction):
    src_file = sacrebleu.get_source_file(testset, direction)
    ref_files = sacrebleu.get_reference_files(testset, direction)
    with open(src_file) as f:
        sources = [line.strip() for line in f]
    with open(ref_files[0]) as f:
        references = [line.strip() for line in f]
    return sources, references


def build_fewshot_prefix(direction, shots, demo_testset="wmt21"):
    block = BLOCK_ZH2EN if direction == "zh-en" else BLOCK_EN2ZH
    demo_src, demo_ref = load_pairs(demo_testset, direction)
    parts = []
    for i in range(min(shots, len(demo_src))):
        parts.append(block.format(src=demo_src[i], tgt=" " + demo_ref[i]))
    return "\n\n".join(parts)


def translate(model, sources, direction, prefix):
    block = BLOCK_ZH2EN if direction == "zh-en" else BLOCK_EN2ZH
    prompts = [prefix + "\n\n" + block.format(src=s, tgt="") for s in sources]
    # Base model: stop at the newline ending the completion (translations are single-line).
    params = SamplingParams(max_tokens=256, temperature=0,
                            stop=["\n", "Translate the following"])
    outputs = model.generate(prompts, params)
    return [out.outputs[0].text.strip() for out in outputs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--testset", type=str, default="wmt23")
    parser.add_argument("--demo_testset", type=str, default="wmt21")
    parser.add_argument("--direction", type=str, default="both", choices=["zh-en", "en-zh", "both"])
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--no_comet", action="store_true")
    args = parser.parse_args()

    print(f"Loading base model {args.model_path} with vLLM ({args.shots}-shot demos from {args.demo_testset})...")
    model = LLM(model=args.model_path, dtype="bfloat16", max_model_len=2048)

    directions = ["zh-en", "en-zh"] if args.direction == "both" else [args.direction]
    all_results, all_tr, all_src, all_ref = {}, {}, {}, {}

    for direction in directions:
        _, tgt_lang = direction.split("-")
        prefix = build_fewshot_prefix(direction, args.shots, args.demo_testset)
        sources, references = load_pairs(args.testset, direction)
        print(f"\nTranslating {args.testset} {direction} ({len(sources)} sentences, {args.shots}-shot)...")
        t0 = time.time()
        translations = translate(model, sources, direction, prefix)
        print(f"Done in {time.time()-t0:.1f}s")
        all_tr[direction], all_src[direction], all_ref[direction] = translations, sources, references
        bleu = sacrebleu.corpus_bleu(translations, [references],
                                     tokenize="zh" if tgt_lang == "zh" else "13a")
        all_results[direction] = {"bleu": bleu.score}
        print(f"BLEU: {bleu.score:.2f}")
        for k in range(min(2, len(sources))):
            print(f"  src: {sources[k][:70]}\n  hyp: {translations[k][:70]}\n  ref: {references[k][:70]}")

    del model
    import torch; torch.cuda.empty_cache()
    import gc; gc.collect()

    if not args.no_comet:
        ckpt = os.path.expanduser("~/.cache/comet/models--Unbabel--wmt22-comet-da/snapshots/2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt")
        if os.path.exists(ckpt):
            print("\nLoading COMET...")
            from comet import load_from_checkpoint
            comet_model = load_from_checkpoint(ckpt)
            for direction in directions:
                data = [{"src": s, "mt": t, "ref": r}
                        for s, t, r in zip(all_src[direction], all_tr[direction], all_ref[direction])]
                out = comet_model.predict(data, batch_size=64, gpus=1)
                all_results[direction]["comet"] = sum(out.scores) / len(out.scores)

    print(f"\n{'='*60}\nBase {args.shots}-shot summary ({args.testset})\n{'='*60}")
    for d, r in all_results.items():
        comet_str = f" | COMET = {r['comet']:.4f}" if 'comet' in r else ""
        print(f"  {d}: BLEU = {r['bleu']:.2f}{comet_str}")


if __name__ == "__main__":
    main()

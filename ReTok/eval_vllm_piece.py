"""
vLLM eval for PieceTokenizer translation models.

Behavior-equivalent to ../Qwen/eval_vllm.py — same WMT testsets, same prompt
templates, same sacrebleu tokenize rules ("zh" for en→zh, "13a" for zh→en), same
wmt22-comet-da COMET checkpoint loaded from the local cache. The ONLY differences
from Qwen/eval_vllm.py are:

  - vLLM is loaded with skip_tokenizer_init=True (PieceTokenizer is not HF-fast).
  - Prompts are fed as token IDs via TokensPrompt(prompt_token_ids=...),
    built from PieceTokenizerWrapper.apply_chat_template (= <bos><user>…<assistant>).
  - SamplingParams.stop_token_ids = [eos_token_id] (id=2, our </s>) — replacing
    Qwen's stop=["<|im_end|>"].
  - Generated token IDs are decoded via wrapper.decode(..., skip_special_tokens=True).

Usage:
    python eval_vllm_piece.py --model_path ./output_v18_sft --testset wmt23 --direction both
    python eval_vllm_piece.py --model_path ./output_v18_sft --testset wmt24 --direction both --no_comet
"""
import os
import sys
import argparse
import time
import sacrebleu

# wrapper lives in the parent Interpreter repo
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer_wrapper import PieceTokenizerWrapper  # noqa: E402

# These templates are character-identical to Qwen/eval_vllm.py
PROMPT_ZH2EN = "Translate the following text from Chinese to English.\nChinese: {src}\nEnglish:"
PROMPT_EN2ZH = "Translate the following text from English to Chinese.\nEnglish: {src}\nChinese:"

# Same COMET location Qwen hardcodes
COMET_CKPT = os.path.expanduser(
    "~/.cache/comet/models--Unbabel--wmt22-comet-da/snapshots/"
    "2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt"
)


def load_testset(testset, direction):
    src_file = sacrebleu.get_source_file(testset, direction)
    ref_files = sacrebleu.get_reference_files(testset, direction)
    with open(src_file) as f:
        sources = [line.strip() for line in f]
    with open(ref_files[0]) as f:
        references = [line.strip() for line in f]
    return sources, references


def build_prompt_ids(tokenizer, prompt_text):
    """Build <bos><user>{prompt}<assistant> as token IDs ready for generation."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=True,
        add_generation_prompt=True,
    )


def translate(model, tokenizer, sources, prompt_template, max_tokens=256):
    """Run vLLM with pre-tokenized prompts; return decoded translations."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    prompts = []
    for s in sources:
        ids = build_prompt_ids(tokenizer, prompt_template.format(src=s))
        prompts.append(TokensPrompt(prompt_token_ids=ids))

    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        stop_token_ids=[tokenizer.eos_token_id],
    )
    outputs = model.generate(prompts, params)

    translations = []
    for out in outputs:
        gen_ids = list(out.outputs[0].token_ids)
        # Trim trailing eos if present (vLLM sometimes emits the stop token)
        if gen_ids and gen_ids[-1] == tokenizer.eos_token_id:
            gen_ids = gen_ids[:-1]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        translations.append(text.strip())
    return translations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--testset", type=str, default="wmt23")
    parser.add_argument("--direction", type=str, default="both", choices=["zh-en", "en-zh", "both"])
    parser.add_argument("--no_comet", action="store_true")
    parser.add_argument("--max_model_len", type=int, default=1024,
                        help="vLLM context budget (prompt + generation).")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Max generation tokens per sentence.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--samples_print", type=int, default=3,
                        help="How many src/hyp/ref triples to print per direction.")
    args = parser.parse_args()

    # 1. Tokenizer (CN dict auto-loaded if present in model_path)
    tokenizer = PieceTokenizerWrapper(args.model_path)
    print(f"Loaded tokenizer (vocab={tokenizer.vocab_size}, "
          f"bos={tokenizer.bos_token_id}, eos={tokenizer.eos_token_id}, "
          f"pad={tokenizer.pad_token_id})")

    # 2. vLLM — skip tokenizer init since piece is not HF-fast
    print(f"Loading vLLM engine from {args.model_path} (skip_tokenizer_init=True)...")
    from vllm import LLM
    model = LLM(
        model=args.model_path,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        skip_tokenizer_init=True,
        trust_remote_code=True,
    )

    directions = ["zh-en", "en-zh"] if args.direction == "both" else [args.direction]
    all_results = {}
    all_translations = {}
    all_sources = {}
    all_references = {}

    # Pass 1: translate + BLEU (vLLM occupies GPU)
    for direction in directions:
        src_lang, tgt_lang = direction.split("-")
        prompt_template = PROMPT_ZH2EN if src_lang == "zh" else PROMPT_EN2ZH
        sources, references = load_testset(args.testset, direction)
        print(f"\nTranslating {args.testset} {direction} ({len(sources)} sentences)...")

        t0 = time.time()
        translations = translate(model, tokenizer, sources, prompt_template, args.max_tokens)
        elapsed = time.time() - t0
        print(f"Done in {elapsed:.1f}s ({len(sources)/elapsed:.1f} sent/s)")

        all_translations[direction] = translations
        all_sources[direction] = sources
        all_references[direction] = references

        bleu = sacrebleu.corpus_bleu(
            translations, [references],
            tokenize="zh" if tgt_lang == "zh" else "13a",
        )
        all_results[direction] = {"bleu": bleu.score}
        print(f"BLEU ({direction}): {bleu.score:.2f}")

        if args.samples_print > 0:
            for k in range(min(args.samples_print, len(sources))):
                print(f"  src: {sources[k][:120]}")
                print(f"  hyp: {translations[k][:120]}")
                print(f"  ref: {references[k][:120]}")
                print()

    # Free vLLM before COMET
    del model
    import torch; torch.cuda.empty_cache()
    import gc; gc.collect()

    # Pass 2: COMET
    if not args.no_comet:
        if not os.path.exists(COMET_CKPT):
            print(f"[warn] COMET ckpt missing: {COMET_CKPT}; skipping COMET.")
        else:
            print("Loading COMET model from local cache...")
            from comet import load_from_checkpoint
            comet_model = load_from_checkpoint(COMET_CKPT)
            for direction in directions:
                print(f"\nCOMET {direction}...")
                comet_data = [
                    {"src": s, "mt": t, "ref": r}
                    for s, t, r in zip(all_sources[direction], all_translations[direction], all_references[direction])
                ]
                out = comet_model.predict(comet_data, batch_size=64, gpus=1)
                all_results[direction]["comet"] = sum(out.scores) / len(out.scores)
                print(f"COMET ({direction}): {all_results[direction]['comet']:.4f}")

    print(f"\n{'='*60}")
    print(f"Summary ({args.testset})  model: {args.model_path}")
    print(f"{'='*60}")
    for d, r in all_results.items():
        comet_str = f" | COMET = {r['comet']:.4f}" if 'comet' in r else ""
        print(f"  {d}: BLEU = {r['bleu']:.2f}{comet_str}")


if __name__ == "__main__":
    main()

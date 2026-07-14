"""
Few-shot baseline eval for the PieceTokenizer *base* checkpoint (phase2_ckpt_v18_tie).

The piece analogue of ../Qwen/eval/eval_base_fewshot.py: measures the pre-SFT
base with K-shot plain-text completion, so ReTok's own base baseline is
reproducible in-repo (not borrowed from the Summer side).

Same WMT23 loading, sacrebleu tokenize, and COMET checkpoint as eval_vllm_piece.py.
Differences from the chat-format eval: NO chat template — the prompt is a plain
few-shot completion (demos from WMT21, disjoint from WMT23), encoded as piece IDs
with a leading <bos>, fed via TokensPrompt; the first output line is the translation.

Usage (from ReTok/ root):
    python eval/eval_base_fewshot_piece.py --model_path ./models/phase2_ckpt_v18_tie --testset wmt23 --shots 5
"""
import os
import sys
import argparse
import time
import sacrebleu

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
from tokenizer_wrapper import PieceTokenizerWrapper  # noqa: E402

BLOCK_ZH2EN = "Translate the following text from Chinese to English.\nChinese: {src}\nEnglish:{tgt}"
BLOCK_EN2ZH = "Translate the following text from English to Chinese.\nEnglish: {src}\nChinese:{tgt}"

COMET_CKPT = os.path.expanduser(
    "~/.cache/comet/models--Unbabel--wmt22-comet-da/snapshots/"
    "2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt"
)


def load_pairs(testset, direction):
    src_file = sacrebleu.get_source_file(testset, direction)
    ref_files = sacrebleu.get_reference_files(testset, direction)
    with open(src_file) as f:
        sources = [line.strip() for line in f]
    with open(ref_files[0]) as f:
        references = [line.strip() for line in f]
    return sources, references


def build_prefix(direction, shots, demo_testset):
    block = BLOCK_ZH2EN if direction == "zh-en" else BLOCK_EN2ZH
    demo_src, demo_ref = load_pairs(demo_testset, direction)
    parts = [block.format(src=demo_src[i], tgt=" " + demo_ref[i]) for i in range(min(shots, len(demo_src)))]
    return "\n\n".join(parts)


def first_line(text):
    for delim in ("\n", "Translate the following"):
        if delim in text:
            text = text.split(delim)[0]
    return text.strip()


def translate(model, tokenizer, sources, direction, prefix, max_tokens):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    block = BLOCK_ZH2EN if direction == "zh-en" else BLOCK_EN2ZH
    prompts = []
    for s in sources:
        text = prefix + "\n\n" + block.format(src=s, tgt="")
        ids = [tokenizer.bos_token_id] + tokenizer.encode(text, add_special_tokens=False)
        prompts.append(TokensPrompt(prompt_token_ids=ids))

    params = SamplingParams(max_tokens=max_tokens, temperature=0.0,
                            stop_token_ids=[tokenizer.eos_token_id])
    outputs = model.generate(prompts, params)
    return [first_line(tokenizer.decode(list(o.outputs[0].token_ids), skip_special_tokens=True))
            for o in outputs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="./models/phase2_ckpt_v18_tie")
    ap.add_argument("--testset", default="wmt23")
    ap.add_argument("--demo_testset", default="wmt21")
    ap.add_argument("--direction", default="both", choices=["zh-en", "en-zh", "both"])
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--max_tokens", type=int, default=160)
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--no_comet", action="store_true")
    args = ap.parse_args()

    tokenizer = PieceTokenizerWrapper(args.model_path)
    print(f"tokenizer vocab={tokenizer.vocab_size} bos={tokenizer.bos_token_id} eos={tokenizer.eos_token_id}")

    from vllm import LLM
    model = LLM(model=args.model_path, dtype="bfloat16", max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                skip_tokenizer_init=True, trust_remote_code=True)

    directions = ["zh-en", "en-zh"] if args.direction == "both" else [args.direction]
    res, tr, src_all, ref_all = {}, {}, {}, {}
    for d in directions:
        _, tgt = d.split("-")
        prefix = build_prefix(d, args.shots, args.demo_testset)
        sources, references = load_pairs(args.testset, d)
        print(f"\n{args.testset} {d} ({len(sources)} sents, {args.shots}-shot)...")
        t0 = time.time()
        translations = translate(model, tokenizer, sources, d, prefix, args.max_tokens)
        print(f"  done {time.time()-t0:.1f}s")
        tr[d], src_all[d], ref_all[d] = translations, sources, references
        bleu = sacrebleu.corpus_bleu(translations, [references], tokenize="zh" if tgt == "zh" else "13a")
        res[d] = {"bleu": bleu.score}
        print(f"  BLEU({d}) = {bleu.score:.2f}")
        for k in range(min(2, len(sources))):
            print(f"    src: {sources[k][:80]}\n    hyp: {translations[k][:80]}\n    ref: {references[k][:80]}")

    del model
    import torch; torch.cuda.empty_cache()
    import gc; gc.collect()

    if not args.no_comet and os.path.exists(COMET_CKPT):
        from comet import load_from_checkpoint
        cm = load_from_checkpoint(COMET_CKPT)
        for d in directions:
            data = [{"src": s, "mt": t, "ref": r} for s, t, r in zip(src_all[d], tr[d], ref_all[d])]
            out = cm.predict(data, batch_size=64, gpus=1)
            res[d]["comet"] = sum(out.scores) / len(out.scores)

    print(f"\n{'='*60}\nBase {args.shots}-shot ({args.testset})  {args.model_path}\n{'='*60}")
    for d, r in res.items():
        c = f" | COMET = {r['comet']:.4f}" if "comet" in r else ""
        print(f"  {d}: BLEU = {r['bleu']:.2f}{c}")


if __name__ == "__main__":
    main()

"""
Evaluate HY-MT translation model on WMT test sets.
Reports BLEU and COMET scores for zh->en and en->zh directions.
Supports both HuggingFace tokenizer and custom piece_tokenizer.

Usage:
    python eval.py --model_path ./HY-MT1.5-1.8B
    python eval.py --model_path ./HY-MT1.5-1.8B-new-tok --max_samples 200
    python eval.py --model_path ./HY-MT1.5-1.8B --no_comet --direction zh-en
"""
import os
import argparse
import time
import torch
import sacrebleu
from transformers import AutoModelForCausalLM

# Prompt templates from HY-MT
PROMPT_ZH2EN = "将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：\n\n{src}"
PROMPT_EN2ZH = "Translate the following segment into Chinese, without additional explanation.\n\n{src}"


def load_tokenizer(model_path):
    """Load tokenizer - auto-detect piece_tokenizer vs HuggingFace."""
    if os.path.exists(os.path.join(model_path, "piece.model")):
        from tokenizer_wrapper import PieceTokenizerWrapper
        return PieceTokenizerWrapper(model_path)
    else:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def load_testset(testset, direction):
    src_file = sacrebleu.get_source_file(testset, direction)
    ref_files = sacrebleu.get_reference_files(testset, direction)
    with open(src_file) as f:
        sources = [line.strip() for line in f]
    with open(ref_files[0]) as f:
        references = [line.strip() for line in f]
    return sources, references


def get_pad_id(tokenizer):
    """Get pad token id from either tokenizer type."""
    if hasattr(tokenizer, 'pad_token_id'):
        return tokenizer.pad_token_id
    return tokenizer.encode(tokenizer.pad_token)[0]


def translate_batch(model, tokenizer, sources, prompt_template, batch_size=4, max_new_tokens=256):
    translations = []
    model.eval()
    pad_id = get_pad_id(tokenizer)

    for i in range(0, len(sources), batch_size):
        batch_src = sources[i:i + batch_size]
        prompts = [prompt_template.format(src=s) for s in batch_src]
        messages_list = [[{"role": "user", "content": p}] for p in prompts]

        # Encode each message
        all_ids = []
        for msgs in messages_list:
            ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
            if not isinstance(ids, list):
                ids = ids.tolist() if hasattr(ids, 'tolist') else list(ids)
            all_ids.append(ids)

        # Left-pad to same length
        max_len = max(len(ids) for ids in all_ids)
        input_ids = torch.tensor([
            [pad_id] * (max_len - len(ids)) + ids for ids in all_ids
        ], device=model.device)
        attention_mask = input_ids.ne(pad_id)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        for j, output in enumerate(outputs):
            gen_ids = output[input_ids.shape[1]:].tolist()
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            translations.append(text)

        if (i // batch_size + 1) % 10 == 0:
            print(f"  translated {min(i + batch_size, len(sources))}/{len(sources)}")

    return translations


def compute_comet(sources, translations, references, comet_model, use_cpu=False):
    data = [{"src": s, "mt": t, "ref": r} for s, t, r in zip(sources, translations, references)]
    output = comet_model.predict(data, batch_size=32, gpus=0 if use_cpu else 1, progress_bar=False)
    return output.system_score


def evaluate_direction(model, tokenizer, testset, direction, batch_size, max_samples, comet_model=None):
    sources, references = load_testset(testset, direction)
    if max_samples:
        sources = sources[:max_samples]
        references = references[:max_samples]

    prompt_template = PROMPT_ZH2EN if direction == "zh-en" else PROMPT_EN2ZH
    tgt_lang = "en" if direction == "zh-en" else "zh"

    print(f"\n{'='*60}")
    print(f"Evaluating {testset} {direction} ({len(sources)} sentences)")
    print(f"{'='*60}")

    t0 = time.time()
    translations = translate_batch(model, tokenizer, sources, prompt_template, batch_size)
    elapsed = time.time() - t0

    bleu = sacrebleu.corpus_bleu(translations, [references],
                                  tokenize="zh" if tgt_lang == "zh" else "13a")

    results = {"bleu": bleu.score, "time": elapsed}
    print(f"Time: {elapsed:.1f}s ({len(sources)/elapsed:.1f} sent/s)")
    print(f"BLEU: {bleu.score:.2f}")

    if comet_model is not None:
        print("Computing COMET...")
        comet_score = compute_comet(sources, translations, references, comet_model, use_cpu=_USE_CPU)
        results["comet"] = comet_score
        print(f"COMET: {comet_score:.4f}")

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
    parser.add_argument("--testset", type=str, default="wmt22",
                        choices=["wmt17", "wmt18", "wmt19", "wmt20", "wmt21", "wmt22", "wmt23"])
    parser.add_argument("--direction", type=str, default="both", choices=["zh-en", "en-zh", "both"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--no_comet", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU (slow but doesn't need GPU)")
    args = parser.parse_args()

    comet_model = None
    if not args.no_comet:
        print("Loading COMET model...")
        from comet import load_from_checkpoint
        # Use local path if available, otherwise download
        local_comet = os.path.expanduser("~/.cache/comet/models--Unbabel--wmt22-comet-da")
        ckpt = None
        if os.path.isdir(local_comet):
            for root, dirs, files in os.walk(local_comet):
                for f in files:
                    if f == "model.ckpt":
                        ckpt = os.path.join(root, f)
                        break
        if ckpt is None:
            # Also check git clone location
            alt = os.path.expanduser("~/new/wmt22-comet-da/checkpoints/model.ckpt")
            if os.path.exists(alt):
                ckpt = alt
        if ckpt is None:
            from comet import download_model
            ckpt = download_model("Unbabel/wmt22-comet-da")
        print(f"Loading from: {ckpt}")
        comet_model = load_from_checkpoint(ckpt)
        print("COMET model loaded.")

    global _USE_CPU
    _USE_CPU = args.cpu
    device = "cpu" if args.cpu else "cuda"
    print(f"Loading model from {args.model_path}... (device={device})")
    tokenizer = load_tokenizer(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True,
        torch_dtype=torch.float32 if args.cpu else torch.bfloat16,
    )
    if not args.cpu:
        model = model.cuda()
    model.eval()

    all_results = {}
    directions = ["zh-en", "en-zh"] if args.direction == "both" else [args.direction]
    for d in directions:
        all_results[d] = evaluate_direction(
            model, tokenizer, args.testset, d, args.batch_size, args.max_samples, comet_model
        )

    print(f"\n{'='*60}")
    print(f"Summary ({args.testset})")
    print(f"{'='*60}")
    for d, scores in all_results.items():
        line = f"  {d}: BLEU = {scores['bleu']:.2f}"
        if "comet" in scores:
            line += f" | COMET = {scores['comet']:.4f}"
        print(line)


if __name__ == "__main__":
    main()

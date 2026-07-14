"""
Evaluate Qwen3 translation model on WMT test sets.
Supports BLEU + COMET (wmt22-comet-da).

Usage:
    python eval.py --model_path ./checkpoints/output_v1 --testset wmt22 --direction both
"""
import os
import argparse
import time
import torch
import sacrebleu
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_ZH2EN = "将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：\n\n{src}"
PROMPT_EN2ZH = "Translate the following segment into Chinese, without additional explanation.\n\n{src}"


def load_testset(testset, direction):
    src_file = sacrebleu.get_source_file(testset, direction)
    ref_files = sacrebleu.get_reference_files(testset, direction)
    with open(src_file) as f:
        sources = [line.strip() for line in f]
    with open(ref_files[0]) as f:
        references = [line.strip() for line in f]
    return sources, references


def translate_batch(model, tokenizer, sources, prompt_template, batch_size=8, max_new_tokens=256):
    translations = []
    device = next(model.parameters()).device

    for i in range(0, len(sources), batch_size):
        batch_src = sources[i:i + batch_size]
        prompts = [prompt_template.format(src=s) for s in batch_src]

        # ChatML format: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
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

        # Use <|im_end|> as eos for generation (Qwen3 convention)
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=im_end_id,
            )

        for j, output in enumerate(outputs):
            gen_ids = output[input_ids.shape[1]:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            # Remove any think tags that might appear
            if "<think>" in text:
                text = text.split("</think>")[-1].strip()
            translations.append(text)

        done = min(i + batch_size, len(sources))
        if (done // batch_size) % 10 == 0:
            print(f"  translated {done}/{len(sources)}")

    return translations


def evaluate(model, tokenizer, testset, direction, batch_size, comet_model=None, max_samples=None):
    src_lang, tgt_lang = direction.split("-")
    prompt_template = PROMPT_ZH2EN if src_lang == "zh" else PROMPT_EN2ZH

    sources, references = load_testset(testset, direction)
    if max_samples and max_samples < len(sources):
        import random
        random.seed(42)
        indices = random.sample(range(len(sources)), max_samples)
        sources = [sources[i] for i in indices]
        references = [references[i] for i in indices]
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

    # COMET
    if comet_model is not None:
        print("Computing COMET...")
        comet_data = [{"src": s, "mt": t, "ref": r}
                      for s, t, r in zip(sources, translations, references)]
        comet_output = comet_model.predict(comet_data, batch_size=64, gpus=1)
        comet_score = comet_output.scores
        results["comet"] = sum(comet_score) / len(comet_score)
        print(f"COMET: {results['comet']:.4f}")

    # Samples
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
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--no_comet", action="store_true")
    args = parser.parse_args()

    # Load COMET
    comet_model = None
    if not args.no_comet:
        print("Loading COMET model...")
        from comet import load_from_checkpoint
        comet_path = None
        for p in [
            os.path.expanduser("~/.cache/comet/models--Unbabel--wmt22-comet-da/snapshots/2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt"),
            os.path.expanduser("~/.cache/comet/wmt22-comet-da/checkpoints/model.ckpt"),
        ]:
            if os.path.exists(p):
                comet_path = p
                break
        if comet_path:
            print(f"Loading from: {comet_path}")
            comet_model = load_from_checkpoint(comet_path)
            print("COMET model loaded.")
        else:
            print("COMET model not found, skipping.")

    # Load model
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
    ).cuda()
    model.eval()

    directions = ["zh-en", "en-zh"] if args.direction == "both" else [args.direction]
    all_results = {}

    for direction in directions:
        results = evaluate(model, tokenizer, args.testset, direction, args.batch_size, comet_model, args.max_samples)
        all_results[direction] = results

    print(f"\n{'='*60}")
    print(f"Summary ({args.testset})")
    print(f"{'='*60}")
    for d, r in all_results.items():
        comet_str = f" | COMET = {r['comet']:.4f}" if 'comet' in r else ""
        print(f"  {d}: BLEU = {r['bleu']:.2f}{comet_str}")


if __name__ == "__main__":
    main()

"""
Self-distillation: use original HY-MT model to translate monolingual texts.
Produces high-quality parallel data with consistent translation style.
"""
import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_ZH2EN = "将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：\n\n{src}"
PROMPT_EN2ZH = "Translate the following segment into Chinese, without additional explanation.\n\n{src}"


def translate_batch(model, tokenizer, sources, prompt_template, batch_size=8, max_new_tokens=256):
    translations = []
    for i in range(0, len(sources), batch_size):
        batch_src = sources[i:i + batch_size]
        prompts = [prompt_template.format(src=s) for s in batch_src]
        messages_list = [[{"role": "user", "content": p}] for p in prompts]

        all_ids = []
        for msgs in messages_list:
            ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=False)
            if not isinstance(ids, list):
                ids = ids.tolist() if hasattr(ids, 'tolist') else list(ids)
            all_ids.append(ids)

        max_len = max(len(ids) for ids in all_ids)
        pad_id = tokenizer.encode(tokenizer.pad_token)[0]
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
            gen_ids = output[input_ids.shape[1]:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            translations.append(text)

        if (i // batch_size + 1) % 50 == 0:
            print(f"  translated {min(i + batch_size, len(sources))}/{len(sources)}")

    return translations


def main(args):
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).cuda()
    model.eval()

    # Read source texts
    with open(args.input, 'r', encoding='utf8') as f:
        sources = [l.strip() for l in f if l.strip()]
    if args.max_samples:
        sources = sources[:args.max_samples]
    print(f"Loaded {len(sources)} source texts")

    # Determine direction
    prompt = PROMPT_ZH2EN if args.direction == "zh2en" else PROMPT_EN2ZH

    # Translate with streaming output
    t0 = time.time()
    max_new_tokens = 256
    f_src = open(args.output_src, 'w', encoding='utf8')
    f_tgt = open(args.output_tgt, 'w', encoding='utf8')

    for i in range(0, len(sources), args.batch_size):
        batch_src = sources[i:i + args.batch_size]
        prompts = [prompt.format(src=s) for s in batch_src]
        messages_list = [[{"role": "user", "content": p}] for p in prompts]

        all_ids = []
        for msgs in messages_list:
            ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=False)
            if not isinstance(ids, list):
                ids = ids.tolist() if hasattr(ids, 'tolist') else list(ids)
            all_ids.append(ids)

        max_len = max(len(ids) for ids in all_ids)
        pad_id = tokenizer.encode(tokenizer.pad_token)[0]
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
            gen_ids = output[input_ids.shape[1]:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            # Replace newlines to keep one line per translation
            text = text.replace('\n', ' ').replace('\r', '')
            f_src.write(batch_src[j].replace('\n', ' ') + '\n')
            f_tgt.write(text + '\n')

        done = min(i + args.batch_size, len(sources))
        elapsed = time.time() - t0
        print(f"  {done}/{len(sources)} | {elapsed:.0f}s | {done/elapsed:.1f} sent/s")

    f_src.close()
    f_tgt.close()
    elapsed = time.time() - t0
    print(f"Done: {len(sources)} texts in {elapsed:.0f}s ({len(sources)/elapsed:.1f} sent/s)")
    print(f"Saved to {args.output_src} and {args.output_tgt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--direction", type=str, required=True, choices=["zh2en", "en2zh"])
    parser.add_argument("--output_src", type=str, required=True)
    parser.add_argument("--output_tgt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    main(args)

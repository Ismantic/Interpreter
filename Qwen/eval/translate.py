"""Interactive translation REPL for the Qwen3-1.7B translator.

Usage:
    PY=/home/tfbao/new/HY-MT/.venv/bin/python
    $PY -u translate.py --model_path ./checkpoints/output_1.7b_grpo_full

Type a sentence + Enter; direction (zh->en / en->zh) is auto-detected by whether
the line contains Chinese characters. Ctrl-C to quit. Prompt format, stop token,
and greedy decoding match eval_vllm.py exactly.
"""
import sys
import argparse
from vllm import LLM, SamplingParams

PROMPT_ZH2EN = "Translate the following text from Chinese to English.\nChinese: {src}\nEnglish:"
PROMPT_EN2ZH = "Translate the following text from English to Chinese.\nEnglish: {src}\nChinese:"


def is_zh(s):
    return any("一" <= c <= "鿿" for c in s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="./checkpoints/output_1.7b_grpo_full")
    ap.add_argument("--gpu_mem", type=float, default=0.5)
    args = ap.parse_args()

    llm = LLM(model=args.model_path, gpu_memory_utilization=args.gpu_mem, max_model_len=1024)
    sp = SamplingParams(max_tokens=256, temperature=0, stop=["<|im_end|>"])

    print("\n>>> 输入句子回车翻译(中↔英自动判向),Ctrl-C / Ctrl-D 退出\n")
    try:
        for line in sys.stdin:
            src = line.strip()
            if not src:
                continue
            tmpl = PROMPT_ZH2EN if is_zh(src) else PROMPT_EN2ZH
            prompt = f"<|im_start|>user\n{tmpl.format(src=src)}<|im_end|>\n<|im_start|>assistant\n"
            out = llm.generate([prompt], sp, use_tqdm=False)[0].outputs[0].text.strip()
            print("→", out, "\n")
    except (KeyboardInterrupt, EOFError):
        print("\nbye")


if __name__ == "__main__":
    main()

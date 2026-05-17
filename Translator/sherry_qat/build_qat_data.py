"""Build the KD/QAT dataset: distill the FP teacher (output_1.7b_grpo_full).

Sequence-level KD (Kim & Rush): collect the translation prompts the model was
trained on, run the FP teacher greedily, and use the teacher's own outputs as
the QAT targets. This preserves the teacher's behaviour -- including the GRPO
gains, which the ALMA gold labels do NOT contain -- without needing teacher
logits in the training loop (so QAT fits on one 4090).

Output: qat_kd.jsonl, ChatML {"messages":[user,assistant]}, ready for train.py.
"""
import json, sys, time
from vllm import LLM, SamplingParams

T = "/home/tfbao/Shiyu/Interpreter/Translator"
TEACHER = f"{T}/output_1.7b_grpo_full"
OUT = f"{T}/sherry_qat/qat_kd.jsonl"


def collect_prompts():
    prompts = set()
    # SFT prompts
    with open(f"{T}/alma_combined_sft_clean.jsonl") as f:
        for line in f:
            if line.strip():
                prompts.add(json.loads(line)["messages"][0]["content"])
    # CPO prompts (WMT17-21)
    with open(f"{T}/cpo_preference.jsonl") as f:
        for line in f:
            if line.strip():
                prompts.add(json.loads(line)["prompt"])
    # GRPO prompts (WMT17-21)
    with open(f"{T}/grpo_data.jsonl") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)["prompt"]
                prompts.add(p[0]["content"] if isinstance(p, list) else p)
    # keep prompts that are not absurdly long
    return [p for p in prompts if len(p) < 1200]


def main():
    prompts = collect_prompts()
    print(f"collected {len(prompts)} unique prompts", flush=True)

    llm = LLM(model=TEACHER, dtype="bfloat16", max_model_len=768,
              gpu_memory_utilization=0.85)
    wrapped = [f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
               for p in prompts]
    params = SamplingParams(max_tokens=256, temperature=0.0, stop=["<|im_end|>"])

    t0 = time.time()
    outs = llm.generate(wrapped, params)
    print(f"teacher generation: {len(outs)} in {time.time()-t0:.0f}s", flush=True)

    n = 0
    with open(OUT, "w", encoding="utf8") as f:
        for p, o in zip(prompts, outs):
            tgt = o.outputs[0].text.strip()
            if not tgt:
                continue
            rec = {"messages": [{"role": "user", "content": p},
                                {"role": "assistant", "content": tgt}]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} KD pairs -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

"""Multi-testset evaluation to detect overfitting.

Reward metric (GRPO) = wmt22-comet-da. To check for reward-hacking, this also
reports BLEU and chrF (lexical, independent of COMET) on multiple held-out sets.
"""
import os
import argparse
import time
import sacrebleu
from vllm import LLM, SamplingParams

PROMPT_ZH2EN = "Translate the following text from Chinese to English.\nChinese: {src}\nEnglish:"
PROMPT_EN2ZH = "Translate the following text from English to Chinese.\nEnglish: {src}\nChinese:"

# Clean held-out only. WMT22 EXCLUDED: 17.5% leaked into ALMA SFT data.
# WMT17-21 are GRPO training sources. WMT23/24 never seen in any stage.
TESTSETS = [
    ("wmt23", "zh-en"),
    ("wmt23", "en-zh"),
    ("wmt24", "en-zh"),    # newest WMT, cleanest
    ("flores", "zh-en"),   # Wikipedia domain — different from WMT news
    ("flores", "en-zh"),
]


FLORES_DIR = "flores200_dataset/devtest"


def load_testset(testset, direction):
    if testset == "flores":
        src_lang, tgt_lang = direction.split("-")
        lang_file = {"zh": "zho_Hans.devtest", "en": "eng_Latn.devtest"}
        with open(os.path.join(FLORES_DIR, lang_file[src_lang])) as f:
            sources = [l.strip() for l in f]
        with open(os.path.join(FLORES_DIR, lang_file[tgt_lang])) as f:
            references = [l.strip() for l in f]
        return sources, references
    src_file = sacrebleu.get_source_file(testset, direction)
    ref_files = sacrebleu.get_reference_files(testset, direction)
    with open(src_file) as f:
        sources = [l.strip() for l in f]
    with open(ref_files[0]) as f:
        references = [l.strip() for l in f]
    return sources, references


def translate(model, sources, prompt_template):
    prompts = [f"<|im_start|>user\n{prompt_template.format(src=s)}<|im_end|>\n<|im_start|>assistant\n"
               for s in sources]
    params = SamplingParams(max_tokens=256, temperature=0, stop=["<|im_end|>"])
    outputs = model.generate(prompts, params)
    return [o.outputs[0].text.strip() for o in outputs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()

    print(f"Loading {args.model_path} with vLLM...")
    model = LLM(model=args.model_path, dtype="bfloat16", max_model_len=512)

    all_data = {}
    for testset, direction in TESTSETS:
        try:
            src_lang, tgt_lang = direction.split("-")
            tmpl = PROMPT_ZH2EN if src_lang == "zh" else PROMPT_EN2ZH
            sources, refs = load_testset(testset, direction)
            print(f"\n{testset} {direction}: {len(sources)} sentences")
            t0 = time.time()
            trans = translate(model, sources, tmpl)
            print(f"  translated in {time.time()-t0:.1f}s")
            bleu = sacrebleu.corpus_bleu(trans, [refs],
                                          tokenize="zh" if tgt_lang == "zh" else "13a")
            chrf = sacrebleu.corpus_chrf(trans, [refs])
            all_data[(testset, direction)] = {
                "sources": sources, "trans": trans, "refs": refs,
                "bleu": bleu.score, "chrf": chrf.score,
            }
            print(f"  BLEU={bleu.score:.2f} chrF={chrf.score:.2f}")
        except Exception as e:
            print(f"  SKIP {testset} {direction}: {e}")

    del model
    import torch, gc
    torch.cuda.empty_cache(); gc.collect()

    print("\nLoading COMET...")
    from comet import load_from_checkpoint
    ckpt = os.path.expanduser(
        "~/.cache/comet/models--Unbabel--wmt22-comet-da/snapshots/2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt"
    )
    comet = load_from_checkpoint(ckpt)

    for key, d in all_data.items():
        cdata = [{"src": s, "mt": t, "ref": r}
                 for s, t, r in zip(d["sources"], d["trans"], d["refs"])]
        scores = comet.predict(cdata, batch_size=64, gpus=1).scores
        d["comet"] = sum(scores) / len(scores)

    print(f"\n{'='*70}")
    print(f"Multi-testset eval: {args.model_path}")
    print(f"{'='*70}")
    print(f"{'testset':<16} {'COMET':>8} {'BLEU':>8} {'chrF':>8}")
    for (testset, direction), d in all_data.items():
        print(f"{testset+' '+direction:<16} {d['comet']:>8.4f} {d['bleu']:>8.2f} {d['chrf']:>8.2f}")


if __name__ == "__main__":
    main()

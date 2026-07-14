# CLAUDE.md

## Project

A/B replication of the **Qwen** project (`../Qwen/`) with two and
only two differences:

| | Qwen baseline | ReTok |
|---|---|---|
| Base model | `Qwen3-1.7B-Base` | `models/phase2_ckpt_v18_tie` (ReTok phase2 v18, copied in from Summer) |
| Tokenizer | Qwen3 HF BPE (ChatML) | PieceTokenizer (`<bos><user>…<assistant>…<eos>`) |

Same data (`alma_combined_sft_clean.jsonl` for SFT, `cpo_v3_plus_7b.jsonl` for
CPO, `grpo_data.jsonl` for GRPO), same hyperparameters, same losses, same
reward functions, same prompt template
(`Translate the following text from X to Y.\n{src lang}: {src}\n{tgt lang}:`).

Pipeline mirrors Qwen: **SFT → CPO (LoRA) → GRPO (full-param)**.

## Directory layout

Code and artifacts are separated (mirrors `../Qwen/`). **Run every command from this
`ReTok/` root** — scripts, defaults, and the commands below are relative to it.

```
ReTok/
├── train/       train.py  train_cpo.py  train_grpo.py  train_grpo_kiwi.py  merge_lora.py
├── eval/        eval_vllm_piece.py
├── lib/         tokenizer_wrapper.py  piece_hf_tokenizer.py  tok_artifacts.py
│                (shared piece-tokenizer modules; train/eval add ReTok/lib to sys.path)
├── data/        *.jsonl SFT/CPO/GRPO data (own copy, decoupled from Qwen)   (git-ignored)
├── models/      phase2_ckpt_v18_tie  (the piece base checkpoint)            (git-ignored)
├── checkpoints/ output_v18_tie_*  model checkpoints                         (git-ignored)
├── logs/        *.log run/eval logs                                         (git-ignored)
├── papers/                                                                  (git-ignored)
├── run_sft.sh  run_cpo.sh  run_grpo.sh  run_grpo_r2.sh  run_grpo_kiwi.sh  run_kiwi_chain.sh  run_all_tie.sh
└── CLAUDE.md  RUN_BEST_MODEL.md  DATA_MANIFEST.md  results.tsv
```

ReTok is self-contained: `tokenizer_wrapper.py` is its own copy (no longer imported
from `../HYMT/`), and the base checkpoint lives in `models/`. Only the C++
`piece_tokenizer` extension (in the venv) is an external dependency.

## Setup

- venv: `/home/tfbao/new/HY-MT/.venv/bin/python -u` (shared with Qwen;
  has torch, transformers, peft, trl, vllm, sacrebleu, unbabel-comet).
- GPU: single RTX 4090 (24 GB). All configs assume bs=1–2 + ga + grad ckpt.
- vLLM-with-piece pattern: `LLM(skip_tokenizer_init=True)` + feed prompts as
  `TokensPrompt(prompt_token_ids=…)`. Eval scripts do this directly; GRPO
  monkey-patches `vllm.LLM.__init__` so TRL's internal LLM(…) call inherits
  the same setting (verified TRL pre-tokenizes prompts before sending to vLLM
  at `trl/generation/vllm_generation.py:685`).

## Common commands

```bash
PY=/home/tfbao/new/HY-MT/.venv/bin/python   # run all bash/eval commands from the ReTok/ root

# === Phase 1: SFT ===
bash run_sft.sh smoke         # 50-step sanity
bash run_sft.sh               # full 1 epoch (~23 min on 4090) → output_v18_sft/

# === Phase 2: CPO ===
bash run_cpo.sh               # LoRA CPO → output_v18_cpo_v3_plus_7b/
                              # then merge → output_v18_cpo_v3_plus_7b_merged/

# === Phase 3: GRPO ===
bash run_grpo.sh              # from CPO-merged → output_v18_grpo_full/
bash run_grpo.sh from_sft     # from SFT directly → output_v18_grpo_from_sft/
                              #   (matches Qwen/output_1.7b_grpo_sft_tuned lineage)

# === Eval (any checkpoint) ===
$PY -u eval/eval_vllm_piece.py --model_path ./checkpoints/output_v18_tie_sft --testset wmt23 --direction both
$PY -u eval/eval_vllm_piece.py --model_path ./checkpoints/output_v18_tie_cpo_v3_plus_7b_merged --testset wmt24 --direction both
$PY -u eval/eval_vllm_piece.py --model_path ./checkpoints/output_v18_tie_grpo_full --testset wmt23 --direction both --no_comet
```

## Architecture

### Stage scripts

- **`train.py`** — SFT. Tokenizer-only fork of `../Qwen/train.py`.
  `TranslationSFTDataset` builds `<bos> <user> {prompt_ids} <assistant>
  {response_ids} <eos>`; loss-mask boundary unchanged (prefix → IGNORE,
  response + eos → supervised).
- **`train_cpo.py`** — CPO LoRA. Tokenizer-only fork of `../Qwen/train_cpo.py`.
  `L_CPO = -log σ(β·(logπ(y_w) - logπ(y_l))) + λ·NLL(y_w)`, LoRA r=16
  all-linear. CPODataset built with piece chat IDs.
- **`merge_lora.py`** — port of `../Qwen/merge_lora_qwen3.py`. Only diff:
  copies the 5 piece tokenizer files into the merged dir.
- **`train_grpo.py`** — GRPO via TRL. Monkey-patches `vllm.LLM.__init__` to set
  `skip_tokenizer_init=True`; uses `PieceTokenizerForTRL` (from
  `piece_hf_tokenizer.py`) as `processing_class`. Reward = same COMET + repetition
  penalty as Qwen; eos forced to `</s>`.
- **`piece_hf_tokenizer.py`** — `PreTrainedTokenizer` subclass that satisfies TRL's
  `isinstance(processing_class, PreTrainedTokenizerBase)` check. Overrides
  `apply_chat_template` directly (no Jinja), because piece's `encode_as_pieces`
  doesn't recognize inline `<user>`/`<assistant>`/etc. as control tokens — it
  BPE-splits them. Delegates vocab/encode/decode to `PieceTokenizerWrapper`.

### Eval

- **`eval_vllm_piece.py`** — behavior-equivalent to `../Qwen/eval_vllm.py`
  (same WMT testsets, same sacrebleu tokenize rules ["zh"/"13a"], same local
  COMET ckpt path). Differences: vLLM loaded with `skip_tokenizer_init=True`,
  prompts as `TokensPrompt(prompt_token_ids=…)`, completions decoded via wrapper.

### Tokenizer artifact convention

Every checkpoint dir is **self-contained**: `model.safetensors` + 5 piece
files (`piece.model`, `dict.txt`, `token_mapping.json`,
`special_tokens_map.json`, `tokenizer_config.json`). The helper
`tok_artifacts._copy_tokenizer_artifacts(base_dir, save_dir)` (in `lib/`, imported by
every module) is the single source of truth — used by SFT/CPO save loops, merge_lora,
and the final GRPO save.

## Comparison targets (from Qwen/results.tsv)

| Stage | Qwen ckpt | WMT23 zh→en (BLEU/COMET) | WMT23 en→zh |
|---|---|---|---|
| SFT  | `output_1.7b_base_v2`              | 21.58 / 0.7924 | 39.82 / 0.8556 |
| CPO  | `output_1.7b_cpo_v3_plus_7b_merged`| 19.16 / 0.8017 | 32.69 / 0.8507 |
| GRPO | `output_1.7b_grpo_sft_tuned`       | **22.85** / 0.8003 | **41.97** / 0.8540 |

Append results to `results.tsv` after each phase.

## BEST ReTok checkpoint

**`checkpoints/output_v18_tie_grpo_full/`** — the only ReTok GRPO checkpoint that
survives on disk and the canonical "best" for deployment.

WMT23: zh→en 18.43 / 0.7967, en→zh 31.79 / 0.8511.

`results.tsv` also lists two follow-up GRPO variants — `output_v18_tie_grpo_full_r2`
(2nd-round GRPO from R1) and `output_v18_tie_grpo_kiwi` (COMET + cometkiwi dual
reward). Both improved over R1 by ≤0.0006 COMET (noise floor) and have since
been deleted. Treat the three as one tier; grpo_full is the surviving
representative.

Self-contained dir: `model.safetensors` + `config.json` + `generation_config.json`
+ the 5 piece artifacts (`piece.model`, `dict.txt`, `token_mapping.json`,
`special_tokens_map.json`, `tokenizer_config.json`). Loadable via
`PieceTokenizerWrapper(model_dir)` and `AutoModelForCausalLM.from_pretrained(model_dir)`
without external file dependencies.

See `RUN_BEST_MODEL.md` for the cross-machine deployment runbook.

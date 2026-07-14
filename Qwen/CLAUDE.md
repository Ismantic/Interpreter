# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Train a 1.7B zh↔en translation model from `Qwen3-1.7B-Base` that closes the gap to
`HY-MT1.5-1.8B` (the 1.8B reference: WMT23 zh→en 0.8052, en→zh 0.8669). This is a
separate project from the sibling `../HYMT/` (ReTok) and `../ReTok/` folders — only the venv is shared.

Pipeline: **SFT → CPO (LoRA) → GRPO (full-param)**. Each stage is a self-contained
script; checkpoints flow between stages by path.

- **`program.md`** is the running experiment design/log. **`results.tsv`** is the
  append-only results table (one row per eval, with a keep/discard/BEST verdict).
  Read both before starting work — they hold the accumulated findings, and updating
  them after each experiment is part of the workflow.

## Directory layout

Code and artifacts are separated. **Run every command from this `Qwen/` root** — all
paths (in scripts, defaults, and the commands below) are relative to it.

```
Qwen/
├── train/        train.py  train_cpo.py  train_grpo.py  merge_lora_qwen3.py
├── eval/         eval_vllm.py  eval_multi.py  eval.py  eval_hymt7b.py
│                 eval_cpo_v3_plus_7b.sh  translate.py   (interactive demo)
├── data_build/   build_cpo_*.py  build_grpo_data.py  generate_candidates.py
│                 hymt7b_translate_*.py
├── data/         *.jsonl / *.json training + preference + GRPO data   (git-ignored)
├── checkpoints/  output_*/ model checkpoints                          (git-ignored)
├── models/       Qwen3-* base/instruct + HY-MT1.5-7B teacher/reference   (git-ignored)
├── datasets/     ALMA-*, X-ALMA-*, flores200_dataset, metricx_repo       (git-ignored)
├── logs/         *.txt run/eval logs                                  (git-ignored)
├── sherry_qat/   downstream low-bit quantization sub-experiment (separate; see its NOTES.md)
└── CLAUDE.md  program.md  results.tsv  DATA_MANIFEST.md  run_experiments.sh
```

Quick model demo: `$PY -u eval/translate.py --model_path ./checkpoints/output_1.7b_grpo_full`
(type sentences, zh↔en auto-detected).

## Setup

- **venv** (not on PATH; always invoke explicitly): `/home/tfbao/new/HY-MT/.venv/bin/python -u`.
  Has `torch`, `transformers`, `peft`, `trl`, `vllm`, `sacrebleu`, `unbabel-comet`, `datasets`.
- **GPU**: single RTX 4090 (24GB). All configs assume batch_size 1–2 + gradient
  accumulation + gradient checkpointing; vLLM runs colocated during GRPO at 0.2 mem fraction.
- **COMET** checkpoint is loaded from a hardcoded local path
  (`~/.cache/comet/models--Unbabel--wmt22-comet-da/.../model.ckpt`) in `train_grpo.py`,
  `eval_vllm.py`, `eval_multi.py` — never downloaded.
- Base models (`Qwen3-1.7B-Base` etc.), all `output_*/` checkpoint dirs, and all
  `*.jsonl`/`*.json` data files are git-ignored (see root `.gitignore`). Only code is committed.

## Common commands

```bash
PY=/home/tfbao/new/HY-MT/.venv/bin/python

# SFT: full fine-tune Qwen3-1.7B-Base on ChatML translation pairs (1 epoch — see below)
$PY -u train/train.py --model_path ./models/Qwen3-1.7B-Base --train_data ./data/alma_combined_sft_clean.jsonl \
  --output_dir ./checkpoints/output_1.7b_base_v2 --num_epochs 1 --lr 2e-5 --lr_scheduler inverse_sqrt \
  --batch_size 2 --gradient_accumulation_steps 8 --gradient_checkpointing

# CPO: LoRA preference training on top of the SFT model
$PY -u train/train_cpo.py --model_path ./checkpoints/output_1.7b_base_v2 --data_path ./data/cpo_v3_plus_7b.jsonl \
  --output_dir ./checkpoints/output_1.7b_cpo_v3_plus_7b --lr 1e-4 --beta 0.05 --nll_weight 1.0 \
  --batch_size 1 --gradient_accumulation_steps 16 --max_length 512

# Merge a LoRA adapter into its base before eval/GRPO
$PY -u train/merge_lora_qwen3.py --base_model_path ./checkpoints/output_1.7b_base_v2 \
  --adapter_model_path ./checkpoints/output_1.7b_cpo_v3_plus_7b \
  --output_path ./checkpoints/output_1.7b_cpo_v3_plus_7b_merged --save_dtype bf16

# GRPO: full-param RL with COMET reward (TRL GRPOTrainer + colocated vLLM)
$PY -u train/train_grpo.py --model_path ./checkpoints/output_1.7b_cpo_v3_plus_7b_merged \
  --data_path ./data/grpo_data.jsonl --output_dir ./checkpoints/output_1.7b_grpo_full --full_finetune

# Eval: fast vLLM eval on one test set (BLEU + COMET)
$PY -u eval/eval_vllm.py --model_path ./checkpoints/output_xxx --testset wmt23 --direction both

# Eval: multi-testset (WMT23/24 + Flores) to catch COMET reward-hacking
$PY -u eval/eval_multi.py --model_path ./checkpoints/output_xxx
```

`run_experiments.sh` holds shell functions for past experiment runs — useful as
templates for new runs. There is no Makefile and no test suite.

## Architecture

### Stage scripts

- **`train.py`** — SFT. Full fine-tune (or `--freeze_embedding`). `TranslationSFTDataset`
  builds ChatML `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>`,
  masks loss to the assistant turn. Uses `<|im_end|>` as eos (LLaMA-Factory `replace_eos`
  convention for Qwen3). Has inline WMT22 eval via subprocess at `--eval_steps`.
- **`train_cpo.py`** — CPO = DPO preference loss (no reference model) + NLL behavior
  cloning on chosen: `L = -log σ(β·(logπ(y_w) - logπ(y_l))) + λ·NLL(y_w)`. **LoRA only**
  (`r=16`, `all-linear`); `--full_finetune` exists but full-param CPO collapses the model —
  do not use it. Logs preference accuracy `acc` per step.
- **`train_grpo.py`** — GRPO via TRL `GRPOTrainer`. Reward = reference-based
  `wmt22-comet-da` COMET (weight 1.0) + a 4-gram `repetition_penalty` (weight 0.3).
  `loss_type="dapo"`, `num_generations=8`, colocated vLLM. Full-param works here (unlike CPO).
  Forces `eos_token_id` to `<|im_end|>` so generation stops at the turn boundary.
- **`merge_lora_qwen3.py`** — merges a PEFT adapter into its base; required before a LoRA
  checkpoint can be eval'd with vLLM or used as a GRPO starting point.

### Data-building scripts

`build_cpo_*.py`, `build_grpo_data.py`, `generate_candidates.py`, `score_data_comet.py`,
`hymt7b_translate_*.py`, `eval_hymt7b.py` — produce the preference/RL `.jsonl` files.
CPO data is `{"prompt", "chosen", "rejected"}` per line; GRPO data is
`{"prompt", "source", "reference"}` (reference feeds the COMET reward). The many
`build_cpo_*` variants are different chosen/rejected sourcing experiments — see
`program.md` for which produced which result.

### Conventions

- **Prompt templates** are inlined identically in every eval script:
  `Translate the following text from {Chinese|English} to {English|Chinese}.\n{src lang}: {src}\n{tgt lang}:`
  Changing them requires re-running SFT.
- **Chat format**: ChatML, `<|im_end|>` as eos everywhere (training + generation `stop`).
- **BLEU tokenization**: `tokenize="zh"` for en→zh, `"13a"` for zh→en.

### Eval discipline (important)

**WMT22 is contaminated** — ~17.5% leaked into the ALMA SFT data. Report final
numbers on **WMT23** (primary), and use `eval_multi.py` (WMT23/24 + Flores) to
confirm gains are real and not COMET reward-hacking, since WMT17–21 are CPO/GRPO
training sources. The COMET reward in GRPO is `wmt22-comet-da`; BLEU/chrF in
`eval_multi.py` are the independent lexical cross-checks.

### Current best lineage

`Qwen3-1.7B-Base` → SFT (`output_1.7b_base_v2`, ALMA 44K, 1 epoch) →
CPO v3+7B LoRA (`output_1.7b_cpo_v3_plus_7b_merged`) → full-param GRPO
(`output_1.7b_grpo_*`). See `program.md` / `results.tsv` for live numbers and the
keep/discard rationale of every experiment.

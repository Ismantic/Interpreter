# Translator: Qwen3-1.7B-Base → Chinese-English Translation Model

## Goal & Best Result

| Model | Size | WMT23 zh→en | WMT23 en→zh |
|-------|------|-------------|-------------|
| HY-MT1.5-7B | 7B | 0.8121 | 0.8679 |
| HY-MT1.5-1.8B | 1.8B | 0.8052 | 0.8669 |
| **Our best: GRPO-from-CPO** | **1.7B** | **0.8054** | **0.8542** |
| SFT baseline | 1.7B | 0.7863 | 0.8384 |

**Best model: `output_1.7b_grpo_full`** — SFT → CPO(v3+7B) → GRPO. WMT23 0.8054/0.8542.
zh→en matches HY-MT 1.8B (0.8052); en→zh gap −0.013.

Multi-testset (held-out, eval metric wmt22-comet-da):
| | wmt23 zh-en | wmt23 en-zh | wmt24 en-zh | flores zh-en | flores en-zh |
|--|--|--|--|--|--|
| GRPO-from-CPO | 0.8054 | 0.8542 | 0.8433 | 0.8649 | 0.8839 |

## RL Stage: GRPO (the breakthrough)

Full-parameter CPO failed 3× (catastrophic collapse — preference term degrades the
model's own decent outputs). Full-parameter **GRPO succeeded**: online sampling +
COMET reward + KL regularization gives a clean gradient direction. COMET/BLEU/chrF
all rose consistently on WMT23/24/Flores → genuine gain, not reward-hacking.

**Pipeline finding — can CPO be dropped?** Mostly yes:
- Standard SFT→GRPO (beta=0.04, 3K prompts) plateaus at 0.79/0.844 (overfits prompt set).
- Tuned SFT→GRPO (beta=0.02, 6K prompts) reaches 0.8003/0.8540 — matches SFT→CPO→GRPO
  on en-zh, beats it on flores zh-en, within 0.005–0.008 elsewhere.
- Lesson: a weak SFT start needs LESS KL anchoring (lower beta) and MORE prompt
  diversity. HY-MT skips CPO because its SFT is already strong (MT-pretrain + huge SFT).
- GRPO data: 3–4K prompts saturates a single round; field practice is 13–16K
  (MT-R1-Zero 13K, TAT-R1 16K). For more gain, do a 2nd GRPO round, not more data.

GRPO config: TRL GRPOTrainer, full-param, Adafactor, G=8, lr=1e-6, beta=0.04,
COMET reward + repetition penalty, vLLM colocate. See train_grpo.py.

## Setup

- **Working dir**: `/home/tfbao/Shiyu/Interpreter/Translator/`
- **Python**: `/home/tfbao/new/HY-MT/.venv/bin/python -u`
- **GPU**: Single NVIDIA RTX 4090 (24GB)
- **Base model**: `./Qwen3-1.7B-Base`
- **SFT model**: `./output_1.7b_base_v2` (ALMA 44K, 1 epoch, best SFT)
- **Best CPO model**: `./output_1.7b_cpo_v3_merged`

## Training Pipeline

### Stage 1: SFT (DONE)
- Data: `alma_combined_sft_clean.jsonl` — ALMA Human Parallel + X-ALMA = 44K pairs (WMT22/23 decontaminated)
- Config: lr=2e-5, inverse_sqrt, warmup=0.01, 1 epoch, ChatML format, `<|im_end|>` as eos
- Result: zh→en 0.7863, en→zh 0.8384

### Stage 2: CPO (iterating)
- Proper CPO = DPO preference loss + NLL behavior cloning on chosen
- LoRA rank=16, all-linear, lr=1e-4, beta=0.1, nll_weight=1.0
- Key finding: **rejected must be on-policy (our SFT model's output)**

## CPO Experiment Plan

### Completed

| Exp | Chosen source | Rejected source | Data size | zh→en | en→zh | Avg |
|-----|--------------|----------------|-----------|-------|-------|-----|
| CPO v3 | our model best (COMET) | our model worst | 44K (WMT17-21) | **0.8005** | 0.8463 | **0.8234** |
| Exp B | ALMA-R GPT-4/ref best | our greedy | 6K (ALMA-R) | 0.7918 | **0.8515** | 0.8217 |
| Exp C | ALMA-R+X-ALMA chosen | our greedy | 37K | 0.7925 | 0.8492 | 0.8209 |

### Experiments Run (10h autonomous session)

| Exp | Data | Result (zh→en/en→zh) | Status |
|-----|------|----------------------|--------|
| HY-MT 7B eval | WMT23 baseline | 0.8121/0.8679 | target confirmed (higher than 1.8B) |
| SFT 2epoch | ALMA 44K × 2 | 0.2851/0.2037 | discard: 2 epoch overfits catastrophically |
| Full CPO + Exp B | GPT-4 chosen 6K full param | 0.7658/0.5921 | discard: full param CPO collapses |
| CPO + 7B chosen | HY-MT 6K LoRA | 0.7422/0.7495 | discard: 7B-only chosen, style mismatch |
| **CPO v3 + 7B** | v3 44K, 7B replaces 25.7% chosen | **0.8017/0.8507** | **BEST — new best model** |
| CPO v3+7B+GPT4 | v3+7B 44K + GPT-4 6K = 50K | 0.7966/0.8478 | discard: extra 6K dilutes signal |
| SFT+CQIA | 80/20 by samples, max_len 512 | 0.6852/0.6171 | discard: token imbalance |
| SFT+CQIA v2 | 80/20 by tokens, gen<256 | 0.7612/0.7139 | discard: general data still hurts |
| SFT+OASST v2 | 80/20 by tokens, gen<256 | 0.7724/0.6321 | discard: en→zh collapsed |

### Key Insights from this Session

1. **2 epoch SFT overfits catastrophically** (COMET 0.28/0.20). 1 epoch is optimal.
2. **Full-param CPO collapses** even on small clean data (en→zh 0.85→0.59). LoRA only.
3. **HY-MT 7B as sole chosen breaks the model** — its longer/different-style output makes the BC term unlearnable. But **7B mixed into our self-gen pool works**: replacing only the 25.7% of chosen where 7B beats our best (COMET vs WMT ref) gave the new best model. The familiar 74% keeps training grounded.
4. **General instruction data hurts a dedicated translator.** Both CQIA (Chinese knowledge QA) failed. Even fixing the token-imbalance (by-sample 80/20 → by-token 38/62 because CQIA avg 540 tokens vs translation 78) only recovered to 0.76/0.71 — still −0.025/−0.125 vs pure SFT. TranslateGemma/HY-MT add general data to *preserve* general capability at a translation cost; for a pure translator that cost is not worth paying.
5. **More CPO data ≠ better** (reconfirmed): v3+7B 44K beats v3+7B+GPT4 50K.

## Key Findings (cumulative)

1. **SFT**: ALMA 44K human parallel data, 1 epoch, is optimal. More data (distillation) or more epochs hurts.
2. **CPO loss**: Must include NLL behavior cloning. DPO ref_model=None breaks model.
3. **Rejected must be on-policy**: Our model's output as rejected > ALMA/X-ALMA's rejected.
4. **Chosen from stronger models helps**: GPT-4 chosen (Exp B) helped en→zh; HY-MT 7B chosen (mixed into pool) helped both.
5. **More CPO data ≠ better**: 6K (Exp B) > 37K (Exp C); v3+7B 44K > v3+7B+GPT4 50K. Quality > quantity.
6. **LoRA > full finetune for CPO**: full-param CPO collapses; LoRA's regularization is essential.
7. **Data leakage**: SFT data leaked 17.5% into WMT22. Use WMT23 for eval.
8. **Stronger-model chosen must be MIXED, not wholesale**: replace only the fraction where the stronger model genuinely beats our best by COMET (~25%). Wholesale replacement makes the BC term unlearnable.
9. **General instruction data hurts a dedicated translator** — do not mix it in. Confirmed on CQIA (Chinese QA), even token-balanced.

## Best Model

`output_1.7b_cpo_v3_plus_7b_merged` — CPO v3 with HY-MT 7B augmented candidate pool.
WMT23: zh→en **0.8017**, en→zh **0.8507**. Gap to HY-MT 1.8B: −0.0035 / −0.016.

## Data Files (preserve for reuse across experiments)

| File | Content | Used by |
|------|---------|---------|
| alma_combined_sft_clean.jsonl | SFT data 36.8K clean | SFT |
| cpo_preference.jsonl | Self-gen 44K pref (WMT17-21) | CPO v3 |
| cpo_gpt4_vs_ours.jsonl | ALMA-R GPT4 chosen + our greedy 6K | Exp B |
| cpo_v3_plus_7b.jsonl | v3 44K, 7B chosen replaces 25.7% | CPO v3+7B (BEST) |
| hymt7b_cpov3_translations.json | HY-MT 7B translations of WMT17-21 sources | build CPO v3+7B |
| hymt7b_almar_translations.json | HY-MT 7B translations of ALMA-R sources | 7B-chosen experiments |
| alma_r_preference.jsonl | ALMA-R original preference 4.7K | Reference |
| xalma_preference.jsonl | X-ALMA original preference 30K | Reference |

## Reproducing the Best Model

```bash
PY=/home/tfbao/new/HY-MT/.venv/bin/python
# 1. HY-MT 7B translates WMT17-21 sources
$PY hymt7b_translate_cpo_sources.py
# 2. Augment CPO v3 pool: replace chosen with 7B where COMET-better
$PY build_cpo_v3_plus_7b.py
# 3. Train LoRA CPO
$PY train_cpo.py --model_path ./output_1.7b_base_v2 \
  --data_path ./cpo_v3_plus_7b.jsonl --output_dir ./output_1.7b_cpo_v3_plus_7b \
  --lr 1e-4 --beta 0.05 --nll_weight 1.0 --batch_size 1 \
  --gradient_accumulation_steps 16 --max_length 512
# 4. Merge + eval
$PY merge_lora_qwen3.py --base_model_path ./output_1.7b_base_v2 \
  --adapter_model_path ./output_1.7b_cpo_v3_plus_7b \
  --output_path ./output_1.7b_cpo_v3_plus_7b_merged --save_dtype bf16
$PY eval_vllm.py --model_path ./output_1.7b_cpo_v3_plus_7b_merged --testset wmt23 --direction both
```

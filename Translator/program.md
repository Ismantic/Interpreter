# Translator: Qwen3-1.7B-Base → Chinese-English Translation Model

## Goal

| Model | Size | WMT23 zh→en COMET | WMT23 en→zh COMET |
|-------|------|-------------------|-------------------|
| HY-MT1.5-1.8B | 1.8B | 0.8052 | 0.8669 |
| **Our best (CPO v3)** | **1.7B** | **0.8005** | **0.8463** |
| **Our best (Exp B)** | **1.7B** | **0.7918** | **0.8515** |

Gap to HY-MT: zh→en -0.005 (CPO v3), en→zh -0.015 (Exp B)

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

### Running

**Exp D**: CPO v3 flow on ALMA-R sources with GPT-4 in candidate pool
- Sources: ALMA-R 3065 sentences
- Our SFT model generates 5 candidates (1 greedy + 4 sampling)
- Pool: our 5 candidates + GPT-4 translation + ALMA-13B translation + reference
- COMET scores all candidates in pool
- chosen = pool highest, rejected = pool lowest
- Expected: ~6K pairs with higher-quality chosen (GPT-4 level)
- Status: generating candidates

### TODO

**Exp E**: Add Codex translations to candidate pool
- Same as Exp D but add Codex-generated translations to the pool
- Codex translates all 3065 × 2 ALMA-R source sentences
- Pool: our 5 + GPT-4 + ALMA + ref + **Codex**
- COMET selects best/worst from expanded pool
- **Reuse all data from Exp D, only generate Codex candidates incrementally**

**Exp F**: Add Claude translations to candidate pool
- Same as Exp E but add Claude-generated translations
- Pool: our 5 + GPT-4 + ALMA + ref + Codex + **Claude**
- Design high-quality translation prompt for Claude
- **Reuse all data from Exp E, only generate Claude candidates incrementally**

## Key Findings

1. **SFT**: ALMA 44K human parallel data is optimal. More data (distillation) hurts zh→en.
2. **CPO loss**: Must include NLL behavior cloning. DPO ref_model=None breaks model.
3. **Rejected must be on-policy**: Our model's output as rejected > ALMA/X-ALMA's rejected.
4. **Chosen from stronger models helps en→zh**: GPT-4 chosen (Exp B) gave en→zh 0.8515.
5. **More CPO data ≠ better**: 6K (Exp B) > 37K (Exp C). Quality > quantity.
6. **LoRA > full finetune for CPO**: LoRA's regularization effect helps.
7. **Data leakage**: SFT data leaked 17.5% into WMT22. Use WMT23 for eval.

## Data Files (preserve for reuse across experiments)

| File | Content | Used by |
|------|---------|---------|
| alma_combined_sft_clean.jsonl | SFT data 36.8K clean | SFT |
| cpo_preference.jsonl | Self-gen 44K pref (WMT17-21) | CPO v3 |
| cpo_gpt4_vs_ours.jsonl | ALMA-R GPT4 chosen + our greedy 6K | Exp B |
| cpo_exp_c.jsonl | ALMA-R+X-ALMA chosen + our greedy 37K | Exp C |
| cpo_exp_d_candidates.jsonl | ALMA-R 5-cand + GPT4/ALMA/ref pool | Exp D (building) |
| alma_r_preference.jsonl | ALMA-R original preference 4.7K | Reference |
| xalma_preference.jsonl | X-ALMA original preference 30K | Reference |

## NEVER STOP

Run experiments autonomously. Each CPO takes ~2 hours. Eval ~10 minutes. Iterate rapidly.
Always reuse previously generated data. Never regenerate what already exists.

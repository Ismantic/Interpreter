# Translator: Qwen3-0.6B → Chinese-English Translation Model

Train Qwen3-0.6B into a high-quality zh↔en translation model, targeting HY-MT1.5-1.8B level quality.

## Goal

| Model | Size | zh→en COMET | en→zh COMET |
|-------|------|-------------|-------------|
| HY-MT1.5-1.8B | 1.8B | 0.8182 | 0.8745 |
| ALMA-13B-R | 13B | TBD | TBD |
| TranslateGemma-4B | 4B | TBD | TBD |
| **Our target** | **0.6B** | **0.80+** | **0.85+** |
| Qwen3-0.6B (base) | 0.6B | TBD | ~0.81* |

## TODO
- [ ] Compute ALMA-13B-R WMT22 COMET from saved outputs in `/home/tfbao/new/ALMA/outputs/`
- [ ] Find TranslateGemma WMT22 COMET scores
- [ ] Try Aurora optimizer
- [ ] Try inverse_sqrt scheduler (ALMA default)
- [ ] Increase batch size (GPU has 11GB headroom)
- [ ] Stage 2: CPO with COMET reward

## Why Qwen3-0.6B

- Already bilingual (Chinese + English), no tokenizer issues
- 0.6B params → fast training on single 4090
- Strong base model quality from Qwen3 family
- Skip ALMA's Stage 1 (monolingual pretraining) — Qwen3 already has language understanding

## Setup

- **Working dir**: `/home/tfbao/Shiyu/Interpreter/Translator/`
- **Python**: `/home/tfbao/new/HY-MT/.venv/bin/python -u`
- **GPU**: Single NVIDIA RTX 4090 (24GB)
- **Base model**: `./Qwen3-0.6B` (downloaded via ModelScope)

## Training Pipeline

### Stage 1: Translation SFT
- **Data**: `/home/tfbao/Shiyu/Interpreter/private/sft_distill_ft.jsonl` (186K pairs)
  - 117K zh→en (HY-MT distillation + FineTranslations)
  - 69K en→zh (HY-MT distillation)
- **Method**: Full fine-tuning, mask_prompt (only compute loss on translation output)
- **Hyperparams** (reference TranslateGemma): lr=1e-4 (AdaFactor) or 1e-5 (AdamW), cosine schedule
- **Prompt format**: Same as HY-MT
  - zh→en: `将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：\n\n{src}`
  - en→zh: `Translate the following segment into Chinese, without additional explanation.\n\n{src}`
- **Eval**: WMT22 full, both directions, COMET + BLEU

### Stage 2: CPO/DPO Preference Optimization
- **Goal**: Directly optimize COMET score
- **Method**: Offline preference learning (ALMA-R style)
  1. Generate N candidates per source with Stage 1 model (temperature sampling)
  2. Score all candidates with COMET
  3. Build preference pairs (best vs worst)
  4. Train with CPO loss
- **Data**: WMT test sets or distillation sources as prompt pool
- **Reference**: ALMA-R CPO code in `/home/tfbao/new/ALMA/run_cpo_llmmt.py`

### Stage 3 (Optional): Translation-oriented CPT
- Continue pretraining with large-scale parallel translation data (HY-MT style)
- Data: WikiMatrix + MultiUN + FineTranslations + distillation pairs as bilingual parallel text
- Format: pack parallel pairs into sequences for CLM (e.g. `{zh}\n{en}` or chat template)
- Goal: teach the model deeper bilingual alignment before SFT
- Only if Stage 1+2 quality is insufficient

## Key Design Decisions

1. **Full fine-tune, not LoRA**: 0.6B model is small enough for full fine-tuning on 4090
2. **mask_prompt**: Only compute loss on assistant response (translation output)
3. **No system message**: HY-MT default system prompt is empty
4. **Evaluation**: Use same eval.py from Interpreter project, adapted for Qwen3 tokenizer

## Available Data

| Source | Pairs | Direction | Quality |
|--------|-------|-----------|---------|
| HY-MT distillation (SkyPile) | 25K | zh→en | High (HY-MT style) |
| HY-MT distillation (Fineweb) | 25K | zh→en | High |
| HY-MT distillation (misc) | 37K | zh→en | High |
| HY-MT distillation (Fineweb-edu) | 50K | en→zh | High |
| HY-MT distillation (misc) | 19K | en→zh | High |
| FineTranslations (Gemma3, filtered) | 40K | zh→en | Good |
| SFT data (X-ALMA style) | 44K | both | Good |
| **Total** | **~270K** | | |

## Experiment Loop

Same as Interpreter/program.md:
1. Modify training config
2. Run training
3. Evaluate on WMT22
4. Record in results.tsv
5. Keep or discard

## NEVER STOP

Run experiments autonomously. 0.6B model trains fast (~10-20 min per epoch on 186K data). Iterate rapidly.

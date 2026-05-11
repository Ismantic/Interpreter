# Translator: Qwen3-Base → Chinese-English Translation Model

Train Qwen3-Base (0.6B/1.7B) into a high-quality zh↔en translation model. Following ALMA → ALMA-R → X-ALMA → TranslateGemma → HY-MT methodology progression.

## Goal

| Model | Size | zh→en COMET | en→zh COMET |
|-------|------|-------------|-------------|
| HY-MT1.5-1.8B | 1.8B | 0.8182 | 0.8745 |
| ALMA-13B-R | 13B | TBD | TBD |
| **Our target** | **0.6B/1.7B** | **0.80+** | **0.85+** |

## Setup

- **Working dir**: `/home/tfbao/Shiyu/Interpreter/Translator/`
- **Python**: `/home/tfbao/new/HY-MT/.venv/bin/python -u`
- **GPU**: Single NVIDIA RTX 4090 (24GB)
- **Base models**: `./Qwen3-0.6B-Base`, `./Qwen3-1.7B-Base` (downloaded via ModelScope)
- **IMPORTANT**: Always use **Base** models, not instruct models. ALMA trains from base.

## Experiment Plan (methodology progression)

### Phase A: ALMA-style SFT (current)
1. ✅ v4: Qwen3-0.6B (instruct) + X-ALMA 14K → zh-en 0.7649, en-zh 0.8175
2. ✅ v6: Qwen3-0.6B (instruct) + ALMA Human + X-ALMA 44K → zh-en 0.7645, en-zh 0.8187
3. ✅ base_v1: Qwen3-0.6B-Base + ALMA Human + X-ALMA 44K → eval running
4. 🔄 1.7b_base_v1: Qwen3-1.7B-Base + same data → training
5. [ ] Compare base vs instruct results

### Phase B: ALMA-R CPO
6. [ ] Generate candidates from best SFT model (temperature sampling)
7. [ ] Score with COMET, build preference pairs
8. [ ] CPO training (ALMA: lr=1e-4, LoRA, beta=0.1)

### Phase C: TranslateGemma techniques
9. [ ] Freeze Embedding SFT (TranslateGemma validated this approach)
10. [ ] Synthetic data: use HY-MT 1.8B as teacher to translate large-scale monolingual data
11. [ ] COMET-QE data filtering (select source sentences that benefit most from translation)

### Phase D: HY-MT techniques
12. [ ] Translation-oriented CPT with large-scale parallel data (WikiMatrix, MultiUN, etc.)
13. [ ] HY-MT config SFT (lr=1e-5, cosine_with_min_lr, mask_prompt)
14. [ ] Teacher distillation from HY-MT 1.8B (the model IS distilled from 7B)
15. [ ] RegMix-style data ratio optimization

## ALMA-aligned Training Config

```bash
python train.py \
    --model_path ./Qwen3-0.6B-Base \
    --train_data ./alma_combined_sft.jsonl \
    --output_dir ./output_base_vXX \
    --max_seq_length 512 \
    --batch_size 4 --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --num_epochs 1 \
    --lr 2e-5 --min_lr 1e-6 --lr_scheduler inverse_sqrt \
    --warmup_ratio 0.01 --weight_decay 0.01 \
    --logging_steps 100 --save_steps 1000
```

No inline eval. Eval after training:
```bash
python eval.py --model_path ./output_base_vXX --testset wmt22 --direction both --batch_size 32
```

## Current Results

| Experiment | Model | Data | zh→en COMET | en→zh COMET |
|------------|-------|------|-------------|-------------|
| v4 | 0.6B instruct | X-ALMA 14K | 0.7649 | 0.8175 |
| v6 | 0.6B instruct | ALMA+X-ALMA 44K | 0.7645 | 0.8187 |
| base_v1 | 0.6B-Base | ALMA+X-ALMA 44K | running | running |
| 1.7b_base_v1 | 1.7B-Base | ALMA+X-ALMA 44K | training | training |

## Key Findings

1. **X-ALMA 14K data is sufficient for decent SFT** — 44K (+ ALMA Human) barely improves over 14K
2. **186K distillation data causes COMET degradation** after 500 steps (v1 experiment)
3. **Instruct vs Base**: must use Base model for SFT (ALMA does this)
4. **inverse_sqrt scheduler** is ALMA default, not cosine
5. **1 epoch is enough** for SFT (ALMA and TranslateGemma both use 1 epoch)
6. **Training is fast**: 0.6B model + 44K data = 14 minutes on 4090

## Available Data

| File | Pairs | Description |
|------|-------|-------------|
| alma_combined_sft.jsonl | 44,624 | ALMA Human + X-ALMA bidirectional (current) |
| xalma_sft.jsonl | 13,812 | X-ALMA only bidirectional |
| ../private/sft_distill_ft.jsonl | 186,323 | HY-MT distillation + FineTranslations |
| ../private/sft_comet90.jsonl | 42,363 | SFT data COMET≥0.90 |

## ALMA Research Notes (from papers)

**ALMA (1st gen) — ICLR 2024:**
- Stage 1: 20B tokens OSCAR monolingual, 600K steps, lr=2e-5 cosine
- Stage 2: 15,406 zh-en parallel pairs (WMT'17-20 + Flores-200), 2 epochs, lr=2e-5 inverse_sqrt
- Prompt: `Translate this from Chinese to English:\nChinese: {src}\nEnglish:`
- Loss: only on target (prompt masked with -100)
- Best zh→en COMET-22: 0.8021 (13B-LoRA), en→zh: 0.8596
- Qwen3 skips Stage 1 (already bilingual)

**ALMA-R (2nd gen) — ICML 2024:**
- CPO on ALMA-13B-LoRA, scorer: kiwi_xcomet (KIWI-XXL + XCOMET ensemble)
- 2K preference triplets per direction from FLORES-200 (reference/GPT-4/ALMA candidates)
- Hyperparams: lr=1e-4 LoRA, beta=0.1, 1 epoch, inverse_sqrt
- CPO = DPO without reference model (memory efficient)
- Best zh→en: 0.8095 (+0.007), en→zh: 0.8685 (+0.009)

**X-ALMA (3rd gen) — ICLR 2025:**
- 5-step recipe: mono CPT base → mono CPT LS modules → pseudo-mono → SFT → ARPO
- zh in Group 6 (Eurasian Mix: et,fi,ja,ka,ko,zh)
- SFT data: ~7K pairs from Flores+NTREX+WMT test sets
- ARPO: adaptive tau prevents over-rejection of dis-preferred style
- Best zh→en: 0.824, en→zh: 0.875

**TranslateGemma:**
- Freeze embedding during SFT
- AdaFactor lr=0.0001, 200K steps
- 30% general instruction data mixed in
- RL with MetricX-QE + AutoMQM + ChrF + naturalness autorater

**HY-MT:**
- MT-oriented CPT + SFT + RL + ensemble RL
- 1.8B model distilled from 7B
- lr=1e-5, cosine_with_min_lr, mask_prompt
- RegMix for data ratio optimization

## NEVER STOP

Run experiments autonomously. Each takes 15-30 minutes. Iterate rapidly.

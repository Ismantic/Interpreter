# Interpreter AutoResearch

Autonomous experiment loop for HY-MT1.5-1.8B tokenizer replacement.

## Goal

Replace HY-MT1.5-1.8B tokenizer (120,818 → 65,007 vocab) and recover translation quality. Target: exceed init model on **both** zh→en and en→zh COMET (wmt22-comet-da, WMT22 full).

## Baselines

| Model | zh→en COMET | en→zh COMET | BPB |
|-------|-------------|-------------|-----|
| Original HY-MT | 0.8182 | 0.8745 | 1.169 |
| init (untrained) | 0.8134 | 0.8571 | 1.570 |

## Two-Phase Training Strategy

### Phase 1: Freeze Transformer, Train Embedding
- **Goal**: Minimize val_loss / BPB. COMET will degrade — that's expected and OK.
- **Rationale**: Embedding needs to learn new token representations. Frozen transformer can't adapt, so COMET drops with more training. But BPB must improve for Phase 2 to work.
- **Current best for Phase 2 input**: v21-200 (val_loss=3.43, en→zh COMET=0.8580)
- **WARNING**: Training beyond step 200 irreversibly damages COMET. Phase 2 cannot recover it. v21-500 (val_loss=3.01) COMET dropped to 0.8370 and Phase 2 could not bring it back.
- **Config**: `--freeze_transformer`, no mask_prompt, lr=2e-5 cosine(min=1e-5), 500 steps

### Phase 2: Freeze Embedding, Train Transformer
- **Goal**: Recover and improve COMET using high-quality translation data.
- **Rationale**: TranslateGemma (Google) validated this approach — freeze embedding during SFT helps translation quality. Transformer adapts to new embeddings.
- **Key insight**: Data quality is critical. Same data that trains Phase 1 will NOT improve COMET in Phase 2. Need COMET-aligned data.
- **Config**: `--freeze_embedding --mask_prompt`, lr=1e-5, cosine(min=1e-6)
- **DO NOT use gradient_checkpointing** (conflicts with frozen embedding, gradients become None)

## Setup

1. **Working directory**: `/home/tfbao/Shiyu/Interpreter/`
2. **Python**: `/home/tfbao/new/HY-MT/.venv/bin/python -u`
3. **GPU**: Single NVIDIA RTX 4090 (24GB)
4. **Phase 1 output**: `./private/phase1_v21` (val_loss=3.01)
5. **Results log**: `results.tsv` in this directory

## Phase 2 Data Strategy (CRITICAL)

The #1 priority for Phase 2 is **data that aligns with COMET preference**. Three approaches to try:

### 1. Self-distillation data (HY-MT translated)
- Available: en→zh 100K pairs, zh→en 50K pairs in `./private/distill_*.txt`
- Style matches original HY-MT → should align well with COMET
- Need to pretokenize into .pt or use as SFT JSONL

### 2. FineTranslations (Gemma3 translated, filtered)
- Available: 40K filtered pairs in `./private/ft_filtered_zh.txt` + `ft_filtered_en.txt`
- zh→en direction only
- edu>=2, quality>=0.05, deduped

### 3. COMET-scored filtering
- Use `score_data_comet.py` to score existing training data
- Keep only high-COMET pairs (>0.85)
- Most direct but requires GPU time for scoring

### 4. SFT data
- `./private/sft_8k.jsonl` (8K pairs) — currently testing in Phase 2 v4
- `./private/sft_filtered.jsonl` (44K pairs)

## Phase 2 Experiment Template

```bash
/home/tfbao/new/HY-MT/.venv/bin/python -u finetune_muon.py \
    --model_path ./private/phase1_v21 \
    --train_data DATA_FILE \
    --mode sft \
    --output_dir ./private/phase2_vXX \
    --freeze_embedding --mask_prompt \
    --max_seq_length 512 \
    --batch_size 8 --gradient_accumulation_steps 8 \
    --max_steps 200 --warmup_steps 10 \
    --muon_lr 1e-5 --adam_lr 1e-5 \
    --lr_scheduler cosine_with_min_lr --min_lr 1e-6 \
    --logging_steps 10 --save_steps 50 --eval_steps 50
```

Tokenizer files auto-copied. Eval runs en-zh COMET inline.

## Decision Rules

**KEEP when:**
- en→zh COMET improves over previous best (currently 0.8370 from Phase 1)
- val_loss decreasing or stable

**DISCARD when:**
- en→zh COMET worse than previous Phase 2 best
- val_loss exploding

**When to change data:**
- If 3 consecutive experiments with same data don't improve COMET → switch data source
- If COMET drops with more training → data style doesn't align, try different data

## Warning Signals

1. **val_loss drops but COMET drops too**: data style doesn't match COMET preference → change data
2. **val_loss flat, COMET flat**: lr too small or data exhausted → increase lr or add data
3. **val_loss drops fast, COMET drops fast**: lr too large → reduce lr

## Key Findings

1. **Phase 1 ceiling**: COMET peaks at step 100-150 then degrades. BPB keeps improving. This is inherent to frozen transformer — can't be fixed with lr/schedule/regularization.
2. **Phase 2 (freeze embed) COMET-stable**: With lr=1e-5, COMET stays at 0.859x for 500 steps. Transformer adapts without damaging COMET.
3. **Phase 2 (freeze embed) lr=5e-5 too high**: COMET drops from 0.8370→0.8338 in 100 steps.
4. **Data quality is the bottleneck**: Same pretokenized data (.pt) doesn't improve COMET in Phase 2. Need COMET-aligned data (distillation or COMET-filtered).
5. **TranslateGemma validates our approach**: Google freezes embedding during SFT, uses MetricX-QE filtered synthetic data, AdaFactor lr=0.0001, 200K steps.
6. **No mask_prompt in Phase 1**: slightly delays COMET degradation, val_loss learns input+output roles.
7. **BPB gap**: original=1.169, init=1.570, Phase 1 best≈1.49. Only 20% recovered.

## Available Data Files

| File | Size | Description |
|------|------|-------------|
| phase1_v18_ft.pt | 257M tok | Phase 1 training data (mixed sources) |
| sft_8k.jsonl | 8K pairs | Translation SFT data |
| sft_filtered.jsonl | 44K pairs | Larger SFT data |
| distill_chat.pt | 6.7M tok | Pure distillation (small) |
| distill_mix_chat.pt | 10M tok | Distillation + Wiki mix |
| distill_fwedu_en/zh.txt | 50K pairs | Fineweb-edu en→zh distillation |
| distill_skypile_en/zh.txt | 25K pairs | SkyPile zh→en distillation |
| ft_filtered_zh/en.txt | 40K pairs | FineTranslations filtered |

## NEVER STOP

Once the loop begins, do NOT pause to ask the human. Run experiments autonomously until manually stopped. Each Phase 2 experiment takes ~30 minutes (training fast, eval ~10min per checkpoint). Iterate rapidly.

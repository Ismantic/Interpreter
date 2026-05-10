# Interpreter AutoResearch

Autonomous experiment loop for HY-MT1.5-1.8B tokenizer replacement (Phase 1: embedding training).

## Goal

Replace HY-MT1.5-1.8B tokenizer (120,818 → 65,007 vocab) and recover translation quality via embedding-only training. Target: exceed init model on **both** zh→en and en→zh COMET (wmt22-comet-da, WMT22 full).

## Baselines

| Model | zh→en BLEU | zh→en COMET | en→zh BLEU | en→zh COMET |
|-------|-----------|-------------|-----------|-------------|
| Original HY-MT | 17.78 | 0.8182 | 32.39 | 0.8745 |
| init (untrained) | 19.56 | 0.8134 | 27.88 | 0.8571 |

**Minimum target**: both COMET > init. **Stretch target**: approach original HY-MT.

## Setup

1. **Working directory**: `/home/tfbao/Shiyu/Interpreter/`
2. **Python**: `/home/tfbao/new/HY-MT/.venv/bin/python -u`
3. **GPU**: Single NVIDIA RTX 4090 (24GB)
4. **Init model**: `./HY-MT1.5-1.8B-new-tok` (tokenizer replaced, embeddings copy-averaged)
5. **Training data**: Pre-tokenized `.pt` files in `./private/`
6. **Results log**: `results.tsv` in this directory

## What you CAN modify

- `finetune_muon.py` — training script. Everything is fair game: lr, schedule, regularization, optimizer config, embedding handling, loss computation.
- Training arguments (lr, steps, schedule, batch size, etc.)
- Data file selection (choose from available `.pt` files in `./private/`)

## What you CANNOT modify

- `eval.py` — evaluation script. Read-only.
- `tokenizer_wrapper.py` — tokenizer. Read-only.
- Model architecture (frozen transformer, only train embedding layer).
- Evaluation protocol: WMT22 full, wmt22-comet-da, both directions.

## Experiment constraints

- **Time budget**: ~100-160 minutes per training run (500 steps, 12-19s/step depending on GPU load).
- **Eval time**: ~10 minutes per direction (both = 20 minutes). Use `--eval_steps` to embed eval into training loop.
- **Total per experiment**: ~2-3 hours (train + eval).
- **Save checkpoints every 100 steps** with `--save_steps 100`. Use `--eval_steps 100` for inline eval.

## The experiment loop

LOOP FOREVER:

1. Review `results.tsv` and previous findings.
2. Form a hypothesis and modify `finetune_muon.py` or training args.
3. Run training with inline eval:
   ```bash
   /home/tfbao/new/HY-MT/.venv/bin/python -u finetune_muon.py \
       --model_path ./HY-MT1.5-1.8B-new-tok \
       --train_data ./private/phase1_v18_ft.pt \
       --output_dir ./private/phase1_vXX \
       --freeze_transformer --gradient_checkpointing --mask_prompt \
       --max_seq_length 384 --batch_size 32 --gradient_accumulation_steps 8 \
       --max_steps 500 --warmup_steps 5 \
       --adam_lr 2e-5 \
       --lr_scheduler cosine_with_min_lr --min_lr 1e-5 \
       --logging_steps 10 --save_steps 100 --eval_steps 100
   ```
   Tokenizer files are auto-copied to checkpoints. Eval runs inline at each save_steps.
4. Read the output for `[eval]` lines showing BLEU/COMET and `[diag]` lines showing diagnostics.
5. Record results in `results.tsv`.
6. Apply decision rules (see below).

## Logging results

`results.tsv` (tab-separated) columns:

```
version	zh_en_bleu	zh_en_comet	en_zh_bleu	en_zh_comet	embed_norm_ratio	final_loss	best_step	status	description
```

- **version**: experiment name (e.g. v20-500, v20-200)
- **zh_en_bleu / zh_en_comet**: WMT22 full zh→en scores
- **en_zh_bleu / en_zh_comet**: WMT22 full en→zh scores
- **embed_norm_ratio**: `||embed_current|| / ||embed_init||` (healthy range: 1.00-1.20)
- **final_loss**: training loss at final step
- **best_step**: which checkpoint had best composite COMET
- **status**: `keep`, `discard`, `baseline`, or `crash`
- **description**: short text of what this experiment tried

## Decision rules

**KEEP (set as new baseline) when ALL conditions met:**
- Both zh→en and en→zh COMET exceed init (0.8134 and 0.8571)
- Composite COMET (average of both) is higher than current best by > 0.001
- embed_norm_ratio in [1.00, 1.25]

**DISCARD when ANY condition met:**
- Either direction COMET below init (broken)
- Composite COMET didn't improve or improved < 0.001 (noise)
- embed_norm_ratio > 1.30 (overtraining)

**TRADEOFF (one direction up, one down):**
- Record the result with detailed analysis
- Write explicit hypothesis for next experiment: "can we preserve X while improving Y by ..."
- If 3 consecutive tradeoffs, shift strategy (change data mix, try untie embedding, etc.)

## Warning signals (act immediately when observed)

**Warning 1: BLEU rising but COMET falling**
- Already observed in v18-500 (BLEU 31.85, COMET 0.8533)
- Cause: model learns surface lexical patterns at expense of semantic quality
- Action: find the optimal checkpoint (step where COMET peaked), reduce max_steps or lower LR

**Warning 2: One direction improves, other degrades**
- Single-direction optimization is a local optimum, not global
- Action: adjust data mix (zh→en vs en→zh balance) or try per-direction LR multiplier

**Warning 3: embed_norm_ratio > 1.25**
- Embeddings have drifted too far from init, further training won't help
- Action: early stop or increase regularization

**Warning 4: Loss plateau but COMET still changing**
- COMET can move even when loss is flat — subtle embedding shifts matter
- Action: check intermediate checkpoints to find true optimum

## Key findings so far

1. **mask_prompt is essential**: only compute loss on assistant response tokens.
2. **lr=2e-5 with linear decay peaks at ~step 150 then en→zh COMET degrades** (BLEU keeps improving).
3. **lr=1e-5 preserves en→zh COMET (0.8570) but BLEU is low** (20.40).
4. **Embedding regularization (L2 toward init) doesn't prevent COMET degradation** — same COMET drop regardless of reg strength. Constrains drift magnitude but not direction.
5. **COMET peak corresponds to lr ≈ 1.3-1.6e-5** in the linear decay schedule.
6. **HY-MT official config**: lr=1e-5, cosine_with_min_lr (min_lr=1e-6), warmup_ratio=0.01.
7. **Tied embeddings** (embed_tokens = lm_head) means lm_head gradients directly alter input representations. Untying and using different LRs is a promising direction (see autoresearch/train.py for reference: embed_lr=0.6, unembed_lr=0.004).
8. **Data is balanced**: ~48% zh→en, ~51% en→zh chat pairs.
9. **One-to-one mapping rates equal** for zh/en tokens (~74%). Multi-to-one quality also equal (avg 2.1 old tokens per new).
10. **COMET stabilizes at lr < 5e-6**: in v19 steps 300-500, COMET stops dropping (0.8528-0.8536) as lr approaches 0.
11. **v19 ckpt-150 is current best en→zh** (COMET 0.8587, +0.0016 over init).

## Ideas to try (prioritized)

1. **cosine_with_min_lr** (lr=2e-5 → min_lr=1e-5) — keep lr in productive range ← RUNNING NOW (v20)
2. **Untie embed_tokens and lm_head**, use different lr for each
3. **Constant lr=1.5e-5** (no decay) + 500 steps — simplest test of "lr in optimal range"
4. **Warmdown only** (constant lr for 80%, linear decay last 20%)
5. **EMA** of embeddings — smooth out training noise
6. **Larger data** (1B tokens) + more steps with optimal lr/schedule
7. **Different embed/unembed lr** without untying — use gradient hooks to scale

## Available data files

| File | Tokens | Description |
|------|--------|-------------|
| phase1_v18_ft.pt | 257M | v17 data + FineTranslations 30K (current default) |
| phase1_combined.pt | ~246M | Multi-source deduped |
| phase1_100M.pt | 100M | Smaller subset |

## NEVER STOP

Once the loop begins, do NOT pause to ask the human. Run experiments autonomously until manually stopped. If stuck, re-read this file, try combining ideas, try more radical changes. Each experiment takes ~2-3 hours, so you can run ~3-4 per overnight session.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Tokenizer-replacement experiment for the HY-MT1.5-1.8B translation model. The pipeline swaps the original HuggingFace tokenizer (vocab ~120K) for a custom SentencePiece-style "piece" tokenizer (vocab ~65K + special tokens), re-initializes the embedding matrix by mapping new tokens through the old tokenizer, then runs two training phases to recover (and ideally improve) zh↔en translation quality.

This is a Python ML codebase despite the C/C++-flavored `.gitignore` (legacy from the sibling Tokenizer repo).

## External dependencies

- **`piece_tokenizer`** — a compiled C++ Python extension from a sibling repo (`/home/tfbao/Shiyu/Tokenizer`). It is *not* on PyPI; it must be built and installed into the active venv. `replace_tokenizer.py`, `pretokenize.py`, `get_frozen_ids.py`, and `tokenizer_wrapper.py` all `import piece_tokenizer as pt`.
- **Project venv** is hardcoded in the `Makefile` and `run_eval.sh`: `/home/tfbao/new/HY-MT/.venv/bin/python`. Use it (or activate it) when running scripts directly — it has `torch`, `transformers`, `sacrebleu`, `unbabel-comet`, and the compiled `piece_tokenizer`.
- **Base model weights** (`HY-MT1.5-1.8B/`, `HY-MT1.5-1.8B-new-tok/`) and the `private/` directory (training data, checkpoints) are git-ignored — they live on disk only.

## Common commands

The `Makefile` is the entry point for the standard workflow. Variables at the top point at concrete data/model paths; edit those rather than overriding on the command line for repeated runs.

```bash
make tokenize   # pretokenize.py: pack cn/en sentences into [N, seq_len] int32 chunks → P1_TRAIN_DATA (.pt)
make phase1     # finetune_muon.py with --freeze_transformer: train embed_tokens + lm_head only
make phase2     # finetune_muon.py SFT mode: full fine-tune on translation JSONL
make eval       # eval.py on wmt22 (BLEU only by default; --no_comet)
make eval-p1    # same, but pointed at the phase 1 checkpoint
```

One-time setup (not in the Makefile, run manually):

```bash
# Add <pad> <user> <assistant> <system> to the piece .model file
python add_special_tokens.py --input <piece.model> --output piece_mt.model

# Resize embeddings + write tokenizer config + chat template
python replace_tokenizer.py \
    --old_model_path ./HY-MT1.5-1.8B \
    --new_tokenizer_path ./piece_mt.model \
    --output_path ./HY-MT1.5-1.8B-new-tok

# Optional: dump IDs of one-to-one mapped tokens (used by --freeze_mapped_embeds)
python get_frozen_ids.py --new_tokenizer ./piece_mt.model \
    --old_model_path ./HY-MT1.5-1.8B --output ./private/frozen_ids.json
```

There is no test suite.

## Architecture

### Data flow

```
old HF model + tokenizer ──┐
                           ├─► replace_tokenizer.py ─► HY-MT1.5-1.8B-new-tok/
piece_mt.model  ───────────┘     (resized embeds, piece.model, tokenizer_config.json,
                                  token_mapping.json, chat_template)

cn/en sentences ─► pretokenize.py ─► train_data.pt (packed [N, seq_len] chunks)
                                          │
                                          ▼
            finetune_muon.py phase 1 (CLM, freeze transformer) ─► phase1_v*/
                                          │
                                          ▼
news_commentary_sft.jsonl ─► finetune_muon.py phase 2 (SFT, full FT) ─► phase2/
                                          │
                                          ▼
                                  eval.py (WMT BLEU/COMET)
```

### Module roles

- `replace_tokenizer.py` — does the embedding-mapping surgery. For each new token, encodes its piece text with the *old* tokenizer and averages the corresponding old embeddings (one-to-one ≈73.5%, multi-to-one ≈25.5%, byte fallback ≈1%). Special tokens (`<s>`, `</s>`, `<pad>`, `<user>`, `<assistant>`, `<system>`) are mapped from the old model's corresponding control tokens (`<｜hy_Assistant｜>`, etc.). `tie_word_embeddings=True`, so resizing `embed_tokens` also drives `lm_head`. Writes `token_mapping.json` recording all the special token IDs.
- `tokenizer_wrapper.py` — `PieceTokenizerWrapper` exposes the HF-compatible surface (`pad_token_id`, `encode`, `decode`, `apply_chat_template`) on top of the C++ `piece_tokenizer`. `eval.py` and `finetune_muon.py` both call `load_tokenizer(model_path)`, which auto-detects `piece.model` in the model dir and falls back to `AutoTokenizer` otherwise. The chat template is hand-rolled in `apply_chat_template` and must stay consistent with the Jinja template emitted by `replace_tokenizer.py:create_hf_tokenizer_files`.
- `finetune_muon.py` — single training script with three dataset modes: `PreTokenizedDataset` (`.pt` file from `pretokenize.py`), `CLMDataset` (streaming text, packs into `<s>…</s>` windows of `max_seq_length`), and `SFTDataset` (chat-format JSONL, masks loss to assistant turns only). Optimizer split in `build_optimizer`: 2D non-embedding params → Muon (`SingleDeviceMuonWithAuxAdam`); embeddings, `lm_head`, and 1D params → AdamW. With `--freeze_transformer` only embed/lm_head are trainable, so it falls back to plain AdamW. `--freeze_mapped_embeds` zeros gradients for one-to-one mapped rows via a `register_hook` on `embed_tokens.weight`. SIGINT handling saves a checkpoint on first Ctrl+C; second Ctrl+C exits.
- `muon.py` — local copy of the Muon optimizer (Newton–Schulz orthogonalization for matrix updates).
- `pretokenize.py` — round-robin reads input files, encodes each line as `<s>…</s>`, concatenates into a flat buffer, slices into fixed-length chunks, saves as int32 `.pt`. Pass `--cn_dict` for Chinese segmentation if the piece tokenizer was built with one.
- `eval.py` — translates with greedy `model.generate`, scores with sacrebleu (`tokenize="zh"` for en→zh, `"13a"` for zh→en) and optionally COMET (`Unbabel/wmt22-comet-da`, looked up locally before falling back to download). Prompt templates `PROMPT_ZH2EN` / `PROMPT_EN2ZH` are inlined here.

### Two-phase training rationale

- **Phase 1** (`make phase1`): freeze the entire transformer, train *only* the embedding matrix on packed CLM data. Goal: align the freshly-initialized rows for the ~26% of tokens that are multi-mapped or fallback, without disturbing the transformer that was trained on the old vocabulary's distribution.
- **Phase 2** (`make phase2`): unfreeze, switch to SFT on translation pairs (chat-format JSONL with `{"messages": [...]}`). Loss is masked to assistant turns. Uses Muon for transformer matrices and Adam for embed/lm_head with separate LRs (`--muon_lr`, `--adam_lr`).

### Token ID conventions (after `replace_tokenizer.py`)

Old model: bos=120000, eos=120020, pad=120002. New model: unk=0, bos=1, eos=2, then base vocab, then `<pad>`, `<user>`, `<assistant>`, `<system>` appended at the end (exact IDs in `token_mapping.json`). When generating, left-pad with `pad_token_id` and pass `attention_mask`.

## `Qwen/` subproject — ReTok on Qwen3-0.6B-Base

Parallel experiment that applies the same tokenizer-replacement idea to a *base* (not translation) model, following the ReTok paper (Gu et al., arXiv:2410.04335). The directory is self-contained — copies of the scripts live there rather than sharing with the root.

- **Target**: `Qwen3-0.6B-Base` from ModelScope/HF; `tie_word_embeddings=True`, vocab 151936 → 65007 after swap.
- **Special-token mapping**: `<s>/</s>/<pad>` ← Qwen `<|endoftext|>` (151643); `<user>/<assistant>/<system>` ← Qwen `<|im_start|>` (151644). Base eval doesn't use chat tokens, so the role markers are init-only.
- **No SFT phase** for this subproject (the base model has no chat behavior to preserve). Per the user's plan: replace, then run benchmarks before any training.
- **Eval**: `lm-evaluation-harness` (CLI) for the original BBPE side; `eval_with_piece.py` for the swapped model (subclasses `HFLM` with a `_TokenizerStub` so the harness's `tok_encode` / `tok_decode` / `tok_batch_encode` / stop-criteria paths flow through `PieceTokenizerWrapper`). Tasks come from the ReTok Table 2 recipe: PIQA (5), ARC-C (25), HellaSwag (10), MMLU (5), CMMLU (5), AGIEval (0), BBH (3), HumanEval (0), GSM8K (5).
- **Workflow**: `make download` → `make replace` → `make eval-base` → `make eval-new`. HumanEval automatically gets `--confirm_run_unsafe_code` from the Makefile loop.

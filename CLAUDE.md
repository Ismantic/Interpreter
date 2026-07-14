# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

Three **independent** projects, one per top-level folder. They share only the venv
(`/home/tfbao/new/HY-MT/.venv`) — there are no cross-folder Python imports and no
cross-folder runtime data references. Each folder has its own `CLAUDE.md`; read it
before working there, and run that project's commands from inside its folder.

| Folder | Project | Base model | Tokenizer |
|---|---|---|---|
| `HYMT/` | Tokenizer-replacement (ReTok) experiment: swap HY-MT1.5-1.8B's HF tokenizer (~120K) for a custom piece tokenizer (~65K), remap embeddings, two-phase retrain. | pre-trained HY-MT1.5-1.8B | custom PieceTokenizer |
| `Qwen/` | Train a 1.7B zh↔en translation model from scratch (SFT → CPO → GRPO). | `Qwen3-1.7B-Base` | Qwen3 HF BPE (ChatML) |
| `ReTok/` | A/B replica of `Qwen/`: same data, hyperparameters, losses, rewards — differs only in base checkpoint (a ReTok phase-2 `v18`) and tokenizer (PieceTokenizer). Notable: vLLM-with-piece integration (`skip_tokenizer_init=True` + `TokensPrompt`, monkey-patched so TRL/GRPO inherits it). | ReTok phase-2 `v18` | PieceTokenizer |

`Qwen/` and `ReTok/` were an A/B pair that originally shared training
data via `../Qwen/`; that coupling has been broken — `ReTok/` now
carries its own copy of the SFT/CPO/GRPO data under `ReTok/data/`
(git-ignored). `ReTok/` still starts from a ReTok checkpoint produced by the
`HYMT/` line of work, but only as a weights file referenced by absolute path — no
code dependency.

## Known residual coupling

`Qwen/sherry_qat/build_qat_data_v2.py` (a leaf quantization sub-experiment) reads
QAT calibration text from `HYMT/private/*.txt` (~994 MB) by absolute path. It was
left as a path reference rather than duplicating ~1 GB. Everything else is decoupled.

## Notes shared across projects

- Large artifacts are git-ignored and live on disk only: model weights, `private/`,
  `*.pt`, `*.safetensors`, training/eval outputs (`output_*/`), logs, and the copied
  `ReTok/data/`. Only code, small configs, and docs are tracked.
- There is no test suite in any of the three projects.

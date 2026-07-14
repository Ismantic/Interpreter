"""Shared PieceTokenizer artifact list + copy helper.

Extracted from train.py so every module (train / eval / merge_lora / train_grpo /
piece_hf_tokenizer) can import it WITHOUT importing the train.py entrypoint.
Each ReTok checkpoint dir must be self-contained: model weights + these 5 files.
"""
import os
import shutil

_TOKENIZER_ARTIFACTS = [
    "piece.model",
    "dict.txt",
    "token_mapping.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
]


def _copy_tokenizer_artifacts(base_dir, save_dir):
    """Mirror PieceTokenizer's 5 files from the original base dir into a save dir."""
    os.makedirs(save_dir, exist_ok=True)
    for fn in _TOKENIZER_ARTIFACTS:
        src = os.path.join(base_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(save_dir, fn))

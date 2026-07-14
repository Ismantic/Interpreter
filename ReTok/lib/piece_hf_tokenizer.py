"""
PreTrainedTokenizer subclass that wraps PieceTokenizerWrapper for TRL compatibility.

Why: TRL GRPOTrainer (trl/trainer/grpo_trainer.py:322) requires
`isinstance(processing_class, PreTrainedTokenizerBase)`. A duck-typed wrapper
isn't enough; we need a real PreTrainedTokenizer subclass.

Approach: subclass `PreTrainedTokenizer` (slow path), delegate vocab/encode/decode
to PieceTokenizerWrapper. Override `apply_chat_template` directly (not via Jinja),
because piece's tokenizer doesn't recognize inline control tokens like "<user>" —
encoding "<user>Hello" BPE-splits it into ['<', 'user', '>', 'Hello']. So we
build chat IDs from the wrapper's apply_chat_template, which uses the right
control token IDs (1, 81900, 81901, 81902, 2).
"""
import os
import sys
from typing import List, Union, Dict, Any

from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

# Shared piece-tokenizer modules are co-located here in ReTok/lib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer_wrapper import PieceTokenizerWrapper  # noqa: E402
from tok_artifacts import _copy_tokenizer_artifacts, _TOKENIZER_ARTIFACTS  # noqa: E402


class PieceTokenizerForTRL(PreTrainedTokenizer):
    """PreTrainedTokenizer subclass backed by PieceTokenizerWrapper.

    `processing_class` for TRL GRPOTrainer. Implements vocab/encode/decode via
    the wrapper; overrides `apply_chat_template` to build piece-format chat IDs
    directly without going through Jinja (since piece doesn't recognize inline
    "<user>" etc. as control tokens).
    """

    vocab_files_names: Dict[str, str] = {}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, model_dir, **kwargs):
        self._piece = PieceTokenizerWrapper(model_dir)
        self._model_dir = model_dir
        self._vocab_cache = None

        super().__init__(
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            unk_token="<unk>",
            additional_special_tokens=["<user>", "<assistant>", "<system>"],
            model_max_length=kwargs.pop("model_max_length", 32768),
            padding_side="left",
            **kwargs,
        )
        # Override the token IDs to match piece's actual vocab IDs.
        # PreTrainedTokenizer's __init__ adds special tokens to its vocab if they
        # don't exist; here we want it to use piece's fixed IDs.
        self.bos_token_id = self._piece.bos_token_id   # 1
        self.eos_token_id = self._piece.eos_token_id   # 2
        self.pad_token_id = self._piece.pad_token_id   # 81899
        self.unk_token_id = 0

    @property
    def vocab_size(self) -> int:
        return self._piece.vocab_size

    def get_vocab(self) -> Dict[str, int]:
        if self._vocab_cache is None:
            v = {}
            for i in range(self.vocab_size):
                try:
                    v[self._piece._tok.id_to_piece(i)] = i
                except UnicodeDecodeError:
                    # Byte-fallback piece (raw byte not valid UTF-8 on its own).
                    # Use a synthetic name so the vocab dict stays consistent.
                    v[f"<byte_{i}>"] = i
            self._vocab_cache = v
        return self._vocab_cache

    # ---- core encode/decode (used when caller hits the slow tokenizer path) ----
    def _tokenize(self, text, **kwargs) -> List[str]:
        return self._piece._tok.encode_as_pieces(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._piece._tok.piece_to_id(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self._piece._tok.id_to_piece(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        # Decode by joining piece bytes; PieceTokenizer uses ▁ for word boundary.
        return "".join(tokens).replace("▁", " ").strip()

    # Fast path: bypass slow encode entirely, use C++ encode_as_ids directly.
    def encode(self, text, add_special_tokens=False, **kwargs) -> List[int]:
        ids = self._piece._tok.encode_as_ids(text)
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, token_ids, skip_special_tokens: bool = True, **kwargs) -> str:
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        return self._piece.decode(list(token_ids), skip_special_tokens=skip_special_tokens)

    def batch_decode(self, sequences, skip_special_tokens: bool = True, **kwargs) -> List[str]:
        return [self.decode(s, skip_special_tokens=skip_special_tokens) for s in sequences]

    # ---- chat template: override, do NOT use Jinja ----
    def apply_chat_template(
        self,
        conversation=None,
        chat_template=None,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: int = None,
        return_tensors: str = None,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        tokenizer_kwargs=None,
        **kwargs,
    ):
        """Build chat-formatted IDs/text. Accepts a single conversation OR a
        batched list-of-conversations (TRL passes batched)."""
        # Detect batched input: list whose first element is also a list of dicts.
        is_batched = (
            isinstance(conversation, list)
            and len(conversation) > 0
            and isinstance(conversation[0], list)
        )
        conversations = conversation if is_batched else [conversation]

        per_seq_ids = [
            self._piece.apply_chat_template(c, tokenize=True, add_generation_prompt=add_generation_prompt)
            for c in conversations
        ]

        if truncation and max_length:
            per_seq_ids = [ids[:max_length] for ids in per_seq_ids]

        if not tokenize:
            decoded = [self._piece.decode(ids, skip_special_tokens=False) for ids in per_seq_ids]
            return decoded if is_batched else decoded[0]

        if return_dict:
            if padding:
                max_len = max(len(ids) for ids in per_seq_ids)
                # left-pad (caller may override)
                padded_ids = [
                    [self.pad_token_id] * (max_len - len(ids)) + list(ids) for ids in per_seq_ids
                ]
                attn = [
                    [0] * (max_len - len(ids)) + [1] * len(ids) for ids in per_seq_ids
                ]
            else:
                padded_ids = per_seq_ids
                attn = [[1] * len(ids) for ids in per_seq_ids]

            if not is_batched:
                padded_ids = padded_ids[0]
                attn = attn[0]

            data = {"input_ids": padded_ids, "attention_mask": attn}
            if return_tensors:
                import torch
                if return_tensors == "pt":
                    data = {k: torch.tensor(v, dtype=torch.long) for k, v in data.items()}
                else:
                    raise NotImplementedError(f"return_tensors={return_tensors!r} not supported")
            return BatchEncoding(data)

        # tokenize=True, return_dict=False: return list of ids (batched) or ids (single)
        return per_seq_ids if is_batched else per_seq_ids[0]

    # ---- vocab save: copy piece's 5 files ----
    def save_vocabulary(self, save_directory: str, filename_prefix=None):
        _copy_tokenizer_artifacts(self._model_dir, save_directory)
        return tuple(
            os.path.join(save_directory, fn)
            for fn in _TOKENIZER_ARTIFACTS
            if os.path.exists(os.path.join(save_directory, fn))
        )

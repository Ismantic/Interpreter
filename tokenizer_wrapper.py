"""
Wrapper around piece_tokenizer that provides a HuggingFace-like interface.
Used by eval.py and finetune_muon.py.
"""
import os
import json
import piece_tokenizer as pt


class PieceTokenizerWrapper:
    def __init__(self, model_dir):
        """Load tokenizer from a model directory containing piece.model and token_mapping.json."""
        self._tok = pt.Tokenizer()

        # Find the .model file
        model_file = os.path.join(model_dir, "piece.model")
        if not os.path.exists(model_file):
            model_file = os.path.join(model_dir, "piece_mt.model")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"No piece model found in {model_dir}")

        # Optional CN segmentation dict — without it, encode is O(n^2) on long
        # input because the tokenizer skips pre-splitting entirely.
        cn_dict = os.path.join(model_dir, "dict.txt")
        if os.path.exists(cn_dict):
            self._tok.load(model_file, cn_dict)
        else:
            self._tok.load(model_file)

        # Load token mapping
        mapping_file = os.path.join(model_dir, "token_mapping.json")
        if os.path.exists(mapping_file):
            with open(mapping_file) as f:
                mapping = json.load(f)
            self.pad_token_id = mapping["pad_id"]
            self.bos_token_id = mapping["bos_id"]
            self.eos_token_id = mapping["eos_id"]
            self.user_token_id = mapping.get("user_id")
            self.assistant_token_id = mapping.get("assistant_id")
            self.system_token_id = mapping.get("system_id")
        else:
            # Fallback to piece_to_id lookups
            self.bos_token_id = self._tok.piece_to_id("<s>")
            self.eos_token_id = self._tok.piece_to_id("</s>")
            self.pad_token_id = self._tok.piece_to_id("<pad>")
            self.user_token_id = self._tok.piece_to_id("<user>")
            self.assistant_token_id = self._tok.piece_to_id("<assistant>")
            self.system_token_id = self._tok.piece_to_id("<system>")
            if self.pad_token_id < 0:
                self.pad_token_id = 0

    @property
    def vocab_size(self):
        return self._tok.vocab_size()

    def encode(self, text, add_special_tokens=False):
        ids = self._tok.encode_as_ids(text)
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        if skip_special_tokens:
            special = {self.bos_token_id, self.eos_token_id, self.pad_token_id,
                       self.user_token_id, self.assistant_token_id, self.system_token_id}
            ids = [i for i in ids if i not in special]
        try:
            return self._tok.decode(ids)
        except UnicodeDecodeError:
            # Model emitted byte-fallback piece(s) that don't form valid UTF-8.
            # Per-piece fallback: keep ids that decode cleanly, drop the rest.
            parts = []
            for i in ids:
                try:
                    parts.append(self._tok.id_to_piece(i))
                except UnicodeDecodeError:
                    continue
            return "".join(parts).replace("▁", " ")

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        """Build chat-formatted token sequence from messages."""
        ids = []

        # Check for system message
        start = 0
        if messages and messages[0]["role"] == "system":
            ids.append(self.bos_token_id)
            ids.extend(self._tok.encode_as_ids(messages[0]["content"]))
            ids.append(self.system_token_id)
            start = 1
        else:
            ids.append(self.bos_token_id)

        for msg in messages[start:]:
            if msg["role"] == "user":
                ids.append(self.user_token_id)
                ids.extend(self._tok.encode_as_ids(msg["content"]))
            elif msg["role"] == "assistant":
                ids.append(self.assistant_token_id)
                ids.extend(self._tok.encode_as_ids(msg["content"]))
                ids.append(self.eos_token_id)

        if add_generation_prompt:
            ids.append(self.assistant_token_id)

        if tokenize:
            return ids
        else:
            # Return as string (rarely needed)
            return self._tok.decode(ids)

    def save_pretrained(self, output_dir):
        """Save tokenizer files to directory (for checkpoint saving)."""
        import shutil
        os.makedirs(output_dir, exist_ok=True)
        # Copy piece.model
        src = os.path.join(os.path.dirname(output_dir), "piece.model")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, "piece.model"))
        # Save mapping
        mapping = {
            "bos_id": self.bos_token_id,
            "eos_id": self.eos_token_id,
            "pad_id": self.pad_token_id,
            "user_id": self.user_token_id,
            "assistant_id": self.assistant_token_id,
            "system_id": self.system_token_id,
        }
        with open(os.path.join(output_dir, "token_mapping.json"), "w") as f:
            json.dump(mapping, f, indent=2)

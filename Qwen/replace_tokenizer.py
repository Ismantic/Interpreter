"""
Replace Qwen3-0.6B-Base tokenizer with the piece_mt tokenizer (ReTok-style).

For each token in the new vocabulary:
  - encode its piece text with Qwen's BBPE
  - mean of those old embeddings -> new embedding
  - if encoding yields exactly one old token, this is equivalent to the
    "copy when token exists in both vocabs" rule from the ReTok paper

Special tokens (already inside piece_mt.model):
  <unk>=0, <s>=1, </s>=2, ..., <pad>, <user>, <assistant>, <system>
For a base model, <user>/<assistant>/<system> are unused at eval time;
we still initialize them from Qwen's <|im_start|> for sanity.

Usage:
    python replace_tokenizer.py \
        --old_model_path ./Qwen/Qwen3-0.6B-Base \
        --new_tokenizer_path ./Qwen/piece_mt.model \
        --output_path ./Qwen/Qwen3-0.6B-Base-new-tok
"""
import os
import json
import argparse
import torch
import piece_tokenizer as pt
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


SPECIAL_TOKENS = ["<pad>", "<user>", "<assistant>", "<system>"]


def build_embedding_mapping(old_tokenizer, new_tok, new_vocab_size, old_embeddings):
    embed_dim = old_embeddings.shape[1]
    new_embeddings = torch.zeros(new_vocab_size, embed_dim, dtype=old_embeddings.dtype)
    fallback = old_embeddings.float().mean(dim=0).to(old_embeddings.dtype)

    one_to_one = multi = skipped = 0

    for i in tqdm(range(new_tok.vocab_size()), desc="Mapping embeddings"):
        try:
            piece = new_tok.id_to_piece(i)
        except UnicodeDecodeError:
            new_embeddings[i] = fallback
            skipped += 1
            continue

        if piece in ("<unk>", "<s>", "</s>") or piece in SPECIAL_TOKENS:
            continue  # filled in by caller

        text = piece.replace("▁", " ")
        if not text.strip():
            text = " "

        try:
            old_ids = old_tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            new_embeddings[i] = fallback
            skipped += 1
            continue

        if old_ids:
            old_vecs = old_embeddings[old_ids].float()
            new_embeddings[i] = old_vecs.mean(dim=0).to(old_embeddings.dtype)
            if len(old_ids) == 1:
                one_to_one += 1
            else:
                multi += 1
        else:
            new_embeddings[i] = fallback
            skipped += 1

    total = new_tok.vocab_size()
    print(f"  one-to-one : {one_to_one:>6} ({100*one_to_one/total:5.1f}%)")
    print(f"  multi-to-one: {multi:>6} ({100*multi/total:5.1f}%)")
    print(f"  fallback   : {skipped:>6} ({100*skipped/total:5.1f}%)")
    return new_embeddings


def write_tokenizer_files(args, special_token_ids, output_path):
    """Minimal tokenizer config — actual decoding goes through PieceTokenizerWrapper."""
    tokenizer_config = {
        "model_type": "qwen3",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "clean_up_tokenization_spaces": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    special_tokens_map = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
    }
    with open(os.path.join(output_path, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_path, "special_tokens_map.json"), "w") as f:
        json.dump(special_tokens_map, f, indent=2, ensure_ascii=False)

    import shutil
    shutil.copy2(args.new_tokenizer_path, os.path.join(output_path, "piece.model"))
    print(f"Saved tokenizer files to {output_path}")


def lookup_id(old_tokenizer, piece):
    tid = old_tokenizer.convert_tokens_to_ids(piece)
    if tid is None or tid == old_tokenizer.unk_token_id:
        return None
    return tid


def main(args):
    print(f"Loading old model + tokenizer from {args.old_model_path} ...")
    old_tokenizer = AutoTokenizer.from_pretrained(args.old_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.old_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    print(f"  old vocab size = {old_tokenizer.vocab_size}")
    print(f"  tie_word_embeddings = {model.config.tie_word_embeddings}")

    print(f"Loading new piece tokenizer from {args.new_tokenizer_path} ...")
    new_tok = pt.Tokenizer()
    new_tok.load(args.new_tokenizer_path)
    new_vocab_size = new_tok.vocab_size()
    print(f"  new vocab size = {new_vocab_size}")

    # piece_mt.model already contains <pad>/<user>/<assistant>/<system>
    special_token_ids = {}
    for tok in SPECIAL_TOKENS:
        idx = new_tok.piece_to_id(tok)
        if idx < 0:
            raise RuntimeError(f"Special token {tok} not found in {args.new_tokenizer_path}")
        special_token_ids[tok] = idx
        print(f"  {tok:12s} -> id {idx}")

    old_embeddings = model.model.embed_tokens.weight.data.clone()
    print(f"Old embedding shape: {tuple(old_embeddings.shape)}")

    print("Building new embeddings...")
    new_embeddings = build_embedding_mapping(
        old_tokenizer, new_tok, new_vocab_size, old_embeddings
    )

    # Map our control / chat tokens to Qwen's analogous embeddings.
    fallback = old_embeddings.float().mean(dim=0).to(old_embeddings.dtype)
    qwen_endoftext = lookup_id(old_tokenizer, "<|endoftext|>")
    qwen_im_start = lookup_id(old_tokenizer, "<|im_start|>")

    # <unk>
    new_embeddings[0] = fallback

    # <s> / </s>: base model has no separate bos/eos; use <|endoftext|>
    if qwen_endoftext is not None:
        new_embeddings[1] = old_embeddings[qwen_endoftext]
        new_embeddings[2] = old_embeddings[qwen_endoftext]
        print(f"  <s>, </s>     <- Qwen <|endoftext|> (id {qwen_endoftext})")
    else:
        new_embeddings[1] = fallback
        new_embeddings[2] = fallback
        print("  <s>, </s>     <- mean fallback (no <|endoftext|> in old vocab?)")

    # <pad>
    if qwen_endoftext is not None:
        new_embeddings[special_token_ids["<pad>"]] = old_embeddings[qwen_endoftext]
        print(f"  <pad>         <- Qwen <|endoftext|> (id {qwen_endoftext})")

    # <user>/<assistant>/<system>: unused for base eval, init to <|im_start|>
    role_src = old_embeddings[qwen_im_start] if qwen_im_start is not None else fallback
    for tok in ("<user>", "<assistant>", "<system>"):
        new_embeddings[special_token_ids[tok]] = role_src
    if qwen_im_start is not None:
        print(f"  <user>/<assistant>/<system> <- Qwen <|im_start|> (id {qwen_im_start})")

    # Resize and inject
    print("Resizing model embeddings...")
    model.resize_token_embeddings(new_vocab_size)
    model.model.embed_tokens.weight.data = new_embeddings
    if model.config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
        print("  lm_head tied to embed_tokens")
    else:
        model.lm_head.weight.data = new_embeddings.clone()
        print("  lm_head set to copy of new embeddings (untied)")

    # Update config
    model.config.vocab_size = new_vocab_size
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.config.pad_token_id = special_token_ids["<pad>"]

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.bos_token_id = 1
        model.generation_config.eos_token_id = 2
        model.generation_config.pad_token_id = special_token_ids["<pad>"]

    os.makedirs(args.output_path, exist_ok=True)
    print(f"Saving model to {args.output_path}...")
    model.save_pretrained(args.output_path)
    write_tokenizer_files(args, special_token_ids, args.output_path)

    mapping = {
        "base_vocab_size": new_vocab_size,
        "total_vocab_size": new_vocab_size,
        "special_tokens": special_token_ids,
        "bos_id": 1,
        "eos_id": 2,
        "unk_id": 0,
        "pad_id": special_token_ids["<pad>"],
        "user_id": special_token_ids["<user>"],
        "assistant_id": special_token_ids["<assistant>"],
        "system_id": special_token_ids["<system>"],
    }
    with open(os.path.join(args.output_path, "token_mapping.json"), "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"\nDone. Saved to {args.output_path}")
    print(f"  vocab: {model.config.vocab_size}")
    print(f"  embed: {tuple(model.model.embed_tokens.weight.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_model_path", type=str, required=True)
    parser.add_argument("--new_tokenizer_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args)

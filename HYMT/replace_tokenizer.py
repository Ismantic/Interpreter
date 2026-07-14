"""
Replace HY-MT1.5-1.8B tokenizer with a custom piece tokenizer.
Initializes new embeddings by mapping each new token through the old tokenizer.

Usage:
    python replace_tokenizer.py \
        --old_model_path ./HY-MT1.5-1.8B \
        --new_tokenizer_path /home/tfbao/Shiyu/Tokenizer/scripts/output/piece.model \
        --output_path ./HY-MT1.5-1.8B-new-tok
"""
import os
import json
import argparse
import torch
import piece_tokenizer as pt
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Chat special tokens to add (matching HY-MT style)
SPECIAL_TOKENS = [
    "<pad>",       # padding
    "<user>",      # user turn marker
    "<assistant>", # assistant turn marker
    "<system>",    # system prompt marker
]


def build_embedding_mapping(old_tokenizer, new_tok, new_vocab_size, old_embeddings):
    """
    For each token in new vocab, encode its text with old tokenizer,
    average the old embeddings as initialization.
    """
    embed_dim = old_embeddings.shape[1]
    new_embeddings = torch.zeros(new_vocab_size, embed_dim, dtype=old_embeddings.dtype)

    # Mean of old embeddings as fallback
    fallback = old_embeddings.float().mean(dim=0).to(old_embeddings.dtype)

    mapped = 0
    skipped = 0
    for i in tqdm(range(new_tok.vocab_size()), desc="Mapping embeddings"):
        try:
            piece = new_tok.id_to_piece(i)
        except UnicodeDecodeError:
            # Raw byte tokens - use fallback
            new_embeddings[i] = fallback
            skipped += 1
            continue

        # Skip control tokens (handle separately)
        if piece in ("<unk>", "<s>", "</s>", "<pad>", "<user>", "<assistant>", "<system>"):
            continue

        # Remove SentencePiece's ▁ prefix for lookup
        text = piece.replace("▁", " ")
        if not text.strip():
            text = " "

        # Encode with old tokenizer
        try:
            old_ids = old_tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            new_embeddings[i] = fallback
            skipped += 1
            continue

        if old_ids:
            old_vecs = old_embeddings[old_ids].float()
            new_embeddings[i] = old_vecs.mean(dim=0).to(old_embeddings.dtype)
            mapped += 1
        else:
            new_embeddings[i] = fallback

    print(f"Mapped {mapped}/{new_tok.vocab_size()} tokens via old tokenizer")

    # Map special tokens from old model
    # Old: bos=120000, eos=120020, pad=120002
    old_bos_emb = old_embeddings[120000]
    old_eos_emb = old_embeddings[120020]
    old_pad_emb = old_embeddings[120002]

    # New: unk=0, bos=1, eos=2
    new_embeddings[0] = fallback  # <unk>
    new_embeddings[1] = old_bos_emb  # <s> -> old bos
    new_embeddings[2] = old_eos_emb  # </s> -> old eos

    return new_embeddings


def create_hf_tokenizer_files(new_tok, special_token_ids, output_path):
    """Create HuggingFace-compatible tokenizer config files for the new tokenizer."""

    pad_id = special_token_ids["<pad>"]
    bos_id = 1
    eos_id = 2

    # tokenizer_config.json
    tokenizer_config = {
        "model_type": "hunyuan_v1_dense",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "clean_up_tokenization_spaces": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": (
            "{% if messages[0]['role'] == 'system' %}"
            "{% set loop_messages = messages[1:] %}"
            "{% set system_message = messages[0]['content'] %}"
            "<s>{{ system_message }}<system>"
            "{% else %}"
            "{% set loop_messages = messages %}"
            "<s>"
            "{% endif %}"
            "{% for message in loop_messages %}"
            "{% if message['role'] == 'user' %}"
            "<user>{{ message['content'] }}"
            "{% elif message['role'] == 'assistant' %}"
            "<assistant>{{ message['content'] }}</s>"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}<assistant>{% endif %}"
        ),
    }

    # special_tokens_map.json
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

    # Copy the piece model
    import shutil
    shutil.copy2(
        args.new_tokenizer_path,
        os.path.join(output_path, "piece.model")
    )

    print(f"Saved tokenizer config to {output_path}")


def main(args):
    device = "cpu"  # Do everything on CPU to save GPU memory

    # Load old model and tokenizer
    print("Loading old model and tokenizer...")
    old_tokenizer = AutoTokenizer.from_pretrained(args.old_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.old_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )

    # Load new tokenizer
    print("Loading new tokenizer...")
    new_tok = pt.Tokenizer()
    new_tok.load(args.new_tokenizer_path)
    base_vocab_size = new_tok.vocab_size()
    print(f"New tokenizer vocab: {base_vocab_size}")

    # Add special tokens beyond base vocab
    special_token_ids = {}
    next_id = base_vocab_size
    for token in SPECIAL_TOKENS:
        if new_tok.piece_to_id(token) < 0:  # not already in vocab
            special_token_ids[token] = next_id
            next_id += 1
            print(f"  Added special token: {token} -> {special_token_ids[token]}")
        else:
            special_token_ids[token] = new_tok.piece_to_id(token)
            print(f"  Existing token: {token} -> {special_token_ids[token]}")

    new_vocab_size = next_id
    print(f"Final vocab size: {new_vocab_size} (base {base_vocab_size} + {len(special_token_ids)} special)")

    # Get old embeddings
    old_embeddings = model.model.embed_tokens.weight.data.clone()
    print(f"Old embedding shape: {old_embeddings.shape}")

    # Build new embeddings
    print("Building new embeddings...")
    new_embeddings = build_embedding_mapping(old_tokenizer, new_tok, new_vocab_size, old_embeddings)

    # Initialize special token embeddings
    # Map <pad> from old pad embedding
    if "<pad>" in special_token_ids:
        new_embeddings[special_token_ids["<pad>"]] = old_embeddings[120002]  # old pad
    # Map chat tokens from old model's corresponding tokens
    old_assistant_id = old_tokenizer.convert_tokens_to_ids('<｜hy_Assistant｜>')
    old_user_id = old_tokenizer.convert_tokens_to_ids('<｜hy_User｜>')
    if "<assistant>" in special_token_ids and old_assistant_id is not None:
        new_embeddings[special_token_ids["<assistant>"]] = old_embeddings[old_assistant_id]
        print(f"  Mapped <assistant> from old token {old_assistant_id}")
    if "<user>" in special_token_ids and old_user_id is not None:
        new_embeddings[special_token_ids["<user>"]] = old_embeddings[old_user_id]
        print(f"  Mapped <user> from old token {old_user_id}")
    # <system> - use average of old bos embedding
    if "<system>" in special_token_ids:
        new_embeddings[special_token_ids["<system>"]] = old_embeddings[120000]

    print(f"New embedding shape: {new_embeddings.shape}")

    # Resize model embeddings
    # Since tie_word_embeddings=True, resizing embed_tokens also affects lm_head
    model.resize_token_embeddings(new_vocab_size)
    model.model.embed_tokens.weight.data = new_embeddings
    # For tied embeddings, lm_head shares the same weight
    if model.config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
        print("Tied lm_head to embed_tokens (shared weights)")

    # Update config
    model.config.vocab_size = new_vocab_size
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.config.pad_token_id = special_token_ids["<pad>"]

    # Update generation config
    model.generation_config.bos_token_id = 1
    model.generation_config.eos_token_id = 2
    model.generation_config.pad_token_id = special_token_ids["<pad>"]

    # Save model
    os.makedirs(args.output_path, exist_ok=True)
    print(f"Saving model to {args.output_path}...")
    model.save_pretrained(args.output_path)

    # Save tokenizer files
    create_hf_tokenizer_files(new_tok, special_token_ids, args.output_path)

    # Save special token mapping for reference
    mapping = {
        "base_vocab_size": base_vocab_size,
        "total_vocab_size": new_vocab_size,
        "special_tokens": special_token_ids,
        "bos_id": 1,
        "eos_id": 2,
        "unk_id": 0,
        "pad_id": special_token_ids["<pad>"],
        "user_id": special_token_ids.get("<user>"),
        "assistant_id": special_token_ids.get("<assistant>"),
        "system_id": special_token_ids.get("<system>"),
    }
    with open(os.path.join(args.output_path, "token_mapping.json"), "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"\nDone! New model saved to {args.output_path}")
    print(f"  Vocab: {model.config.vocab_size} (was 120818)")
    print(f"  Embedding: {model.model.embed_tokens.weight.shape}")
    print(f"  Special tokens: {special_token_ids}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_model_path", type=str, required=True)
    parser.add_argument("--new_tokenizer_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args)

"""
Add special tokens to a piece .model file.
Appends CONTROL-type tokens and updates the header.

Usage:
    python add_special_tokens.py \
        --input /home/tfbao/Shiyu/Tokenizer/scripts/output/piece.model \
        --output ./piece_mt.model \
        --tokens '<pad>,<user>,<assistant>,<system>'
"""
import argparse
import re


def main(args):
    tokens_to_add = [t.strip() for t in args.tokens.split(",")]

    with open(args.input, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse current size from [Pieces] section
    m = re.search(r"^\[Pieces\]\nsize=(\d+)", content, re.MULTILINE)
    if not m:
        raise ValueError("Cannot find [Pieces] section")
    old_size = int(m.group(1))
    print(f"Old vocab size: {old_size}")

    # Check which tokens already exist
    existing = set()
    for line in content.split("\n"):
        if "\t" in line:
            parts = line.split("\t")
            if len(parts) >= 2:
                existing.add(parts[1])

    # Filter out tokens already in vocab
    new_tokens = [t for t in tokens_to_add if t not in existing]
    skip_tokens = [t for t in tokens_to_add if t in existing]
    if skip_tokens:
        print(f"Skipping (already exist): {skip_tokens}")

    if not new_tokens:
        print("No new tokens to add.")
        return

    # Build new lines: id, piece, score=0, type=3 (CONTROL), u="", v=""
    new_lines = []
    for i, token in enumerate(new_tokens):
        idx = old_size + i
        # Format: index\tpiece\tscore\ttype\tu\tv
        new_lines.append(f"{idx}\t{token}\t0\t3\t\t")
        print(f"  Added: {idx}\t{token}\t(CONTROL)")

    new_size = old_size + len(new_tokens)

    # Update size in [Pieces] header
    content = content.replace(f"size={old_size}", f"size={new_size}")

    # Update pad_id in [CounterSpec] if <pad> was added
    if "<pad>" in new_tokens:
        pad_id = old_size + new_tokens.index("<pad>")
        content = re.sub(r"pad_id=-?\d+", f"pad_id={pad_id}", content)
        print(f"  Updated pad_id to {pad_id}")

    # Update vocab_size in [CounterSpec]
    content = re.sub(r"vocab_size=\d+", f"vocab_size={new_size}", content)

    # Append new tokens at end
    content = content.rstrip("\n") + "\n" + "\n".join(new_lines) + "\n"

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\nNew vocab size: {new_size}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tokens", type=str, default="<pad>,<user>,<assistant>,<system>",
                        help="Comma-separated special tokens to add")
    args = parser.parse_args()
    main(args)

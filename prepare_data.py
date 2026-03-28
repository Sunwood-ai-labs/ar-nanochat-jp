"""
Download and prepare Japanese text data for nanoJP-GPT training.
Uses HuggingFace datasets (CC-100 Japanese subset) or Wikipedia JP.
Outputs: data/train.bin, data/val.bin (tokenized as uint16)
"""

import os
import sys

# Byte-level tokenizer for Japanese — no external tokenizer needed
# This is intentionally simple so AutoResearch can discover improvements

def build_vocab():
    """Build a character-level vocabulary covering common Japanese text."""
    chars = []
    # ASCII
    for i in range(32, 127):
        chars.append(chr(i))
    # Hiragana
    for i in range(0x3040, 0x309F + 1):
        chars.append(chr(i))
    # Katakana
    for i in range(0x30A0, 0x30FF + 1):
        chars.append(chr(i))
    # CJK Unified Ideographs — extended to cover all Joyo + Jinmeiyo kanji
    for i in range(0x4E00, 0x9FFF + 1):
        chars.append(chr(i))
    # Full-width symbols
    for i in range(0xFF00, 0xFFEF + 1):
        chars.append(chr(i))
    # Special tokens
    chars = ["<PAD>", "<UNK>"] + chars

    vocab = {c: i for i, c in enumerate(chars)}
    return vocab, chars


def encode(text, vocab):
    """Encode text to token IDs. Unknown chars map to UNK."""
    unk_id = vocab.get("<UNK>", 1)
    return [vocab.get(c, unk_id) for c in text]


def decode(token_ids, chars):
    """Decode token IDs back to text."""
    return "".join(chars[i] if i < len(chars) else "<UNK>" for i in token_ids)


def prepare_wikipedia_jp(output_dir="data", val_ratio=0.05, max_samples=None):
    """Download Japanese Wikipedia dump and prepare training data."""
    os.makedirs(output_dir, exist_ok=True)

    # Check if already prepared
    if os.path.exists(f"{output_dir}/train.bin") and os.path.exists(f"{output_dir}/val.bin"):
        print(f"Data already exists in {output_dir}/. Skipping.")
        return

    try:
        from datasets import load_dataset
        print("Downloading CC-100 Japanese dataset from HuggingFace...")
        ds = load_dataset("cc100", lang="ja", split="train", streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"Could not load cc100: {e}")
        print("Falling back to synthetic Japanese data for testing...")
        _create_synthetic_data(output_dir, val_ratio)
        return

    vocab, chars = build_vocab()
    print(f"Vocabulary size: {len(vocab)}")

    # Collect text and tokenize
    all_tokens = []
    count = 0
    for item in ds:
        text = item.get("text", "")
        if len(text) < 50:  # Skip very short texts
            continue
        tokens = encode(text, vocab)
        all_tokens.extend(tokens)
        count += 1
        if count % 10000 == 0:
            print(f"  Processed {count} documents, {len(all_tokens):,} tokens...")
        if max_samples and count >= max_samples:
            break
        # Cap at ~50M tokens for 6GB VRAM
        if len(all_tokens) >= 50_000_000:
            print(f"  Reached 50M token cap at {count} documents.")
            break

    if len(all_tokens) < 10_000:
        print("Not enough data. Falling back to synthetic data.")
        _create_synthetic_data(output_dir, val_ratio)
        return

    print(f"Total tokens: {len(all_tokens):,}")

    # Split into train/val
    import numpy as np
    val_size = int(len(all_tokens) * val_ratio)
    train_tokens = np.array(all_tokens[:-val_size], dtype=np.uint16)
    val_tokens = np.array(all_tokens[-val_size:], dtype=np.uint16)

    train_tokens.tofile(f"{output_dir}/train.bin")
    val_tokens.tofile(f"{output_dir}/val.bin")

    print(f"Train: {len(train_tokens):,} tokens -> {output_dir}/train.bin")
    print(f"Val:   {len(val_tokens):,} tokens -> {output_dir}/val.bin")
    print(f"Vocab: {len(vocab)}")

    # Save vocab metadata
    import json
    meta = {
        "vocab_size": len(vocab),
        "total_tokens": len(all_tokens),
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
    }
    with open(f"{output_dir}/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _create_synthetic_data(output_dir, val_ratio):
    """Create synthetic Japanese data for testing when no download is available."""
    print("Creating synthetic Japanese data...")
    vocab, chars = build_vocab()

    # Generate synthetic Japanese-like text
    import random
    random.seed(42)

    hiragana = [chr(i) for i in range(0x3042, 0x3093 + 1)]  # あ-ん
    katakana = [chr(i) for i in range(0x30A2, 0x30F3 + 1)]  # ア-ン
    kanji_sample = [chr(i) for i in range(0x4E00, 0x5000 + 1)]  # Common kanji

    all_tokens = []
    for _ in range(500_000):  # ~500K tokens of synthetic data
        # Mix of character types
        r = random.random()
        if r < 0.5:
            c = random.choice(hiragana)
        elif r < 0.75:
            c = random.choice(katakana)
        elif r < 0.95:
            c = random.choice(kanji_sample)
        else:
            c = random.choice("。、！？「」")  # Punctuation
        all_tokens.append(vocab.get(c, 1))

    import numpy as np
    val_size = int(len(all_tokens) * val_ratio)
    train_tokens = np.array(all_tokens[:-val_size], dtype=np.uint16)
    val_tokens = np.array(all_tokens[-val_size:], dtype=np.uint16)

    train_tokens.tofile(f"{output_dir}/train.bin")
    val_tokens.tofile(f"{output_dir}/val.bin")

    print(f"Synthetic data — Train: {len(train_tokens):,}, Val: {len(val_tokens):,}")

    import json
    meta = {
        "vocab_size": len(vocab),
        "total_tokens": len(all_tokens),
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "synthetic": True,
    }
    with open(f"{output_dir}/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    prepare_wikipedia_jp()

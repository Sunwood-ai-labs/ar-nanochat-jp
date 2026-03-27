#!/usr/bin/env python3
"""
Japanese text generation benchmark for nanoJP-GPT.
Evaluates model quality on character-level metrics specific to Japanese.
"""

import os
import sys
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, BASE_CONFIG
from prepare_data import build_vocab, encode, decode


# ─── Japanese Benchmark Prompts ────────────────────────────────
BENCHMARK_PROMPTS = [
    "日本の首都は",
    "今日はとても良い天気で",
    "機械学習とは",
    "東京タワーは",
    "プログラミング言語の",
    "桜の花が咲く季節は",
    "人工知能の研究は",
    "日本語にはひらがなと",
]

# Expected character type distribution for natural Japanese (approximate)
# Based on CC-100 Japanese statistics
EXPECTED_DIST = {
    "hiragana": 0.40,
    "katakana": 0.10,
    "kanji": 0.35,
    "ascii": 0.05,
    "punctuation": 0.10,
}


def classify_char(c):
    cp = ord(c)
    if 0x3040 <= cp <= 0x309F:
        return "hiragana"
    elif 0x30A0 <= cp <= 0x30FF:
        return "katakana"
    elif 0x4E00 <= cp <= 0x9FFF:
        return "kanji"
    elif 0x0041 <= cp <= 0x007A or 0x0030 <= cp <= 0x0039:
        return "ascii"
    else:
        return "punctuation"


def compute_char_distribution(text):
    """Compute character type distribution of text."""
    if not text:
        return {k: 0.0 for k in EXPECTED_DIST}
    counts = {k: 0 for k in EXPECTED_DIST}
    for c in text:
        cat = classify_char(c)
        if cat in counts:
            counts[cat] += 1
    total = sum(counts.values())
    if total == 0:
        return {k: 0.0 for k in EXPECTED_DIST}
    return {k: v / total for k, v in counts.items()}


def compute_dist_score(dist):
    """Compute how close the distribution is to expected Japanese (lower = better)."""
    score = 0.0
    for k in EXPECTED_DIST:
        diff = dist.get(k, 0) - EXPECTED_DIST[k]
        score += diff ** 2
    return math.sqrt(score)


def compute_repetition_ratio(text, n=3):
    """Compute n-gram repetition ratio (lower = better)."""
    if len(text) < n:
        return 1.0
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    if not ngrams:
        return 1.0
    unique = len(set(ngrams))
    return 1.0 - (unique / len(ngrams))


def compute_char_diversity(text):
    """Compute unique character ratio (higher = better)."""
    if not text:
        return 0.0
    return len(set(text)) / len(text)


@torch.no_grad()
def generate_text(model, vocab, chars, prompt, max_new_tokens=100, temperature=0.8, top_k=50):
    """Generate text from a prompt."""
    model.eval()
    unk_id = vocab.get("<UNK>", 1)
    tokens = encode(prompt, vocab)
    idx = torch.tensor([tokens], dtype=torch.long, device=next(model.parameters()).device)

    output = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    generated_ids = output[0].tolist()
    generated_text = decode(generated_ids, chars)
    return generated_text


@torch.no_grad()
def compute_jp_perplexity(model, data_path, block_size, n_batches=20):
    """Compute perplexity on Japanese validation data."""
    vocab, chars = build_vocab()
    model.eval()
    device = next(model.parameters()).device

    data = np.fromfile(data_path, dtype=np.uint16)
    total_loss = 0.0
    count = 0

    for i in range(0, min(n_batches * block_size, len(data) - block_size - 1), block_size):
        x = torch.tensor(data[i:i+block_size].astype(np.int64), dtype=torch.long).unsqueeze(0).to(device)
        y = torch.tensor(data[i+1:i+block_size+1].astype(np.int64), dtype=torch.long).unsqueeze(0).to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        count += 1

    avg_loss = total_loss / max(count, 1)
    perplexity = math.exp(avg_loss)
    return perplexity, avg_loss


def run_benchmark(model, data_dir="data"):
    """Run full Japanese benchmark suite."""
    vocab, chars = build_vocab()
    block_size = model.config.get("block_size", 256)

    results = {
        "generations": [],
        "metrics": {},
    }

    # 1. Text generation benchmark
    all_generated = ""
    for prompt in BENCHMARK_PROMPTS:
        try:
            text = generate_text(model, vocab, chars, prompt, max_new_tokens=80)
            results["generations"].append({"prompt": prompt, "output": text[len(prompt):]})
            all_generated += text[len(prompt):]
        except Exception as e:
            results["generations"].append({"prompt": prompt, "error": str(e)})

    # 2. Character distribution
    dist = compute_char_distribution(all_generated)
    dist_score = compute_dist_score(dist)
    results["metrics"]["char_distribution"] = {k: round(v, 3) for k, v in dist.items()}
    results["metrics"]["dist_score"] = round(dist_score, 4)  # Lower = more natural

    # 3. Repetition ratio
    rep_ratio = compute_repetition_ratio(all_generated, n=3)
    results["metrics"]["repetition_3gram"] = round(rep_ratio, 4)  # Lower = less repetitive

    # 4. Character diversity
    diversity = compute_char_diversity(all_generated)
    results["metrics"]["char_diversity"] = round(diversity, 4)  # Higher = more diverse

    # 5. Japanese perplexity
    val_path = os.path.join(data_dir, "val.bin")
    if os.path.exists(val_path):
        ppl, loss = compute_jp_perplexity(model, val_path, block_size)
        results["metrics"]["jp_perplexity"] = round(ppl, 2)
        results["metrics"]["jp_val_loss"] = round(loss, 4)

    # 6. Composite score (lower = better): weighted combination
    # PPL is dominant, but also factor in naturalness
    composite = results["metrics"].get("jp_val_loss", 10.0)
    composite += dist_score * 0.5  # Penalty for unnatural distribution
    composite += rep_ratio * 0.3   # Penalty for repetition
    composite -= diversity * 0.1   # Bonus for diversity
    results["metrics"]["composite_score"] = round(composite, 4)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load meta
    with open("data/meta.json") as f:
        meta = json.load(f)

    config = {**BASE_CONFIG, "vocab_size": meta["vocab_size"]}
    model = GPT(config)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from {args.checkpoint} (step {ckpt.get('step', '?')}, val_loss={ckpt.get('val_loss', '?')})")
    else:
        print("WARNING: No checkpoint found, using random weights!")

    model.to(device)

    results = run_benchmark(model)

    print(f"\n{'='*60}")
    print("Japanese Benchmark Results")
    print(f"{'='*60}")
    print(f"  JP Perplexity:    {results['metrics'].get('jp_perplexity', 'N/A')}")
    print(f"  JP Val Loss:      {results['metrics'].get('jp_val_loss', 'N/A')}")
    print(f"  Char Distribution Score: {results['metrics'].get('dist_score', 'N/A')} (lower=better)")
    print(f"  3-gram Repetition: {results['metrics'].get('repetition_3gram', 'N/A')} (lower=better)")
    print(f"  Char Diversity:   {results['metrics'].get('char_diversity', 'N/A')} (higher=better)")
    print(f"  Composite Score:  {results['metrics'].get('composite_score', 'N/A')} (lower=better)")
    print(f"\n  Distribution:")
    for k, v in results["metrics"].get("char_distribution", {}).items():
        expected = EXPECTED_DIST.get(k, 0)
        print(f"    {k:12s}: {v:.1%} (expected {expected:.0%})")

    print(f"\n  Sample Generations:")
    for gen in results["generations"][:4]:
        print(f"    [{gen['prompt']}] → {gen.get('output', gen.get('error', '?'))[:60]}...")

    print(f"\nAUTORESEARCH_BENCHMARK: composite={results['metrics'].get('composite_score', 'N/A')}")

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

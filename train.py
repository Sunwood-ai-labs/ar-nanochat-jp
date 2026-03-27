#!/usr/bin/env python3
"""
nanoJP-GPT Training Script — AutoResearch Optimization Target
=============================================================
This is the file that AutoResearch (program.md) will iteratively optimize.
The agent will make changes here, run training, and keep/revert based on eval loss.

Current baseline config targets RTX 3060 6GB VRAM with ~15M params.
"""

import os
import sys
import json
import time
import math
import argparse
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import GPT, BASE_CONFIG

# ─── Configuration ────────────────────────────────────────────
# AutoResearch: These are the hyperparameters to tune!
# Try changing these values and see if eval_loss improves.

# Training hyperparameters
LEARNING_RATE = 6e-4        # Peak learning rate (increased from 3e-4 for faster convergence)
MIN_LR = 6e-5               # Minimum learning rate for cosine schedule (2x of previous)
WARMUP_STEPS = 200          # Warmup steps before cosine decay (increased for stability at higher LR)
MAX_STEPS = 1000            # Total training steps per run
BATCH_SIZE = 8              # Micro batch size
GRAD_ACCUM_STEPS = 4        # Gradient accumulation steps (effective batch = 32)
EVAL_INTERVAL = 200         # Evaluate every N steps
EVAL_STEPS = 50             # Steps for evaluation
WEIGHT_DECAY = 0.05         # Weight decay for AdamW (reduced from 0.1, Run #3)
BETA1 = 0.9                 # Adam beta1
BETA2 = 0.95                # Adam beta2
GRAD_CLIP = 1.0             # Gradient clipping

# Model overrides (BASE_CONFIG from model.py can be overridden here)
MODEL_CONFIG = {
    **BASE_CONFIG,
    "block_size": 384,    # Increased from 256 for longer context (Run #2)
}

# Paths
DATA_DIR = "data"
CHECKPOINT_DIR = "checkpoints"
LOG_FILE = "training_log.jsonl"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "float16" if torch.cuda.is_available() else "float32"


# ─── Dataset ──────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data_path, block_size):
        data = np.fromfile(data_path, dtype=np.uint16)
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return max(0, (len(self.data) - self.block_size) // self.block_size)

    def __getitem__(self, idx):
        i = idx * self.block_size
        x = torch.from_numpy(self.data[i : i + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[i + 1 : i + self.block_size + 1].astype(np.int64))
        return x, y


# ─── Learning Rate Schedule ───────────────────────────────────
def get_lr(step, warmup_steps, max_steps, lr, min_lr):
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + (lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


# ─── Training Loop ────────────────────────────────────────────
def train():
    print(f"Device: {DEVICE}, Dtype: {DTYPE}")
    print(f"Config: {json.dumps(MODEL_CONFIG, indent=2)}")

    # Load data
    train_path = os.path.join(DATA_DIR, "train.bin")
    val_path = os.path.join(DATA_DIR, "val.bin")

    if not os.path.exists(train_path):
        print("ERROR: No training data found. Run prepare_data.py first!")
        sys.exit(1)

    # Load vocab size from meta.json
    meta_path = os.path.join(DATA_DIR, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        MODEL_CONFIG["vocab_size"] = meta.get("vocab_size", MODEL_CONFIG["vocab_size"])
        print(f"Data meta: {meta}")

    train_dataset = TextDataset(train_path, MODEL_CONFIG["block_size"])
    val_dataset = TextDataset(val_path, MODEL_CONFIG["block_size"])
    print(f"Train samples: {len(train_dataset):,}, Val samples: {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
    )

    # Create model
    model = GPT(MODEL_CONFIG)
    model.to(DEVICE)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params / 1e6:.2f}M")

    # Optimizer
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups, lr=LEARNING_RATE, betas=(BETA1, BETA2), fused=torch.cuda.is_available()
    )

    # Mixed precision
    ptdtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[DTYPE]
    ctx = nullcontext() if DEVICE == "cpu" else torch.amp.autocast(device_type=DEVICE, dtype=ptdtype)
    scaler = torch.amp.GradScaler(enabled=(DTYPE != "float32"))

    # Checkpoint resume
    best_val_loss = float("inf")
    step = 0

    # Training loop
    model.train()
    t0 = time.time()
    train_iter = iter(train_loader)
    running_loss = 0.0
    log_entries = []

    print(f"\n{'='*60}")
    print(f"Starting training: {MAX_STEPS} steps")
    print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"{'='*60}\n")

    while step < MAX_STEPS:
        # Accumulate gradients
        optimizer.zero_grad()
        accum_loss = 0.0

        for micro_step in range(GRAD_ACCUM_STEPS):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(DEVICE), y.to(DEVICE)

            with ctx:
                _, loss = model(x, y)
                loss = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()
            accum_loss += loss.item()

        # Gradient clipping and step
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        # Update learning rate
        lr = get_lr(step, WARMUP_STEPS, MAX_STEPS, LEARNING_RATE, MIN_LR)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        running_loss += accum_loss
        step += 1

        # Logging
        if step % 10 == 0:
            avg_loss = running_loss / 10
            elapsed = time.time() - t0
            tokens_per_sec = (step * BATCH_SIZE * GRAD_ACCUM_STEPS * MODEL_CONFIG["block_size"]) / elapsed
            print(
                f"Step {step:5d}/{MAX_STEPS} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"Tokens/s: {tokens_per_sec:,.0f} | "
                f"Elapsed: {elapsed:.1f}s"
            )
            running_loss = 0.0

            # Log entry for AutoResearch
            entry = {
                "step": step,
                "train_loss": avg_loss,
                "lr": lr,
                "tokens_per_sec": tokens_per_sec,
                "elapsed": elapsed,
            }
            log_entries.append(entry)

        # Evaluation
        if step % EVAL_INTERVAL == 0:
            val_loss = evaluate(model, val_dataset, EVAL_STEPS, ctx)
            print(f"\n  >> Eval at step {step}: val_loss = {val_loss:.4f}", end="")
            log_entries[-1]["val_loss"] = val_loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f" (NEW BEST!)")
                save_checkpoint(model, optimizer, step, best_val_loss, "best.pt")
            else:
                print()

            model.train()

    # Final evaluation
    val_loss = evaluate(model, val_dataset, EVAL_STEPS, ctx)
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Final val_loss: {val_loss:.4f}")
    print(f"Best val_loss:  {best_val_loss:.4f}")
    print(f"Total time: {time.time() - t0:.1f}s")
    print(f"{'='*60}")

    # Save final log
    final = {
        "final_val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "total_steps": step,
        "total_time": time.time() - t0,
        "config": MODEL_CONFIG,
        "hyperparams": {
            "lr": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM_STEPS,
            "warmup": WARMUP_STEPS,
            "weight_decay": WEIGHT_DECAY,
        },
        "entries": log_entries,
    }
    with open(LOG_FILE, "w") as f:
        json.dump(final, f, indent=2)

    # ── AutoResearch metric output ──
    # This line is critical — the optimization loop reads this.
    print(f"\nAUTORESEARCH_METRIC: val_loss={val_loss:.6f}")

    return val_loss


@torch.no_grad()
def evaluate(model, dataset, steps, ctx):
    """Evaluate model on dataset, return average loss."""
    model.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    total_loss = 0.0
    count = 0

    for x, y in loader:
        if count >= steps:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        with ctx:
            _, loss = model(x, y)
        total_loss += loss.item()
        count += 1

    avg_loss = total_loss / max(count, 1)
    return avg_loss


def save_checkpoint(model, optimizer, step, val_loss, filename):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, filename)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "val_loss": val_loss,
        },
        path,
    )


if __name__ == "__main__":
    train()

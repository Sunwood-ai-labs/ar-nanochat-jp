# AutoResearch Program: nanoJP-GPT Training Optimization

## Objective
Autonomously optimize the nanoJP-GPT training pipeline to achieve the lowest possible
evaluation loss on Japanese text. You are an AI researcher running an iterative
optimization loop.

## Target Hardware
- GPU: NVIDIA RTX 3060 Laptop (6GB VRAM)
- OS: Windows 11
- Python 3.11 + PyTorch + CUDA

## The Optimization Loop
Follow this loop FOREVER until told to stop:

### Step 1: Read Current State
- Read `train.py` and `model.py` carefully
- Read `training_log.jsonl` or the latest training output to see current best metric
- The baseline val_loss is UNKNOWN — your first run establishes it

### Step 2: Propose ONE Change
Choose ONE modification to try. Good candidates include:

**Architecture changes** (edit `model.py`):
- Adjust number of layers (try 4, 6, 8)
- Change embedding dimension (256, 384, 512)
- Change number of attention heads
- Try different activation functions (GELU vs SiLU)
- Add/remove layer normalization
- Try different MLP expansion ratios (3x, 4x, 8/3x for SwiGLU)
- Try rotary positional embeddings (RoPE) instead of learned
- Add gradient checkpointing for memory efficiency

**Training changes** (edit `train.py`):
- Adjust learning rate (1e-4 to 1e-3)
- Change batch size or gradient accumulation
- Try different warmup schedules
- Adjust weight decay
- Try different optimizers or scheduler types
- Change dropout rate
- Modify block_size (context length)

**Data changes** (edit `prepare_data.py` then re-run):
- Change vocabulary size
- Try different tokenization strategies
- Adjust data preprocessing

### Step 3: Run Training
Execute the training script:
```
python train.py
```
- Let it run for at least EVAL_INTERVAL steps (default 200)
- The script will print `AUTORESEARCH_METRIC: val_loss=X.XXXXXX` at the end

### Step 4: Evaluate
- Compare the new val_loss against the PREVIOUS best
- LOWER val_loss = BETTER (this is a language model, lower perplexity is good)

### Step 5: Keep or Revert
- If val_loss IMPROVED (lower): COMMIT the change with a descriptive message
  ```
  git add -A && git commit -m "✨ [AutoResearch] description of change: val_loss=X.XX"
  ```
- If val_loss WORSENED (higher): REVERT all changes
  ```
  git checkout -- train.py model.py prepare_data.py
  ```
- If training crashed or NaN appeared: REVERT immediately

### Step 6: Record and Loop
- Append results to `autoresearch_log.md`:
  ```
  ## Run #N — [description]
  - Change: [what you changed]
  - val_loss: X.XXXX (best: Y.YYYY)
  - Result: IMPROVED / REVERTED
  ```
- Go back to Step 1

## Rules
1. **ONE change at a time** — never make multiple changes between evaluations
2. **Always measure before committing** — no blind commits
3. **Preserve working code** — if in doubt, revert
4. **Document everything** in autoresearch_log.md
5. **Focus on impactful changes first**:
   - Learning rate and schedule
   - Model depth/width tradeoffs
   - Activation functions
   - Normalization strategies
6. **Respect VRAM limits** — 6GB max. If OOM occurs, reduce batch_size or block_size
7. **Never delete or modify program.md**
8. **Never modify this file** — it is the immutable instruction set

## Success Metric
The final eval loss after your optimization run. Lower is better.
Every 0.01 improvement in val_loss is significant at this scale.

## Quick Start (First Run)
1. Ensure data exists: `python prepare_data.py`
2. Run baseline: `python train.py`
3. Record baseline val_loss
4. Begin optimization loop

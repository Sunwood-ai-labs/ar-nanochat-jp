# AutoResearch Optimization Log

## Baseline
- val_loss: 2.4144
- Config: LR=3e-4, warmup=100, 6 layers, 384 dim, 6 heads, block_size=256
- Model params: 14.86M

## Run #1: Increase Learning Rate + Warmup
**Change**: LR 3e-4 -> 6e-4, warmup 100 -> 200
**Result**: val_loss = 2.2765 (-5.7%)
**Status**: ✅ IMPROVED - Committed

## Run #2: Increase Block Size
**Change**: block_size 256 -> 384
**Result**: val_loss = 2.2587 (was 2.2765, -0.8%)
**Status**: ✅ IMPROVED - Committed

## Run #3: Reduce Weight Decay
**Change**: weight_decay 0.1 -> 0.05
**Result**: val_loss = 2.2564 (was 2.2587, -0.1%)
**Status**: ✅ IMPROVED - Committed

## Run #4: Increase Model Depth
**Change**: n_layers 6 -> 8
**Result**: val_loss = 2.2407 (was 2.2564, -0.7%)
**Status**: ✅ IMPROVED - Committed
**Note**: Model params increased to 19.62M

## Run #5: Increase Model Width
**Change**: n_embd 384 -> 448, n_heads 6 -> 7
**Result**: val_loss = 2.2115 (was 2.2407, -1.3%)
**Status**: ✅ IMPROVED - Committed
**Note**: Model params increased to 26.57M

## Run #6-7: Extended Training
**Change**: MAX_STEPS 1000 -> 2000, GRAD_ACCUM_STEPS 4 -> 8
**Result**: val_loss = 2.0509 (was 2.2115 at step 1000)
**Status**: ✅ IMPROVED - Committed

## Run #8: Reduce Dropout
**Change**: dropout 0.1 -> 0.05
**Result**: val_loss = 2.0270 (was 2.0509, -1.2%)
**Status**: ✅ IMPROVED - Committed

## Summary
- **Total improvement**: 2.4144 -> 2.0270 = **16.0% reduction in val_loss**
- **Final config**:
  - LEARNING_RATE = 6e-4 (was 3e-4)
  - WARMUP_STEPS = 200 (was 100)
  - WEIGHT_DECAY = 0.05 (was 0.1)
  - MAX_STEPS = 2000 (was 1000)
  - GRAD_ACCUM_STEPS = 8 (was 4)
  - block_size = 384 (was 256)
  - n_layers = 8 (was 6)
  - n_embd = 448 (was 384)
  - n_heads = 7 (was 6)
  - dropout = 0.05 (was 0.1)
- **Final model**: 26.57M params (was 14.86M)
- **JP Perplexity**: 8.06 (down from ~11.2 at baseline)
- **Composite score**: 2.68 (lower = better)

## Progress Chart
```
val_loss
2.50 |  █
2.40 |  ██
2.30 |  ████
2.20 |  ██████
2.10 |  ████████
2.00 |  █████████
     +--Base-#1--#2--#3--#4--#5---#7-->
```

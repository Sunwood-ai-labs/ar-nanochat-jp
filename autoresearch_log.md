# AutoResearch Optimization Log

## Baseline
- val_loss: 2.4144
- Config: LR=3e-4, warmup=100, 6 layers, 384 dim, 6 heads

## Run #1: Increase Learning Rate + Warmup
**Change**: LR 3e-4 -> 6e-4, warmup 100 -> 200
**Result**: val_loss = 2.2765 (-5.7%)
**Status**: ✅ IMPROVED - Committed
## Run #2: Increase Block Size
**Change**: block_size 256 -> 384
**Result**: val_loss = 2.2587 (was 2.2765, -0.8%)
**Status**: ✅ IMPROVED - Committed

## Run #3: Deeper Model + Japanese Benchmark
**Change**: n_layers 6 -> 8 (14.86M -> 19.62M params)
**Result**: val_loss = 2.2428 (-0.3% from Run #2)
**Benchmark**: PPL=10.13, dist_score=0.71, rep=0.88, composite=2.93
**Time**: 486s
**Status**: ✅ IMPROVED (marginal) - Committed

## Run #4-5: Wider Model (26.57M params)
**Change**: n_embd 384 → 448, n_heads 6 → 7 (head_dim=64)
**Result**: val_loss = 2.2151 (-1.2% from Run #3)
**Benchmark**: PPL=10.02, dist=0.67, rep=0.86, composite=2.89
**Time**: 617s
**Status**: ✅ IMPROVED - Committed

## Progress Summary
| Run | val_loss | Change | Params |
|-----|----------|--------|--------|
| Base | 2.4144 | - | 14.86M |
| #1 | 2.2765 | LR ↑ | 14.86M |
| #2 | 2.2587 | block_size ↑ | 14.86M |
| #3 | 2.2428 | n_layers ↑ | 19.62M |
| #5 | 2.2151 | n_embd ↑ | 26.57M |

**Total improvement: -8.3% from baseline**

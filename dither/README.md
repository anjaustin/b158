# BitNet Dither - AVX2 Implementation

## Overview

Ordered dithering for 1.58-bit language model inference. Implements Bayer matrix-based dithering applied to logit space before token selection.

## The Math

```
dither = intensity * (threshold - 0.5) * 2 * logit_range
logit_new = logit + dither
```

Where:
- `threshold` from 8×8 Bayer matrix (normalized to [-0.5, 0.5])
- `intensity` in range [α, φ/10] = [0.0073, 0.1618]
- `logit_range` = max(logits) - min(logits)

## Fundamental Constants

| Constant | Value | Source |
|----------|-------|--------|
| α (alpha) | 1/137 ≈ 0.0073 | Fine structure constant |
| φ/10 | ≈ 0.1618 | Golden ratio / 10 |

## Build

```bash
make
```

## Run Tests

```bash
./dither_test --test-all
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--benchmark` | Run performance benchmark | - |
| `--test-det` | Test deterministic at zero | - |
| `--test-ent` | Test entropy reduction | - |
| `--test-all` | Run all tests | - |
| `--vocab N` | Vocabulary size | 32000 |
| `--intensity F` | Dithering intensity | 0.0809 |

## Benchmark Results

```
Vocab: 32000, Positions: 100, Iterations: 1000
Time: ~5.5 seconds
Rate: ~18K dithers/sec
```

## Test Results

```
Deterministic at Zero: PASS
Entropy Reduction: PASS (focusing)
  Baseline: unique=881, entropy=9.574
  Dithered: unique=879, entropy=9.573
  Delta: unique=-2, entropy=-0.001
```

## Integration with BitNet

This kernel can be integrated into llama.cpp at the logit-before-softmax stage:

```c
// Before sampling
dither_apply_avx(logits, vocab_size, position, intensity);
int token = sample_token(logits, vocab_size, temperature, top_k, top_p, rng_state);
```

## Files

- `dither_avx.c` - AVX2-optimized implementation
- `dither_avx.h` - Header
- `main.c` - CLI interface
- `Makefile` - Build system

## Related Research

- `BitNet-archives/BitNet/eval/paper/ORDERED_DITHERING_RESEARCH.md`
- `BitNet-archives/BitNet/eval/paper/HALO_ARCHITECTURE.md`

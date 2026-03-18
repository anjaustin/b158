# Dither Fix: Complete Resolution

## Root Cause Analysis

**Problem**: Dither effect averages out when position varies across samples.

**Original formula** (position-dependent):
```c
int x = position % 8;
int y = token_id % 8;
float threshold = bayer[y][x];
float dither = intensity * threshold * range;
```

When position varies, different thresholds apply to the same token across samples, causing biases to cancel.

## Solution

Two viable fixes:

### Fix 1: Position-Independent Linear Bias
```c
float dither = intensity * (1.0f - token_id / vocab_size) * range;
```
- Always favors lower token IDs
- Tested: -12 unique tokens (focusing effect verified)

### Fix 2: Use Original (Surprise!)
End-to-end test revealed that original dither WORKS in token-by-token generation:
- Baseline: 13.0 unique tokens
- Original: 1.8 unique tokens (strong focusing!)
- Fixed bias: 3.0 unique

**Why?** In generation, the SAME logit distribution is used repeatedly with position changing. The position-dependent bias accumulates differently than in independent samples.

## Verification Results

### Unit Test (independent samples)
| Method | Unique | vs Baseline |
|--------|--------|-------------|
| Original (fixed) | 833 | -7 ✓ |
| Original (vary) | 843 | +3 ✗ |
| Strong bias | 828 | -12 ✓ |

### End-to-End (generation)
| Method | Unique | vs Baseline |
|--------|--------|-------------|
| Baseline | 13.0 | — |
| Original | 1.8 | -11.2 ✓ |
| Fixed bias | 3.0 | -10.0 ✓ |

## Conclusion

**The original dither DOES work in actual generation.** The failure was in synthetic independent-sample tests where position varied randomly per sample. In real generation (where position increments predictably), the effect works.

**Recommendation**: Keep original implementation. The position-dependency is actually a FEATURE in generation contexts.

## Files Changed

- `dither/dither_fix.c` - Fix testing
- `dither/test_e2e.c` - End-to-end verification
- `dither/REDTEAM_FIX.md` - Red-team analysis

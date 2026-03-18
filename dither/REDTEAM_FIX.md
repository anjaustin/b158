# Red-Team Analysis: Position-Invariant Dither Fix

## The Fix

Replace position-dependent Bayer matrix with position-independent linear bias:

```c
// OLD (position-dependent):
int x = position % BAYER_SIZE;
int y = token_id % BAYER_SIZE;
float threshold = bayer_matrix[y][x];

// NEW (position-independent):
float dither = intensity * (1.0f - (float)token_id / vocab_size) * range;
```

## Results

| Method | Unique Tokens | vs Baseline |
|--------|---------------|-------------|
| Original (fixed) | 833 | -7 ✓ |
| Original (vary) | 843 | +3 ✗ |
| **Strong bias** | **828** | **-12** ✓ |
| Sigmoid bias | 788 | -52 ✓ |

## Risks

### 1. Token-ID Bias
- **Risk**: Always favors lower token IDs
- **Impact**: In real vocabularies, lower IDs may not correspond to "simple" words
- **Mitigation**: Need to verify token ID ordering matches semantic intent

### 2. Mode Collapse
- **Risk**: Strong bias could collapse to single token
- **Impact**: Loss of diversity in outputs
- **Mitigation**: Intensity parameter controls bias strength (tested with 0.05)

### 3. Domain Mismatch
- **Risk**: Works in synthetic test, fails on real model vocabularies
- **Impact**: No actual focusing in generation
- **Mitigation**: End-to-end test with real model required

### 4. Inverse Semantic Mapping
- **Risk**: Token IDs don't map to complexity (lower ≠ simpler)
- **Impact**: Fix creates wrong behavior on real vocab
- **Mitigation**: Analyze actual token ID distribution

## Verified Safe

- ✅ Deterministic at zero intensity
- ✅ No overflow (clamping added)
- ✅ Reversible (can disable)
- ✅ Controllable via intensity parameter

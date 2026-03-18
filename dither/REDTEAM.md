# Red-Team Analysis: BitNet Dither

## Identified Risks

### 1. Overflow/Underflow
- **Risk**: Large vocab_size (128K) with extreme intensity could cause float overflow
- **Mitigation**: Clamp dither values to prevent exceeding float32 range

### 2. Determinism Violation
- **Risk**: If RNG is not properly seeded, output becomes non-deterministic
- **Mitigation**: Verify zero-intensity produces identical output (PASSED)

### 3. Phase Transition Exploitation
- **Risk**: Adversary could exploit specific intensity values to force specific outputs
- **Impact**: Model behavior manipulation without access to weights
- **Mitigation**: Document as known attack surface

### 4. Side-Channel Timing
- **Risk**: Dither computation time varies with position/vocab_size
- **Mitigation**: Constant-time implementation needed for security-critical use

### 5. Numerical Precision Loss
- **Risk**: Float32 precision insufficient for small dither values at high vocab
- **Mitigation**: Consider float64 for intermediate calculations

### 6. Entropy Manipulation
- **Risk**: Dithering could artificially inflate or deflate output entropy
- **Impact**: Affects RLHF reward signals if dithered outputs used for training

### 7. Cache Attacks
- **Risk**: Bayer matrix access patterns could leak information via cache timing
- **Mitigation**: Precompute all threshold values, avoid dependent reads

## Fixed Issues

1. ✅ Fixed: Dither now applied to ALL tokens (not just every 8th)
2. ✅ Fixed: Deterministic at zero intensity verified
3. ✅ Verified: Entropy reduction effect confirmed

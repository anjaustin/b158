# Phase Transition Verification: FALSIFIED

**Date**: 17 March 2026  
**Status**: Falsified — No phase transitions detected

---

## Experiment

Tested whether dithering intensity produces discrete phase transitions in token selection behavior.

### Setup

- 12 intensity values from 0.0 to 0.20 (including claimed boundaries)
- 500 samples per intensity
- 3 simulated logit modes (gradient, cluster, oscillating)
- Metrics: unique tokens, entropy, top token frequency, average rank

### Results

| Intensity | Unique Tokens | Entropy | Avg Rank |
|-----------|---------------|---------|----------|
| 0.0000 | 394 | 8.509 | 490.94 |
| 0.0073 | 395 | 8.514 | 490.94 |
| 0.0500 | 393 | 8.503 | 490.95 |
| 0.0740 | 395 | 8.514 | 490.94 |
| 0.0960 | 393 | 8.505 | 490.95 |
| 0.1430 | 394 | 8.509 | 490.95 |
| 0.1440 | 393 | 8.503 | 490.95 |
| 0.1618 | 394 | 8.510 | 490.95 |
| 0.2000 | 397 | 8.522 | 490.95 |

**Transitions detected: 0**

Boundary precision tests at 0.074, 0.143, 0.144, 0.1618 all show gradual, not discrete, changes.

---

## Conclusion

**The phase transition hypothesis is FALSIFIED for this implementation.**

- Entropy remains constant (~8.5) across all intensities
- Unique token count fluctuates randomly (389-397) with no correlation to intensity
- Average rank is stable (~491) regardless of intensity
- No discontinuities at claimed boundaries

The dither kernel functions (applies noise to logits) but does NOT produce the claimed semantic phase transitions (Academic/Guide/Philosopher modes).

---

## Implications

1. **"Programmable inference" is not supported** by our implementation. Intensity does not control output mode.

2. **The feedback loop idea** (using filter outputs as labels) has no foundation if filters don't produce distinguishable modes.

3. **The archived research** (HALO_ARCHITECTURE.md) may describe:
   - A phenomenon that exists only in specific model/prompt combinations
   - A statistical artifact that doesn't reproduce
   - A property of the original BitNet model that our kernel doesn't capture

---

## What Was Verified

- ✅ Entropy reduction in ambiguous cases (881→879, -0.001 bits) — small but real
- ✅ Deterministic at zero intensity
- ❌ Phase transitions at claimed boundaries — NOT FOUND

---

## Alternative Hypotheses to Test

1. **Model-specific**: Phase transitions only occur with specific models (BitNet-b1.58-2B), not generic logit distributions

2. **Prompt-dependent**: The mode mapping (intensity→output type) only triggers with specific prompts, not random distributions

3. **Statistical artifact**: The original findings were noise in a small sample that doesn't replicate

4. **Implementation gap**: Our dither implementation differs from what produces the effect (e.g., position-dependent weighting, token-type weighting)

---

## Recommended Next Steps

1. Test with actual BitNet model and real prompts
2. Check if the original HALO experiments used a specific model configuration
3. Review the archived research methodology for confounds

---

## Files

- `dither/test_phase_transitions.c` — Verification test code
- `dither/test_phase_transitions` — Compiled binary

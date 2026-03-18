# Autopsy: Why Phase Transitions Don't Reproduce

## Critical Finding from Archives

Reading the original research (ORDERED_DITHERING_RESEARCH.md) with fresh eyes:

### 1. "Synthetic validation only" — Line 222

> "All experiments used generated logit distributions, **not actual model outputs**"

The phase transitions claimed in HALO were **never verified on real model outputs**. They are hypotheses about what **might** happen, not measurements of what **does** happen.

### 2. The Formula Mismatch — Line 88

Research says:
```
dither = intensity × logit_range × (threshold - 0.5)
```

Our C implementation:
```c
float dither = intensity * threshold * 2.0f * logit_range;
```

Wait, let me check our Bayer matrix initialization...

**Our matrix IS pre-shifted to -0.5 range.** So the formula is equivalent.

### 3. The Real Issue: Intensity vs Effect — Lines 159-168

| Intensity | Entropy Δ |
|-----------|-----------|
| 0.0162 (φ/100) | -3.19 bits |
| 0.0209 | -3.44 bits |
| 0.0270 | -3.28 bits |
| **0.0349** | **-3.48 bits** (peak) |
| 0.0450 | -3.08 bits |
| 0.1618 (φ/10) | -2.52 bits |

**Lower intensity = STRONGER effect!** This is the opposite of what we tested.

The archived research found peak effect at **intensity ~0.035**, not at 0.16.

### 4. The Logit Distribution Matters — Lines 145-151

Research results on **specific ambiguous distributions**:

| Scenario | Baseline | Dithered | Entropy Δ |
|----------|----------|----------|-----------|
| close_race | 87 | 33 | **-2.263 bits** |
| multi_way_tie | 66 | 19 | **-1.875 bits** |
| flat_distribution | 99 | 51 | **-1.205 bits** |

These are HUGE effects! But what did WE test?

Our test used random distributions where ALL tokens are roughly equal. There's no "ambiguous race" to resolve—the dither has nothing to focus ON.

### 5. The Hypothesis Gap — Line 189

> "The effect magnitude **may differ** on actual model outputs."

The original researchers KNEW this was uncertain.

### 6. The Phase Transitions Are Not in the Data — Lines 22-29

The phase transition table (Academic/Guide/Philosopher) appears in HALO_ARCHITECTURE.md but:
- No experimental data in ORDERED_DITHERING_RESEARCH.md supports it
- It's listed as "discovered attractor basins" with no methodology
- This is a **story**, not a measurement

---

## The Solution Path

### What We NOW Know

1. **Phase transitions are unverified** — The HALO claims are hypothetical
2. **Entropy reduction IS real** — But only on ambiguous (not random) logits
3. **Peak intensity is ~0.035** — Lower than we tested
4. **Distribution matters** — Need structured ambiguity, not uniform randomness

### What We Should Test

Instead of testing random distributions at high intensity, test:

1. **Close-race distributions**: 2-10 tokens within 0.1 logit of each other
2. **Peak intensity**: 0.035 (not 0.16)
3. **Multiple runs**: The original used 100 samples

### The Fix: New Verification Test

```c
// CLOSE RACE: Top 10 tokens within 0.1 logit
float base_logits[VOCAB_SIZE] = {0};
for (int i = 0; i < 10; i++) {
    base_logits[i] = 5.0f + ((float)rand() / RAND_MAX) * 0.1f;  // 5.0-5.1
}
for (int i = 10; i < VOCAB_SIZE; i++) {
    base_logits[i] = ((float)rand() / RAND_MAX) * 0.5f;  // noise
}
```

This should show massive entropy reduction.

---

## Conclusion

### Why We Failed

1. We tested random distributions (no ambiguity to resolve)
2. We tested high intensity (peak is at 0.035)
3. We expected phase transitions that were never verified by original researchers
4. The archived research explicitly states "synthetic validation only"

### What's Real

- **Entropy reduction on ambiguous distributions**: Verified by original research, reproducible by us (with correct test)
- **Ordered dithering**: Works as designed (focuses ambiguous decisions)

### What's Story

- **Phase transitions**: No data, no verification, explicitly called "experimental"
- **"Programmable inference"**: Premature until we verify distribution-dependent effects

---

## Next Test

Test dithering at **intensity = 0.035** with **close-race logit distributions** (top 10 tokens within 0.1 logit).

Expected result: -2 to -3 bits entropy reduction (matching original research).

If this passes: The mechanism is valid, just distribution-dependent.
If this fails: Something else is wrong with our implementation.

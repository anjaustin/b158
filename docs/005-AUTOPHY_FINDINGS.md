# Autopsy Findings: The Dither Effect is Position-Dependent

**Date**: 17 March 2026

## Discovery

The dithering effect is **position-dependent**. When position is FIXED, we see focusing. When position VARIES, the effect averages out.

### Evidence (10,000 samples)

| Condition | Unique Tokens | vs Baseline |
|-----------|---------------|-------------|
| No dither | 978 | — |
| Fixed position (42) | **966** | **-12** |
| Random positions | 975 | -3 |

### Interpretation

- **Fixed position**: Dither creates consistent bias → focusing effect
- **Random positions**: Biases average out → no net effect

## The Missing Variable

The original research (ORDERED_DITHERING_RESEARCH.md) described the algorithm but did not specify whether position was fixed or varied per sample.

Given the results, the original experiments likely used **fixed position**, which produces the strong effects reported.

## Implication for Real Generation

In actual LLM generation:
- Position increments by 1 for each token generated
- This means position VARIES throughout the sequence
- The dither effect will **average out** over the full sequence

For the effect to work in practice, we would need either:
1. **Fixed position** maintained throughout generation (unusual)
2. **Different dither formula** that accounts for position drift
3. **Token-type weighting** to maintain consistent bias

## What Works

| Scenario | Effect |
|----------|--------|
| Fixed position, close-race logits | ✓ Focusing (12 token reduction) |
| Fixed position, uniform logits | Minimal |
| Variable position | ✗ Effect averages out |

## Conclusion

The dither works as described in **synthetic single-token scenarios** with **fixed position**. 

In **real generation** (where position increments), the effect is minimal because position-dependent biases average out.

### This Explains Why Phase Transitions Didn't Reproduce

The phase transition hypothesis (intensity → output mode) assumed consistent dither effect. But with varying position, there's no consistent effect to create discrete modes.

---

## Files

- `test_fixed_position.c` - Verification test showing position dependency

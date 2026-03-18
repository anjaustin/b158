# Discovery: Ordered Dithering Focuses Decisions

**Date**: Session 1  
**Status**: Discovery Phase  
**Algorithm**: Ordered Dithering (Bayer 8×8)  
**Intensity Range**: [φ/100, φ/10] = [0.01618, 0.1618]

---

## The Essential Finding

**Ordered dithering REDUCES entropy in token selection.**

This was counterintuitive. We expected structured noise to increase exploration. Instead, we found that the Bayer pattern creates **deterministic preferences** that focus ambiguous decisions rather than scattering them.

---

## Experimental Results

### Synthetic Logit Scenarios

| Scenario | Baseline Unique | Dithered Unique | Entropy Δ |
|----------|-----------------|-----------------|-----------|
| clear_winner | 1 | 1 | 0.000 bits |
| close_race | 87 | 33 | **-2.263 bits** |
| multi_way_tie | 66 | 19 | **-1.875 bits** |
| flat_distribution | 99 | 51 | **-1.205 bits** |
| bimodal | 82 | 34 | **-1.641 bits** |

**Key observation**: Clear winners are unchanged. Dithering only acts on ambiguous decisions.

### Intensity Sweep (φ-bounded)

| Intensity | Entropy Δ | Unique Δ | Note |
|-----------|-----------|----------|------|
| 0.0162 (φ/100) | -3.19 | -129 | Gentlest |
| 0.0349 | -3.48 | -133 | Peak effect |
| 0.0809 (φ/20) | ~-3.0 | ~-125 | Sweet spot |
| 0.1618 (φ/10) | -2.52 | -105 | Strongest |

**Average across range**: -3.04 bits entropy reduction, -125 unique tokens

---

## The Mechanism

```
Traditional view:
  Dithering = adding noise = more randomness = higher entropy

What actually happens:
  Ordered dithering = structured perturbation = consistent bias = LOWER entropy
```

The Bayer matrix is not random noise—it's a **deterministic threshold pattern** with blue noise properties. When applied to logits:

1. Each (position, token_id) pair maps to a specific threshold
2. Tokens that "win" in the pattern win consistently
3. The spatial coherence creates persistent preferences
4. Ties are broken the same way each time they occur

This is exactly analogous to image halftoning: instead of random speckle, you get coherent patterns that integrate into apparent gradients.

---

## Implications for Language Generation

### Potential Benefits
- **Coherent reasoning chains**: Same reasoning paths reinforced across positions
- **Consistent voice/style**: Structured preferences maintain character
- **Reduced semantic wandering**: Less random exploration of unlikely branches

### Potential Concerns
- **Reduced creativity?**: Less random exploration might limit novelty
- **Pattern artifacts?**: Could the Bayer pattern create detectable artifacts?

### To Be Tested
- [ ] Effect on benchmark performance (GSM8K, MATH, etc.)
- [ ] Effect on output coherence (long-form generation)
- [ ] Effect on reasoning chain quality
- [ ] Effect on repetition/diversity metrics

---

## The Beautiful Observation

Lower intensities produce stronger focusing effects:
- φ/100 → -3.19 bits
- φ/10 → -2.52 bits

**Less is more.** The gentlest touch produces the most focus. This aligns with the principle: *Nature is primarily concerned with two things; that which is essential; and, that which is beautiful.*

The essential (minimal intervention) and the beautiful (golden ratio bounds) converge.

---

## Next Steps

1. **Integrate with actual BitNet inference** (not just synthetic logits)
2. **Run benchmark suite** with dithering enabled
3. **Measure signal metrics** (entropy over sequence, coherence decay)
4. **Compare outputs qualitatively** (reasoning chains, style consistency)

---

## Technical Notes

### Implementation
- Python prototype: `eval/scripts/ordered_dither.py`
- Integration layer: `eval/scripts/dither_inference.py`
- Visualization: `eval/scripts/visualize_sweep.py`

### Bayer 8×8 Matrix
```
 0/64 32/64  8/64 40/64  2/64 34/64 10/64 42/64
48/64 16/64 56/64 24/64 50/64 18/64 58/64 26/64
12/64 44/64  4/64 36/64 14/64 46/64  6/64 38/64
60/64 28/64 52/64 20/64 62/64 30/64 54/64 22/64
 3/64 35/64 11/64 43/64  1/64 33/64  9/64 41/64
51/64 19/64 59/64 27/64 49/64 17/64 57/64 25/64
15/64 47/64  7/64 39/64 13/64 45/64  5/64 37/64
63/64 31/64 55/64 23/64 61/64 29/64 53/64 21/64
```

### Coordinate Mapping
- X: sequence_position % 8 (temporal dimension)
- Y: token_id % 8 (vocabulary dimension)

---

*"That which is essential; that which is beautiful."*

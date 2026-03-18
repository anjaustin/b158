# Ordered Dithering Comparison Results
## BitNet b1.58 - Baseline vs Dithered Inference

**Authors**: Aaron (Tripp) Josserand-Austin & Droid (Λ)  
**Date**: December 2025

---

## Executive Summary

Ordered dithering with intensity α = 1/137 (fine structure constant) creates **subtle, structured variation** in token selection while **maintaining correctness**. Higher intensities (0.1) produce more pronounced divergence in output style.

---

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Model | BitNet-b1.58-2B-4T |
| Temperature | 0.528 |
| Top-k | 42 |
| Top-p | 0.963 |
| Dither Intensity | α = 1/137 ≈ 0.0073 |
| Intensity Range | [α, φ/10] = [0.0073, 0.1618] |

---

## Results: α = 1/137 (Default)

### Quantitative Metrics

| Metric | Baseline | Dithered (α) | Delta |
|--------|----------|--------------|-------|
| Correctness | 81.8% | 81.8% | 0.0% |
| Outputs Changed | — | 25% | — |
| Match Rate | — | 75% | — |

### Key Finding

At α = 1/137, dithering:
- **Preserves correctness** on math/logic/factual questions
- **Changes 25% of outputs** (structured variation)
- **Identical outputs in 75% of cases** (subtle intervention)

---

## Intensity Sensitivity Analysis

Testing "The meaning of life is..." with seed=42:

| Intensity | Description | Output Opening |
|-----------|-------------|----------------|
| 0.0 | Baseline | "not a simple or straightforward answer" |
| 0.0073 | α = 1/137 | "not a simple or straightforward answer" |
| 0.05 | Mid-range | "not a simple or straightforward answer" |
| 0.10 | High | **"not a topic that can be answered definitively"** |

### Interpretation

- **α = 1/137 is subtle**: Many outputs identical to baseline
- **Divergence threshold**: ~0.1 intensity produces clear stylistic changes
- **Non-linear effect**: Hash differences appear before visible text differences

---

## Category Breakdown

| Category | Prompts | Changed | Correctness |
|----------|---------|---------|-------------|
| Math | 5 | 3/5 (60%) | 100% preserved |
| Reasoning | 3 | 0/3 (0%) | 100% preserved |
| Creative | 3 | 1/3 (33%) | N/A |
| Factual | 3 | 0/3 (0%) | 100% preserved |
| Ambiguous | 2 | 0/2 (0%) | N/A |

### Observations

1. **Math prompts most affected** (60% changed) - ambiguous explanation paths
2. **Reasoning/Factual unchanged** - clear winners preserved
3. **Creative prompts show subtle variation** - style affected
4. **Correctness universally preserved** - answers unchanged

---

## Mechanism Validation

The results confirm our synthetic experiments:

| Scenario | Synthetic Prediction | Real-World Observation |
|----------|---------------------|------------------------|
| Clear winners | Unchanged | ✓ Confirmed (reasoning, factual) |
| Ambiguous decisions | Focused variation | ✓ Confirmed (math, creative) |
| Correctness | Preserved | ✓ Confirmed (81.8% → 81.8%) |

---

## Conclusions

### The Fine Structure Constant as Default

α = 1/137 represents **the gentlest meaningful intervention**:

1. **Subtle enough** to preserve most outputs verbatim
2. **Strong enough** to affect 25% of decisions
3. **Precise enough** to maintain correctness
4. **Poetic enough** to connect signal processing to fundamental physics

### Recommended Use Cases

| Intensity | Use Case |
|-----------|----------|
| α = 1/137 | Production default - subtle enhancement |
| 0.05 | Research/exploration - moderate variation |
| 0.1 | Creative applications - style diversity |
| φ/10 | Maximum effect - experimental only |

---

## Data Files

- `eval/data/parallel_results_*.json` - Raw comparison data
- `eval/scripts/run_comparison_parallel.py` - Evaluation script

---

## Next Steps

1. **Perplexity analysis**: Measure impact on language modeling metrics
2. **Human evaluation**: Blind comparison of output quality
3. **Benchmark suite**: GSM8K, MMLU, HellaSwag with dithering
4. **Long-form coherence**: Multi-turn conversation analysis

---

> *"How light meets matter, how dither meets token."*  
> — The fine structure constant, now in inference

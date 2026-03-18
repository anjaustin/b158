# 3. Test Battery Design

## 3.1 Philosophy: Extreme Sampling

Rather than uniform random sampling, we employ **extreme sampling** from the difficulty distribution of each benchmark.

### Sampling Strategy

| Category | Percentile | Purpose |
|----------|------------|---------|
| **Easiest** | Bottom 10% | Detect degradation (breaking what works) |
| **Toughest** | Top 10% | Detect enhancement (unlocking new capability) |

### Sample Size Per Benchmark

| Category | Count |
|----------|-------|
| Easiest questions | 50 |
| Toughest questions | 50 |
| **Total per benchmark** | 100 |

## 3.2 Rationale

Optimization effects manifest differently across the difficulty spectrum:

```
                    EASY                              HARD
    ┌────────────────────────────────────────────────────────┐
    │  ████████                                    ████████  │
    │  Degradation                                Enhancement │
    │  Detection                                   Detection  │
    │  Zone                                        Zone       │
    └────────────────────────────────────────────────────────┘
         ▲                                              ▲
         │                                              │
    If optimization                               If optimization
    breaks baseline                               unlocks capability
    capability                                    at the edge
```

## 3.3 Difficulty Stratification Methods

### Explicit Labels (Preferred)

| Benchmark | Difficulty Field | Values |
|-----------|------------------|--------|
| MATH | `level` | 1-5 |
| ARC | Split name | Easy / Challenge |

### Proxy-Based

| Benchmark | Proxy Metric |
|-----------|--------------|
| GSM8K | Number of reasoning steps |
| MMLU | Historical accuracy rates |
| HellaSwag | Adversarial filtering score |

### Empirical (Fallback)

For benchmarks without labels:

1. Run baseline model on full dataset
2. Rank by model confidence (logit margin)
3. Bottom 10% confidence → Hardest
4. Top 10% confidence → Easiest

## 3.4 Cross-Benchmark Consistency

To ensure fair comparison:

- Same random seed for sampling across algorithms
- Identical prompt formatting per benchmark
- Matched evaluation conditions (hardware, load)

---

<!-- LINKED EXPERIMENTS -->
<!-- exp: -->

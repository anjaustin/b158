# 5. Signal Processing Metrics

These metrics are specifically designed to measure the effects of dithering and signal processing optimizations on 1-bit LLM inference.

---

## 5.1 Token Distribution Analysis

**Purpose**: Detect probability shaping effects from optimizations

### Measurements

| Metric | Description | Expected Effect |
|--------|-------------|-----------------|
| Token frequency distribution | Compare output token frequencies to baseline | Shifts indicate distribution shaping |
| Top-k probability mass | Sum of probabilities in top-k tokens | Changes reveal concentration effects |
| Tail distribution | Probability mass beyond top-k | Dithering may redistribute tail |
| Vocabulary utilization | Unique tokens / total tokens | Higher = more diverse output |

### Analysis Method

```
For each generation:
1. Record all token probabilities at each step
2. Compute cumulative distribution function (CDF)
3. Compare CDF between baseline and optimized
4. Measure Kullback-Leibler divergence
```

---

## 5.2 Entropy Measurements

**Purpose**: Quantify information density changes

### Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Per-token entropy** | H(t) = -Σ p(t) log p(t) | Uncertainty at each position |
| **Sequence entropy** | Mean H(t) across sequence | Overall generation uncertainty |
| **Entropy stability** | Std(H(t)) over positions | Consistency of uncertainty |
| **Cross-entropy** | H(P,Q) = -Σ P log Q | Divergence from reference |

### Entropy Over Generation Length

```
Entropy
   ▲
   │    ____
   │   /    \____
   │  /          \___
   │ /               \____  ← Baseline
   │/                     \
   ├──────────────────────────► Token Position
   
   Goal: Optimizations should not increase entropy decay
```

---

## 5.3 Response Stability

**Purpose**: Measure variance across identical conditions

### Protocol

1. Run identical prompts N times (N ≥ 10)
2. Use fixed random seed where possible
3. Measure variance metrics

### Metrics

| Metric | Description |
|--------|-------------|
| **Token agreement rate** | % of positions with same token across runs |
| **Semantic similarity** | Embedding cosine similarity between outputs |
| **Answer consistency** | For objective Q: same answer across runs? |
| **First-divergence position** | Token position where outputs first differ |

### Visualization

```
Run 1: The quick brown fox jumps over the lazy dog
Run 2: The quick brown fox leaps over the lazy dog
Run 3: The quick brown fox jumps over the sleepy dog
                          ▲                    ▲
                          │                    │
                   First divergence      Secondary divergence
                   (position 5)          (position 8)
```

---

## 5.4 Logit Distribution Analysis

**Purpose**: Direct measurement of optimization effects on model outputs

### Pre/Post Optimization Comparison

| Measurement | Baseline | Optimized | Delta |
|-------------|----------|-----------|-------|
| Mean logit | μ_b | μ_o | Δμ |
| Logit std | σ_b | σ_o | Δσ |
| Max logit | max_b | max_o | Δmax |
| Logit range | range_b | range_o | Δrange |

### Probability Calibration (ECE)

Expected Calibration Error measures if model confidence matches accuracy:

```
ECE = Σ (|accuracy(bin) - confidence(bin)|) × weight(bin)
```

| ECE Value | Interpretation |
|-----------|----------------|
| < 0.05 | Well calibrated |
| 0.05 - 0.15 | Moderately calibrated |
| > 0.15 | Poorly calibrated |

**Goal**: Optimizations should improve or maintain calibration

### Confidence Distribution

```
Count
  ▲
  │  ██
  │  ██ ██
  │  ██ ██    ██
  │  ██ ██ ██ ██ ██
  ├──────────────────► Confidence
     0.5  0.7  0.9  1.0
     
  Baseline: Should show peaks at decision boundaries
  Optimized: May show sharper or smoother distribution
```

---

## 5.5 Composite Signal Quality Score

A unified metric combining all signal processing measurements:

```
SQS = w1·(1 - entropy_decay) + 
      w2·token_stability + 
      w3·(1 - ECE) + 
      w4·vocab_utilization
      
where Σwi = 1
```

Default weights (adjustable):
- w1 = 0.25 (entropy)
- w2 = 0.30 (stability)
- w3 = 0.25 (calibration)
- w4 = 0.20 (diversity)

---

## 5.6 Data Collection Requirements

For each experiment run, capture:

```yaml
signal_metrics:
  token_distribution:
    top_k_mass: [float per position]
    vocab_utilization: float
    kl_divergence_from_baseline: float
  
  entropy:
    per_token: [float per position]
    mean: float
    std: float
    decay_slope: float
  
  stability:
    runs: int
    token_agreement: float
    semantic_similarity: float
    first_divergence_mean: float
  
  logits:
    mean: float
    std: float
    ece: float
    confidence_histogram: [int per bin]
```

---

<!-- LINKED EXPERIMENTS -->
<!-- exp: -->

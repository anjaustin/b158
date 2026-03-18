---
# ============================================================
# EXPERIMENT RECORD TEMPLATE
# ============================================================
# Copy this file to create a new experiment record.
# Filename convention: YYYY-MM-DD_<short_description>.md
# ============================================================

id: exp_XXX                    # Unique identifier (assigned from registry)
algorithm: <algorithm_name>    # Name of optimization being tested
version: "0.0.0"               # Algorithm version or git commit
date: YYYY-MM-DD
time_utc: "HH:MM:SS"
status: pending                # pending | running | complete | failed | rejected

# Linkage
parent_experiment: null        # Previous experiment this builds on
sections: []                   # Paper sections this informs [1,2,3...]
tags: []                       # Searchable tags [baseline, math, dithering, ...]

# Configuration
model:
  name: BitNet-b1.58-2B-4T
  path: models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
  checksum: null               # sha256 of model file

settings:
  top_k: 42
  top_p: 0.963
  temperature: 0.528
  ctx_size: 4096
  n_predict: 1536
  threads: null                # Set per-run
  
hardware:
  cpu: null
  cores: null
  threads_used: null
  ram_gb: null
  os: null
  kernel: null

environment:
  python_version: null
  bitnet_commit: null
  conda_env: null

# Results (populated after run)
results: {}
signal_metrics: {}
primary_criteria:
  no_regression: null
  perplexity_preserved: null
  math_accuracy_preserved: null
  stability_preserved: null
enhancement_score: null
---

# Experiment: <Title>

## 1. Objective

What are we testing? What hypothesis does this address?

```
<Clear statement of experimental goal>
```

## 2. Algorithm Description

Brief description of the optimization being tested.

```
<Algorithm overview>
<Key parameters>
<Expected effect>
```

## 3. Methodology

### 3.1 Modifications Made

```diff
# Code changes (if applicable)
- old code
+ new code
```

### 3.2 Benchmarks Run

- [ ] Perplexity (WikiText-2)
- [ ] MMLU
- [ ] HellaSwag
- [ ] ARC-Challenge
- [ ] WinoGrande
- [ ] GSM8K
- [ ] MATH
- [ ] GPQA
- [ ] SciQ

### 3.3 Signal Metrics Captured

- [ ] Token distribution
- [ ] Entropy
- [ ] Stability (N runs)
- [ ] Logit analysis

## 4. Results

### 4.1 Benchmark Performance

| Benchmark | Baseline | This Run | Δ | p-value | Cohen's d | Sig? |
|-----------|----------|----------|---|---------|-----------|------|
| MMLU | | | | | | |
| GSM8K | | | | | | |
| MATH | | | | | | |
| ... | | | | | | |

### 4.2 Easy vs Hard Stratification

| Benchmark | Easy (Δ) | Hard (Δ) | Notes |
|-----------|----------|----------|-------|
| | | | |

### 4.3 Signal Metrics

| Metric | Baseline | This Run | Δ | Interpretation |
|--------|----------|----------|---|----------------|
| Entropy (mean) | | | | |
| ECE | | | | |
| Stability | | | | |
| Vocab utilization | | | | |

## 5. Analysis

### 5.1 Primary Criteria Assessment

| Criterion | Pass/Fail | Evidence |
|-----------|-----------|----------|
| No regression | | |
| Perplexity preserved | | |
| Math accuracy preserved | | |
| Stability preserved | | |

**Primary Score**: X/4

### 5.2 Enhancement Assessment

| Indicator | Achieved? | Evidence |
|-----------|-----------|----------|
| Coherence improvement | | |
| Hard reasoning improvement | | |
| Variance reduction | | |
| Calibration improvement | | |
| Entropy stability | | |

**Enhancement Score**: X/5

### 5.3 Interpretation

```
<What do these results mean?>
<Why did we see these effects?>
<What does this tell us about the algorithm?>
```

## 6. Observations & Notes

### 6.1 Unexpected Findings

```
<Anything surprising or anomalous>
```

### 6.2 Failure Modes (if any)

```
<Where did the algorithm fail or underperform?>
```

### 6.3 Ideas for Improvement

```
<What could be tried next based on these results?>
```

## 7. Conclusion

```
<Summary statement>
<Recommendation: Accept/Reject/Modify>
<Next steps>
```

---

## Data Files

| File | Path | Description |
|------|------|-------------|
| Config | `data/<exp_id>/config.yaml` | |
| Results | `data/<exp_id>/metrics.json` | |
| Outputs | `data/<exp_id>/outputs/` | |

---

## Changelog

| Date | Change |
|------|--------|
| YYYY-MM-DD | Created |

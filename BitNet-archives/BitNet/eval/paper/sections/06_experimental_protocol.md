# 6. Experimental Protocol

## 6.1 Phase 1: Baseline Establishment

### Objectives
- Establish ground truth performance metrics
- Define performance envelope (min/max across runs)
- Create reference point for all comparisons

### Procedure

```
1. Environment Setup
   ├── Document hardware specifications
   ├── Record software versions
   └── Verify model integrity (checksum)

2. Benchmark Execution
   ├── Run complete benchmark suite (Sections 4.1-4.4)
   ├── Execute N=5 runs per benchmark
   └── Record all metrics with timestamps

3. Statistical Analysis
   ├── Compute mean ± std for each metric
   ├── Establish confidence intervals (95%)
   └── Identify any anomalous runs

4. Signal Metrics Baseline
   ├── Capture all Section 5 metrics
   └── Store raw logit data for comparison
```

### Deliverable
- `experiments/baseline/YYYY-MM-DD_baseline_complete.md`
- Associated data in `data/baseline/`

---

## 6.2 Phase 2: Individual Algorithm Evaluation

### Per-Algorithm Procedure

```
For each algorithm A in [1..N]:

1. Implementation
   ├── Document algorithm specification
   ├── Implement in bitnet.cpp fork
   └── Verify compilation and basic functionality

2. Smoke Test
   ├── Run 10 quick inference tests
   ├── Verify no crashes or obvious errors
   └── Check output sanity

3. Full Evaluation
   ├── Run complete benchmark suite
   ├── Execute N=5 runs per benchmark
   └── Capture all signal metrics

4. Comparison Analysis
   ├── Paired t-test against baseline (α=0.05)
   ├── Effect size (Cohen's d)
   ├── Regression analysis by difficulty
   └── Computational overhead measurement

5. Documentation
   └── Create experiment record with all findings
```

### Statistical Significance

| Test | Application |
|------|-------------|
| Paired t-test | Compare means (baseline vs optimized) |
| Wilcoxon signed-rank | Non-parametric alternative |
| Cohen's d | Effect size magnitude |
| Bonferroni correction | Multiple comparison adjustment |

### Effect Size Interpretation

| Cohen's d | Interpretation |
|-----------|----------------|
| < 0.2 | Negligible |
| 0.2 - 0.5 | Small |
| 0.5 - 0.8 | Medium |
| > 0.8 | Large |

---

## 6.3 Phase 3: Combination Studies

### Objectives
- Test promising algorithm combinations
- Identify synergies and interference
- Find optimal configuration

### Procedure

```
1. Selection
   ├── Rank algorithms by Phase 2 performance
   └── Select top K for combination testing

2. Pairwise Testing
   ├── Test all pairs from top K
   ├── Measure interaction effects
   └── Identify synergies (super-additive)

3. Multi-way Testing (if synergies found)
   ├── Test 3-way combinations
   └── Apply greedy selection if needed

4. Optimal Configuration
   ├── Identify best combination
   ├── Full benchmark validation
   └── Document final configuration
```

### Interaction Analysis

| Outcome | Interpretation | Action |
|---------|----------------|--------|
| Additive | Effects sum linearly | Safe to combine |
| Synergistic | Combined > sum of parts | Priority combination |
| Interference | Combined < individual | Avoid combination |

---

## 6.4 Phase 4: Deep Analysis

### Objectives
- Understand *why* optimizations work (or don't)
- Analyze signal processing effects in detail
- Generate insights for paper

### Analysis Dimensions

```
1. Difficulty-Stratified Analysis
   ├── Performance by difficulty percentile
   ├── Where do optimizations help most?
   └── Any difficulty-dependent failures?

2. Domain Analysis
   ├── Performance by subject (MMLU domains)
   ├── Math vs reasoning vs knowledge
   └── Domain-specific optimization effects?

3. Length Analysis
   ├── Performance vs output length
   ├── Coherence decay patterns
   └── Long-form specific effects?

4. Signal Deep Dive
   ├── Logit distribution evolution over generation
   ├── Entropy patterns in successful vs failed outputs
   └── Calibration by task type
```

---

## 6.5 Reproducibility Requirements

### Per-Experiment

| Requirement | How |
|-------------|-----|
| Random seed | Documented in experiment file |
| Environment | Conda/pip freeze stored |
| Code version | Git commit hash |
| Hardware | Exact specifications |
| Timestamps | UTC for all runs |

### Data Retention

```
data/
└── <experiment_id>/
    ├── config.yaml          # Full configuration
    ├── environment.txt      # pip freeze / conda list
    ├── raw_outputs/         # Model outputs
    ├── metrics.json         # Computed metrics
    └── logits/              # Raw logit data (optional, large)
```

---

## 6.6 Quality Control

### Pre-Run Checks

- [ ] Hardware idle (no background load)
- [ ] Model checksum verified
- [ ] Previous experiment completed and saved
- [ ] Disk space sufficient
- [ ] Environment activated

### Post-Run Checks

- [ ] All benchmarks completed
- [ ] No error logs
- [ ] Metrics within expected ranges
- [ ] Data files saved and verified
- [ ] Experiment record created

---

<!-- LINKED EXPERIMENTS -->
<!-- exp: -->

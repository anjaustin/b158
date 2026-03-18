# 8. Data Collection & Reporting

## 8.1 Experiment Record Schema

Every experiment generates a structured record:

```yaml
---
# HEADER
id: exp_XXX
algorithm: <name>
version: <semver or commit>
date: YYYY-MM-DD
time_utc: HH:MM:SS
status: [running|complete|failed|rejected]

# LINKAGE
parent_experiment: <id or null>
sections: [1,2,3...]  # Paper sections this informs
tags: [tag1, tag2, ...]

# CONFIGURATION
model:
  name: BitNet-b1.58-2B-4T
  path: models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
  checksum: <sha256>

settings:
  top_k: 42
  top_p: 0.963
  temperature: 0.528
  ctx_size: 4096
  n_predict: 1536

hardware:
  cpu: <model>
  cores: <n>
  threads_used: <n>
  ram_gb: <n>
  os: <string>
  kernel: <string>

environment:
  python: <version>
  commit: <hash>
  conda_env: <name or null>
---

# NARRATIVE CONTENT BELOW
```

---

## 8.2 Results Schema

### Benchmark Results

```yaml
results:
  <benchmark_name>:
    # Aggregate metrics
    accuracy: <float>
    accuracy_std: <float>
    n_runs: <int>
    
    # Stratified by difficulty
    easy:
      accuracy: <float>
      n_samples: <int>
      examples: [...]  # Notable cases
    hard:
      accuracy: <float>
      n_samples: <int>
      examples: [...]
    
    # Comparison to baseline
    vs_baseline:
      delta: <float>
      p_value: <float>
      cohens_d: <float>
      significant: <bool>
```

### Signal Metrics

```yaml
signal_metrics:
  token_distribution:
    top_k_mass_mean: <float>
    vocab_utilization: <float>
    kl_from_baseline: <float>
  
  entropy:
    mean: <float>
    std: <float>
    decay_slope: <float>
    vs_baseline_delta: <float>
  
  stability:
    n_runs: <int>
    token_agreement: <float>
    semantic_similarity: <float>
    first_divergence_position_mean: <float>
  
  calibration:
    ece: <float>
    ece_vs_baseline: <float>
    confidence_mean: <float>
```

---

## 8.3 Data Directory Structure

```
data/
└── <experiment_id>/
    ├── config.yaml              # Full configuration snapshot
    ├── environment.txt          # pip freeze output
    ├── git_info.txt             # Commit, branch, diff
    │
    ├── benchmarks/
    │   ├── mmlu/
    │   │   ├── results.json     # Per-question results
    │   │   ├── easy_samples.json
    │   │   └── hard_samples.json
    │   ├── gsm8k/
    │   │   └── ...
    │   └── .../
    │
    ├── signal/
    │   ├── entropy.csv          # Per-token entropy
    │   ├── logits/              # Raw logit dumps (optional)
    │   └── stability/
    │       ├── run_001.txt
    │       ├── run_002.txt
    │       └── ...
    │
    └── outputs/
        ├── raw/                 # Raw model outputs
        └── processed/           # Cleaned/parsed outputs
```

---

## 8.4 Reproducibility Artifacts

### Required Files

| File | Contents | Purpose |
|------|----------|---------|
| `config.yaml` | All parameters | Exact reproduction |
| `environment.txt` | pip/conda freeze | Dependency matching |
| `git_info.txt` | Commit + diff | Code state |
| `hardware.json` | Full specs | Hardware matching |
| `random_seeds.json` | All seeds used | Determinism |

### Git Info Template

```
commit: <full sha>
branch: <branch name>
dirty: <true/false>
diff: |
  <git diff output if dirty>
```

---

## 8.5 Reporting Formats

### Summary Table (Per Algorithm)

| Benchmark | Baseline | Optimized | Δ | p-value | d | Sig? |
|-----------|----------|-----------|---|---------|---|------|
| MMLU | 0.XX | 0.XX | +0.XX | 0.XXX | 0.XX | ✓/✗ |
| GSM8K | 0.XX | 0.XX | +0.XX | 0.XXX | 0.XX | ✓/✗ |
| ... | ... | ... | ... | ... | ... | ... |

### Signal Metrics Table

| Metric | Baseline | Optimized | Δ | Interpretation |
|--------|----------|-----------|---|----------------|
| Entropy (mean) | X.XX | X.XX | ±X.XX | Better/Worse/Same |
| ECE | X.XX | X.XX | ±X.XX | Better/Worse/Same |
| Stability | X.XX | X.XX | ±X.XX | Better/Worse/Same |

### Visualization Requirements

1. **Performance radar chart**: All benchmarks normalized
2. **Difficulty curve**: Performance vs difficulty percentile
3. **Entropy evolution**: Entropy over token position
4. **Calibration plot**: Confidence vs accuracy

---

## 8.6 Automation Scripts

### Data Collection

```bash
# Run experiment with full data capture
python scripts/run_experiment.py \
  --algorithm <name> \
  --config config.yaml \
  --output data/<exp_id>/
```

### Aggregation

```bash
# Generate summary across experiments
python scripts/aggregate_results.py \
  --experiments data/exp_* \
  --output reports/summary.md
```

### Comparison

```bash
# Compare two experiments
python scripts/compare_experiments.py \
  --baseline data/baseline/ \
  --experimental data/exp_001/ \
  --output reports/comparison_001.md
```

---

## 8.7 Quality Assurance

### Data Validation Checks

- [ ] All required fields present
- [ ] Numeric values in expected ranges
- [ ] No null/missing benchmark results
- [ ] File checksums match
- [ ] Timestamps consistent

### Integrity Verification

```bash
# Verify experiment data integrity
python scripts/verify_experiment.py data/<exp_id>/
```

---

<!-- LINKED EXPERIMENTS -->
<!-- exp: -->

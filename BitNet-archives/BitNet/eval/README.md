# BitNet Evaluation Framework

Rigorous evaluation infrastructure for testing dithering and signal processing optimizations on BitNet b1.58 inference.

## Quick Start

```bash
# Install dependencies
pip install -r scripts/requirements.txt

# Run quick baseline (10 samples per benchmark)
python scripts/run_baseline.py --quick

# Run full baseline (extreme sampling)
python scripts/run_baseline.py

# Run specific benchmark
python scripts/harness.py --benchmark gsm8k --difficulty extreme

# Create new experiment
python scripts/new_experiment.py my_algorithm --category dithering --tags math,signal

# Generate aggregation report
python scripts/aggregate_results.py
```

## Directory Structure

```
eval/
├── paper/                      # Paper sections (modular)
│   ├── PAPER.md               # Master assembly document
│   └── sections/              # Individual sections 1-9
│
├── experiments/               # Experiment records
│   ├── _template.md          # Template for new experiments
│   ├── baseline/             # Control experiments
│   └── algorithms/           # Optimization algorithm experiments
│
├── scripts/                   # Automation & harness
│   ├── config.py             # Centralized configuration
│   ├── harness.py            # Core evaluation harness
│   ├── signal_metrics.py     # Signal processing metrics
│   ├── run_baseline.py       # Baseline experiment runner
│   ├── new_experiment.py     # Create experiments from template
│   └── aggregate_results.py  # Generate summary reports
│
├── data/                      # Raw results, outputs, logs
├── reports/                   # Generated reports
└── registry.yaml             # Central experiment index
```

## Locked Inference Settings

All experiments use these fixed settings for consistency:

| Parameter | Value |
|-----------|-------|
| top_k | 42 |
| top_p | 0.963 |
| temperature | 0.528 |
| context_window | 4096 |
| tokens_to_predict | 1536 |

## Benchmark Suite

### Tier 1: Standard ML Evals
- MMLU (knowledge & reasoning)
- HellaSwag (commonsense)
- ARC-Challenge (science reasoning)
- WinoGrande (pronoun resolution)

### Tier 2: Mathematics
- GSM8K (grade school math)
- MATH (competition-level)

### Tier 3: Physics & Science
- GPQA (graduate-level)
- SciQ (high school science)

## Extreme Sampling

Instead of random sampling, we pull from difficulty extremes:
- **Easiest 10%**: Detect degradation
- **Toughest 10%**: Detect enhancement

## Signal Metrics

Specialized metrics for dithering/signal processing effects:
- **Entropy**: Per-token and decay analysis
- **Token Distribution**: Vocabulary utilization, top-k mass
- **Stability**: Response consistency across runs
- **Calibration**: ECE (Expected Calibration Error)

## Workflow

1. **Establish Baseline**: `python scripts/run_baseline.py`
2. **Create Experiment**: `python scripts/new_experiment.py algo_name`
3. **Implement & Run**: Modify code, run harness
4. **Compare**: Use signal metrics and benchmark results
5. **Document**: Fill in experiment record
6. **Aggregate**: Generate summary with `aggregate_results.py`

## Paper Sections

Each section is a separate markdown file that can reference experiments:

1. Introduction
2. Model Configuration  
3. Test Battery Design
4. Benchmark Suite
5. Signal Processing Metrics
6. Experimental Protocol
7. Success Criteria
8. Data Collection
9. References & Appendices

Build the full paper:
```bash
cat paper/sections/*.md > paper/full_paper.md
```

#!/usr/bin/env python3
"""
Run Baseline Experiment
Establishes control measurements for all benchmarks and signal metrics.

Usage:
    python run_baseline.py [--quick] [--output DIR]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    EVAL_DIR, BENCHMARKS, INFERENCE_SETTINGS, 
    DEFAULT_MODEL, SAMPLING_CONFIG
)
from harness import BitNetHarness, BenchmarkRunner
from new_experiment import create_experiment, load_registry, save_registry


def run_baseline(quick: bool = False, output_dir: Path = None):
    """
    Run complete baseline evaluation.
    
    Args:
        quick: If True, run with reduced samples for testing
        output_dir: Override output directory
    """
    print("=" * 60)
    print("BitNet Baseline Experiment")
    print("=" * 60)
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print(f"Model: {DEFAULT_MODEL}")
    print(f"Settings: top_k={INFERENCE_SETTINGS.top_k}, top_p={INFERENCE_SETTINGS.top_p}, temp={INFERENCE_SETTINGS.temperature}")
    print("=" * 60)
    
    # Create experiment record
    exp_path = create_experiment(
        algorithm="baseline",
        category="control",
        tags=["baseline", "control"]
    )
    print(f"\nExperiment record: {exp_path}")
    
    # Initialize harness
    try:
        harness = BitNetHarness(verbose=True)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure BitNet is built and model is downloaded:")
        print("  1. python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s")
        print("  2. huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T")
        sys.exit(1)
    
    runner = BenchmarkRunner(harness)
    
    # Determine sample count
    samples = 10 if quick else None  # None = full sampling config
    
    # Output directory
    if output_dir is None:
        output_dir = EVAL_DIR / "data" / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run benchmarks
    all_results = {}
    
    # Priority order: Math/Physics first (most sensitive to optimizations)
    priority_benchmarks = ["gsm8k", "math", "gpqa", "mmlu", "hellaswag", "arc_challenge", "winogrande", "sciq"]
    benchmarks_to_run = [b for b in priority_benchmarks if b in BENCHMARKS]
    
    print(f"\nRunning {len(benchmarks_to_run)} benchmarks...")
    print(f"Sample mode: {'quick (10 samples)' if quick else 'full (extreme sampling)'}")
    print()
    
    for benchmark in benchmarks_to_run:
        print(f"\n{'='*40}")
        print(f"BENCHMARK: {benchmark.upper()}")
        print(f"{'='*40}")
        
        try:
            result = runner.run_benchmark(
                benchmark,
                difficulty="extreme",
                max_samples=samples
            )
            
            all_results[benchmark] = {
                "accuracy_overall": result.accuracy_overall,
                "accuracy_easy": result.accuracy_easy,
                "accuracy_hard": result.accuracy_hard,
                "samples_total": result.samples_total,
                "samples_easy": result.samples_easy,
                "samples_hard": result.samples_hard,
                "mean_time": result.mean_time_per_sample,
                "timestamp": result.timestamp,
            }
            
            # Save per-benchmark results
            benchmark_file = output_dir / f"{benchmark}_results.json"
            with open(benchmark_file, "w") as f:
                json.dump({
                    "summary": all_results[benchmark],
                    "raw_results": result.raw_results,
                }, f, indent=2)
            
            print(f"\n{benchmark} Summary:")
            print(f"  Overall Accuracy: {result.accuracy_overall:.1%}")
            print(f"  Easy Accuracy:    {result.accuracy_easy:.1%} ({result.samples_easy} samples)")
            print(f"  Hard Accuracy:    {result.accuracy_hard:.1%} ({result.samples_hard} samples)")
            print(f"  Mean Time:        {result.mean_time_per_sample:.2f}s per sample")
            print(f"  Saved to: {benchmark_file}")
            
        except Exception as e:
            print(f"Error running {benchmark}: {e}")
            import traceback
            traceback.print_exc()
            all_results[benchmark] = {"error": str(e)}
    
    # Save aggregate results
    aggregate_file = output_dir / "baseline_aggregate.json"
    with open(aggregate_file, "w") as f:
        json.dump({
            "experiment_id": "baseline",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model": str(DEFAULT_MODEL),
            "settings": {
                "top_k": INFERENCE_SETTINGS.top_k,
                "top_p": INFERENCE_SETTINGS.top_p,
                "temperature": INFERENCE_SETTINGS.temperature,
                "ctx_size": INFERENCE_SETTINGS.ctx_size,
                "n_predict": INFERENCE_SETTINGS.n_predict,
            },
            "sampling": {
                "easiest_percentile": SAMPLING_CONFIG.easiest_percentile,
                "toughest_percentile": SAMPLING_CONFIG.toughest_percentile,
                "samples_per_tail": SAMPLING_CONFIG.samples_per_tail,
            },
            "results": all_results,
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("BASELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Aggregate results: {aggregate_file}")
    print(f"Experiment record: {exp_path}")
    
    # Print summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"{'Benchmark':<15} {'Overall':>10} {'Easy':>10} {'Hard':>10}")
    print("-"*60)
    
    for benchmark, result in all_results.items():
        if "error" in result:
            print(f"{benchmark:<15} {'ERROR':>10}")
        else:
            print(f"{benchmark:<15} {result['accuracy_overall']:>9.1%} {result['accuracy_easy']:>9.1%} {result['accuracy_hard']:>9.1%}")
    
    print("="*60)
    
    # Update registry
    registry = load_registry()
    for exp in registry.get("experiments", []):
        if exp.get("algorithm") == "baseline" and exp.get("status") == "pending":
            exp["status"] = "complete"
            break
    save_registry(registry)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run baseline experiment")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Quick mode with 10 samples per benchmark")
    parser.add_argument("--output", "-o", type=Path,
                        help="Output directory for results")
    args = parser.parse_args()
    
    run_baseline(quick=args.quick, output_dir=args.output)


if __name__ == "__main__":
    main()

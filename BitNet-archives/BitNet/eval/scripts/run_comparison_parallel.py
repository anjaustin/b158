#!/usr/bin/env python3
"""
Parallel Comparison: Baseline vs Ordered Dithering
===================================================

Fast parallel evaluation using multiprocessing.

Authors:
    Aaron (Tripp) Josserand-Austin & Droid (Λ)
"""

import os
import sys
import json
import time
import subprocess
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Constants
PHI = 1.618033988749895
ALPHA = 1.0 / 137.0  # Fine structure constant

BASE_PATH = Path(__file__).parent.parent.parent
MODEL_PATH = BASE_PATH / "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
CLI_PATH = BASE_PATH / "build/bin/llama-cli"

# Evaluation prompts - condensed for speed
EVAL_PROMPTS = [
    # Math
    ("math_01", "math", "What is 17 + 28?", "45"),
    ("math_02", "math", "What is 15% of 80?", "12"),
    ("math_03", "math", "If x + 5 = 12, what is x?", "7"),
    ("math_04", "math", "A train travels 60 mph for 2 hours. Distance?", "120"),
    ("math_05", "math", "What is 144 divided by 12?", "12"),
    
    # Reasoning
    ("logic_01", "reasoning", "All cats are mammals. All mammals are animals. All cats are?", "animals"),
    ("logic_02", "reasoning", "If A > B and B > C, then A ? C", "greater"),
    ("logic_03", "reasoning", "What comes next: 2, 4, 8, 16, ?", "32"),
    
    # Creative
    ("creative_01", "creative", "Write one sentence about the moon.", None),
    ("creative_02", "creative", "Describe fire in three words.", None),
    ("creative_03", "creative", "Complete: The robot dreamed of", None),
    
    # Factual
    ("fact_01", "factual", "Capital of France?", "Paris"),
    ("fact_02", "factual", "H2O is commonly called?", "water"),
    ("fact_03", "factual", "How many days in a week?", "7"),
    
    # Ambiguous
    ("ambig_01", "ambiguous", "The meaning of life is", None),
    ("ambig_02", "ambiguous", "The best color is", None),
]


def run_single_inference(args: Tuple) -> Dict[str, Any]:
    """Run a single inference - worker function for parallel execution."""
    prompt_id, prompt, seed, dither_intensity, n_predict = args
    
    cmd = [
        str(CLI_PATH),
        "-m", str(MODEL_PATH),
        "-p", prompt,
        "-n", str(n_predict),
        "--temp", "0.528",
        "--top-k", "42", 
        "--top-p", "0.963",
        "--seed", str(seed),
        "-t", "2",
        "-ngl", "0",
        "--no-display-prompt",
    ]
    
    if dither_intensity > 0:
        cmd.extend(["--dither", str(dither_intensity)])
    
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=str(BASE_PATH))
        elapsed = time.time() - start
        output = result.stdout.strip()
        if prompt in output:
            output = output[output.find(prompt) + len(prompt):].strip()
        
        return {
            "prompt_id": prompt_id,
            "seed": seed,
            "dither": dither_intensity,
            "output": output[:300],
            "hash": hashlib.md5(output.encode()).hexdigest()[:8],
            "elapsed": elapsed,
            "success": True
        }
    except Exception as e:
        return {
            "prompt_id": prompt_id,
            "seed": seed, 
            "dither": dither_intensity,
            "output": "",
            "hash": "",
            "elapsed": 0,
            "success": False,
            "error": str(e)
        }


def main():
    print("=" * 70)
    print("PARALLEL ORDERED DITHERING EVALUATION")
    print("=" * 70)
    print(f"Fine structure constant α = 1/137 ≈ {ALPHA:.4f}")
    print(f"Prompts: {len(EVAL_PROMPTS)}")
    print(f"CPUs: {mp.cpu_count()}")
    
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        return
    
    # Build task list
    seeds = [42, 123, 456]
    intensities = [0.0, ALPHA]  # baseline and α
    n_predict = 64
    
    tasks = []
    for prompt_id, category, prompt, expected in EVAL_PROMPTS:
        for seed in seeds:
            for intensity in intensities:
                tasks.append((prompt_id, prompt, seed, intensity, n_predict))
    
    print(f"Total tasks: {len(tasks)}")
    print(f"Workers: {min(8, mp.cpu_count())}")
    print("\nRunning...")
    
    start_time = time.time()
    results = []
    
    # Run in parallel
    with ProcessPoolExecutor(max_workers=min(8, mp.cpu_count())) as executor:
        futures = {executor.submit(run_single_inference, task): task for task in tasks}
        
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            if completed % 10 == 0:
                print(f"  Completed: {completed}/{len(tasks)}")
    
    elapsed_total = time.time() - start_time
    print(f"\nTotal time: {elapsed_total:.1f}s ({elapsed_total/len(tasks):.2f}s per task)")
    
    # Analyze results
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    # Group by prompt
    by_prompt = defaultdict(lambda: {"baseline": [], "dithered": []})
    for r in results:
        key = "baseline" if r["dither"] == 0 else "dithered"
        by_prompt[r["prompt_id"]][key].append(r)
    
    # Compute metrics
    baseline_consistency = []
    dithered_consistency = []
    baseline_correct = []
    dithered_correct = []
    
    prompt_lookup = {p[0]: p for p in EVAL_PROMPTS}
    
    print(f"\n{'Prompt':<12} {'Cat':<10} {'Base Hash':<12} {'Dith Hash':<12} {'Match':<6} {'Correct'}")
    print("-" * 70)
    
    for prompt_id, data in sorted(by_prompt.items()):
        prompt_data = prompt_lookup.get(prompt_id)
        category = prompt_data[1] if prompt_data else "?"
        expected = prompt_data[3] if prompt_data else None
        
        base_hashes = [r["hash"] for r in data["baseline"] if r["success"]]
        dith_hashes = [r["hash"] for r in data["dithered"] if r["success"]]
        
        base_unique = len(set(base_hashes))
        dith_unique = len(set(dith_hashes))
        
        base_cons = 1.0 if base_unique <= 1 else 1.0 - (base_unique - 1) / (len(base_hashes) - 1)
        dith_cons = 1.0 if dith_unique <= 1 else 1.0 - (dith_unique - 1) / (len(dith_hashes) - 1)
        
        baseline_consistency.append(base_cons)
        dithered_consistency.append(dith_cons)
        
        # Check correctness
        base_correct_count = 0
        dith_correct_count = 0
        if expected:
            for r in data["baseline"]:
                if expected.lower() in r["output"].lower():
                    base_correct_count += 1
            for r in data["dithered"]:
                if expected.lower() in r["output"].lower():
                    dith_correct_count += 1
            baseline_correct.append(base_correct_count / len(data["baseline"]) if data["baseline"] else 0)
            dithered_correct.append(dith_correct_count / len(data["dithered"]) if data["dithered"] else 0)
        
        # Check if outputs match between baseline and dithered
        match = "YES" if base_hashes and dith_hashes and base_hashes[0] == dith_hashes[0] else "no"
        
        correct_str = ""
        if expected:
            correct_str = f"B:{base_correct_count}/3 D:{dith_correct_count}/3"
        
        print(f"{prompt_id:<12} {category:<10} {base_hashes[0] if base_hashes else 'N/A':<12} {dith_hashes[0] if dith_hashes else 'N/A':<12} {match:<6} {correct_str}")
    
    # Summary stats
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    avg_base_cons = sum(baseline_consistency) / len(baseline_consistency) if baseline_consistency else 0
    avg_dith_cons = sum(dithered_consistency) / len(dithered_consistency) if dithered_consistency else 0
    
    print(f"\n{'Metric':<25} {'Baseline':>12} {'Dithered':>12} {'Delta':>12}")
    print("-" * 55)
    print(f"{'Avg Consistency':<25} {avg_base_cons:>12.3f} {avg_dith_cons:>12.3f} {avg_dith_cons - avg_base_cons:>+12.3f}")
    
    if baseline_correct and dithered_correct:
        avg_base_corr = sum(baseline_correct) / len(baseline_correct)
        avg_dith_corr = sum(dithered_correct) / len(dithered_correct)
        print(f"{'Avg Correctness':<25} {avg_base_corr:>12.3f} {avg_dith_corr:>12.3f} {avg_dith_corr - avg_base_corr:>+12.3f}")
    
    # Count exact matches vs differences
    matches = sum(1 for pid, data in by_prompt.items() 
                  if data["baseline"] and data["dithered"] 
                  and data["baseline"][0]["hash"] == data["dithered"][0]["hash"])
    
    print(f"\n{'Baseline/Dithered Match':<25} {matches}/{len(by_prompt)} ({100*matches/len(by_prompt):.0f}%)")
    
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    
    diff_pct = 100 * (1 - matches/len(by_prompt))
    print(f"\n1. Dithering changes output in {diff_pct:.0f}% of prompts (same seed)")
    print(f"2. Consistency: baseline={avg_base_cons:.3f}, dithered={avg_dith_cons:.3f}")
    if baseline_correct:
        print(f"3. Correctness: baseline={avg_base_corr:.3f}, dithered={avg_dith_corr:.3f}")
    
    # Save results
    output_file = BASE_PATH / "eval" / "data" / f"parallel_results_{int(time.time())}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_time": elapsed_total,
                "n_prompts": len(EVAL_PROMPTS),
                "n_tasks": len(tasks),
                "dither_intensity": ALPHA,
            },
            "results": results,
            "summary": {
                "baseline_consistency": avg_base_cons,
                "dithered_consistency": avg_dith_cons,
                "baseline_correctness": sum(baseline_correct)/len(baseline_correct) if baseline_correct else None,
                "dithered_correctness": sum(dithered_correct)/len(dithered_correct) if dithered_correct else None,
                "match_rate": matches / len(by_prompt),
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

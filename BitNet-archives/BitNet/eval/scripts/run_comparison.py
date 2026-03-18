#!/usr/bin/env python3
"""
Rigorous Comparison: Baseline vs Ordered Dithering
===================================================

Comprehensive evaluation for the research paper.

Authors:
    Aaron (Tripp) Josserand-Austin <tripp@anjaustin.com>
    Droid (Λ) <Factory AI>

"It's all about the Deltas!"
"""

import os
import sys
import json
import time
import subprocess
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import statistics

# Golden ratio
PHI = 1.618033988749895

@dataclass
class InferenceConfig:
    """Configuration for inference runs."""
    model_path: str = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
    cli_path: str = "build/bin/llama-cli"
    
    # Locked sampling parameters
    temperature: float = 0.528
    top_k: int = 42
    top_p: float = 0.963
    
    # Dithering
    dither_intensity: float = 0.0  # 0 = disabled
    
    # Generation
    n_predict: int = 128
    seed: int = 42
    threads: int = 4


@dataclass
class EvalPrompt:
    """A prompt for evaluation."""
    id: str
    category: str
    prompt: str
    expected: Optional[str] = None  # For math problems
    difficulty: str = "medium"


# Evaluation prompts across categories
EVAL_PROMPTS = [
    # Math / Reasoning (GSM8K-style)
    EvalPrompt("math_001", "math", 
               "What is 17 + 28? Think step by step, then give the final answer.",
               expected="45", difficulty="easy"),
    EvalPrompt("math_002", "math",
               "If a train travels 60 miles per hour for 2.5 hours, how far does it travel? Show your work.",
               expected="150", difficulty="easy"),
    EvalPrompt("math_003", "math",
               "A store sells apples for $0.50 each. If you buy 12 apples and pay with a $10 bill, how much change do you get?",
               expected="$4", difficulty="medium"),
    EvalPrompt("math_004", "math",
               "What is 15% of 80? Explain your reasoning.",
               expected="12", difficulty="medium"),
    EvalPrompt("math_005", "math",
               "If x + 5 = 12, what is x? Show the algebraic steps.",
               expected="7", difficulty="easy"),
    
    # Reasoning / Logic
    EvalPrompt("logic_001", "reasoning",
               "All cats are mammals. All mammals are animals. Therefore, all cats are ___?",
               expected="animals", difficulty="easy"),
    EvalPrompt("logic_002", "reasoning",
               "If it rains, the ground gets wet. The ground is wet. Can we conclude it rained? Explain why or why not.",
               difficulty="medium"),
    EvalPrompt("logic_003", "reasoning",
               "A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost?",
               expected="$0.05", difficulty="hard"),
    
    # Creative / Open-ended
    EvalPrompt("creative_001", "creative",
               "Write a haiku about artificial intelligence.",
               difficulty="medium"),
    EvalPrompt("creative_002", "creative",
               "Describe a sunset in three sentences.",
               difficulty="easy"),
    EvalPrompt("creative_003", "creative",
               "What would happen if humans could photosynthesize like plants?",
               difficulty="medium"),
    
    # Factual / Knowledge
    EvalPrompt("fact_001", "factual",
               "What is the capital of France?",
               expected="Paris", difficulty="easy"),
    EvalPrompt("fact_002", "factual",
               "Who wrote Romeo and Juliet?",
               expected="Shakespeare", difficulty="easy"),
    EvalPrompt("fact_003", "factual",
               "What is the chemical symbol for water?",
               expected="H2O", difficulty="easy"),
    
    # Ambiguous / Multiple valid answers
    EvalPrompt("ambig_001", "ambiguous",
               "Complete this sentence: The best programming language is",
               difficulty="medium"),
    EvalPrompt("ambig_002", "ambiguous",
               "What is the meaning of life?",
               difficulty="hard"),
    EvalPrompt("ambig_003", "ambiguous",
               "Should AI be regulated? Give a brief opinion.",
               difficulty="medium"),
]


def run_inference(config: InferenceConfig, prompt: str) -> Dict[str, Any]:
    """Run a single inference and return results."""
    
    cmd = [
        config.cli_path,
        "-m", config.model_path,
        "-p", prompt,
        "-n", str(config.n_predict),
        "--temp", str(config.temperature),
        "--top-k", str(config.top_k),
        "--top-p", str(config.top_p),
        "--seed", str(config.seed),
        "-t", str(config.threads),
        "-ngl", "0",
        "--no-display-prompt",
    ]
    
    if config.dither_intensity > 0:
        cmd.extend(["--dither", str(config.dither_intensity)])
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(Path(__file__).parent.parent.parent)
        )
        elapsed = time.time() - start_time
        
        # Extract output (remove prompt echo if present)
        output = result.stdout.strip()
        if prompt in output:
            output = output[output.find(prompt) + len(prompt):].strip()
        
        # Parse timing from stderr
        tokens_per_sec = 0.0
        for line in result.stderr.split('\n'):
            if 'eval time' in line and 'tokens per second' in line:
                try:
                    tokens_per_sec = float(line.split('(')[1].split('tokens per second')[0].strip().split()[-1])
                except:
                    pass
        
        return {
            "success": True,
            "output": output,
            "elapsed": elapsed,
            "tokens_per_second": tokens_per_sec,
            "return_code": result.returncode,
        }
        
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "timeout", "output": ""}
    except Exception as e:
        return {"success": False, "error": str(e), "output": ""}


def compute_text_hash(text: str) -> str:
    """Compute hash of text for comparison."""
    return hashlib.md5(text.encode()).hexdigest()[:8]


def compute_similarity(text1: str, text2: str) -> float:
    """Compute character-level similarity between two texts."""
    if not text1 or not text2:
        return 0.0
    
    # Simple character overlap
    shorter = min(len(text1), len(text2))
    matches = sum(1 for a, b in zip(text1, text2) if a == b)
    return matches / max(len(text1), len(text2))


def check_answer(output: str, expected: str) -> bool:
    """Check if expected answer appears in output."""
    if not expected:
        return True  # No expected answer
    return expected.lower() in output.lower()


def run_evaluation(
    prompts: List[EvalPrompt],
    n_runs: int = 3,
    seeds: List[int] = [42, 123, 456],
    dither_intensities: List[float] = [0.0, 0.0809],  # baseline and phi/20
    config: Optional[InferenceConfig] = None
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation.
    
    For each prompt:
    - Run baseline (no dither) with multiple seeds
    - Run dithered with multiple seeds
    - Compare outputs, consistency, correctness
    """
    
    if config is None:
        config = InferenceConfig()
    
    results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": config.model_path,
            "n_runs": n_runs,
            "seeds": seeds,
            "dither_intensities": dither_intensities,
            "sampling": {
                "temperature": config.temperature,
                "top_k": config.top_k,
                "top_p": config.top_p,
            }
        },
        "prompts": [],
        "summary": {}
    }
    
    total_prompts = len(prompts)
    
    for idx, prompt_data in enumerate(prompts):
        print(f"\n[{idx+1}/{total_prompts}] {prompt_data.category}: {prompt_data.id}")
        print(f"    Prompt: {prompt_data.prompt[:50]}...")
        
        prompt_result = {
            "id": prompt_data.id,
            "category": prompt_data.category,
            "prompt": prompt_data.prompt,
            "expected": prompt_data.expected,
            "difficulty": prompt_data.difficulty,
            "runs": {}
        }
        
        for intensity in dither_intensities:
            intensity_key = f"dither_{intensity:.4f}" if intensity > 0 else "baseline"
            prompt_result["runs"][intensity_key] = {
                "intensity": intensity,
                "outputs": [],
                "metrics": {}
            }
            
            config.dither_intensity = intensity
            
            for seed in seeds[:n_runs]:
                config.seed = seed
                
                result = run_inference(config, prompt_data.prompt)
                
                prompt_result["runs"][intensity_key]["outputs"].append({
                    "seed": seed,
                    "output": result.get("output", "")[:500],  # Truncate for storage
                    "hash": compute_text_hash(result.get("output", "")),
                    "elapsed": result.get("elapsed", 0),
                    "tokens_per_second": result.get("tokens_per_second", 0),
                    "correct": check_answer(result.get("output", ""), prompt_data.expected) if prompt_data.expected else None,
                })
                
                status = "✓" if result.get("success") else "✗"
                dither_str = f"d={intensity:.3f}" if intensity > 0 else "baseline"
                print(f"        {status} seed={seed} {dither_str}: {result.get('output', '')[:40]}...")
            
            # Compute metrics for this intensity
            outputs = prompt_result["runs"][intensity_key]["outputs"]
            output_texts = [o["output"] for o in outputs]
            output_hashes = [o["hash"] for o in outputs]
            
            unique_outputs = len(set(output_hashes))
            
            # Pairwise similarities
            similarities = []
            for i in range(len(output_texts)):
                for j in range(i+1, len(output_texts)):
                    similarities.append(compute_similarity(output_texts[i], output_texts[j]))
            
            avg_similarity = statistics.mean(similarities) if similarities else 1.0
            
            # Correctness rate
            correct_count = sum(1 for o in outputs if o.get("correct") is True)
            total_with_expected = sum(1 for o in outputs if o.get("correct") is not None)
            correctness_rate = correct_count / total_with_expected if total_with_expected > 0 else None
            
            prompt_result["runs"][intensity_key]["metrics"] = {
                "unique_outputs": unique_outputs,
                "total_runs": len(outputs),
                "consistency": 1.0 - (unique_outputs - 1) / max(len(outputs) - 1, 1),
                "avg_similarity": avg_similarity,
                "correctness_rate": correctness_rate,
                "avg_tokens_per_second": statistics.mean([o["tokens_per_second"] for o in outputs if o["tokens_per_second"] > 0]) if outputs else 0,
            }
        
        results["prompts"].append(prompt_result)
    
    # Compute overall summary
    summary = compute_summary(results)
    results["summary"] = summary
    
    return results


def compute_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute summary statistics across all prompts."""
    
    summary = {
        "by_category": defaultdict(lambda: {"baseline": [], "dithered": []}),
        "overall": {"baseline": {}, "dithered": {}},
        "comparison": {}
    }
    
    for prompt in results["prompts"]:
        category = prompt["category"]
        
        baseline_metrics = prompt["runs"].get("baseline", {}).get("metrics", {})
        dithered_key = [k for k in prompt["runs"].keys() if k.startswith("dither_") and k != "dither_0.0000"]
        dithered_metrics = prompt["runs"].get(dithered_key[0] if dithered_key else "baseline", {}).get("metrics", {})
        
        summary["by_category"][category]["baseline"].append(baseline_metrics)
        summary["by_category"][category]["dithered"].append(dithered_metrics)
    
    # Aggregate by category
    for category, data in summary["by_category"].items():
        for mode in ["baseline", "dithered"]:
            metrics_list = data[mode]
            if metrics_list:
                summary["by_category"][category][f"{mode}_avg"] = {
                    "consistency": statistics.mean([m.get("consistency", 0) for m in metrics_list]),
                    "similarity": statistics.mean([m.get("avg_similarity", 0) for m in metrics_list]),
                    "correctness": statistics.mean([m.get("correctness_rate", 0) for m in metrics_list if m.get("correctness_rate") is not None]) if any(m.get("correctness_rate") is not None for m in metrics_list) else None,
                }
    
    # Overall aggregation
    all_baseline = []
    all_dithered = []
    for prompt in results["prompts"]:
        baseline = prompt["runs"].get("baseline", {}).get("metrics", {})
        dithered_key = [k for k in prompt["runs"].keys() if k.startswith("dither_") and k != "dither_0.0000"]
        dithered = prompt["runs"].get(dithered_key[0] if dithered_key else "baseline", {}).get("metrics", {})
        all_baseline.append(baseline)
        all_dithered.append(dithered)
    
    for mode, metrics_list in [("baseline", all_baseline), ("dithered", all_dithered)]:
        if metrics_list:
            correctness_values = [m.get("correctness_rate") for m in metrics_list if m.get("correctness_rate") is not None]
            summary["overall"][mode] = {
                "avg_consistency": statistics.mean([m.get("consistency", 0) for m in metrics_list]),
                "avg_similarity": statistics.mean([m.get("avg_similarity", 0) for m in metrics_list]),
                "avg_correctness": statistics.mean(correctness_values) if correctness_values else None,
                "total_prompts": len(metrics_list),
            }
    
    # Comparison deltas
    if summary["overall"]["baseline"] and summary["overall"]["dithered"]:
        summary["comparison"] = {
            "consistency_delta": summary["overall"]["dithered"]["avg_consistency"] - summary["overall"]["baseline"]["avg_consistency"],
            "similarity_delta": summary["overall"]["dithered"]["avg_similarity"] - summary["overall"]["baseline"]["avg_similarity"],
        }
        
        if summary["overall"]["baseline"].get("avg_correctness") is not None and summary["overall"]["dithered"].get("avg_correctness") is not None:
            summary["comparison"]["correctness_delta"] = summary["overall"]["dithered"]["avg_correctness"] - summary["overall"]["baseline"]["avg_correctness"]
    
    return dict(summary)


def print_results(results: Dict[str, Any]):
    """Print formatted results."""
    
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\nModel: {results['metadata']['model']}")
    print(f"Prompts: {len(results['prompts'])}")
    print(f"Seeds: {results['metadata']['seeds']}")
    print(f"Dither intensities: {results['metadata']['dither_intensities']}")
    
    print("\n" + "-" * 70)
    print("OVERALL SUMMARY")
    print("-" * 70)
    
    summary = results["summary"]
    
    print(f"\n{'Metric':<25} {'Baseline':>15} {'Dithered':>15} {'Delta':>15}")
    print("-" * 70)
    
    baseline = summary["overall"]["baseline"]
    dithered = summary["overall"]["dithered"]
    comparison = summary["comparison"]
    
    print(f"{'Consistency':<25} {baseline['avg_consistency']:>15.3f} {dithered['avg_consistency']:>15.3f} {comparison.get('consistency_delta', 0):>+15.3f}")
    print(f"{'Similarity':<25} {baseline['avg_similarity']:>15.3f} {dithered['avg_similarity']:>15.3f} {comparison.get('similarity_delta', 0):>+15.3f}")
    
    if baseline.get('avg_correctness') is not None:
        print(f"{'Correctness':<25} {baseline['avg_correctness']:>15.3f} {dithered['avg_correctness']:>15.3f} {comparison.get('correctness_delta', 0):>+15.3f}")
    
    print("\n" + "-" * 70)
    print("BY CATEGORY")
    print("-" * 70)
    
    for category, data in summary["by_category"].items():
        print(f"\n{category.upper()}:")
        baseline_avg = data.get("baseline_avg", {})
        dithered_avg = data.get("dithered_avg", {})
        
        if baseline_avg and dithered_avg:
            print(f"  Consistency: baseline={baseline_avg.get('consistency', 0):.3f}, dithered={dithered_avg.get('consistency', 0):.3f}")
            print(f"  Similarity:  baseline={baseline_avg.get('similarity', 0):.3f}, dithered={dithered_avg.get('similarity', 0):.3f}")
            if baseline_avg.get('correctness') is not None:
                print(f"  Correctness: baseline={baseline_avg.get('correctness', 0):.3f}, dithered={dithered_avg.get('correctness', 0):.3f}")
    
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    
    # Interpretation
    cons_delta = comparison.get('consistency_delta', 0)
    sim_delta = comparison.get('similarity_delta', 0)
    
    print(f"\n1. Consistency: Dithering {'increases' if cons_delta > 0 else 'decreases'} output consistency by {abs(cons_delta):.1%}")
    print(f"2. Similarity: Dithering {'increases' if sim_delta > 0 else 'decreases'} pairwise similarity by {abs(sim_delta):.1%}")
    
    if comparison.get('correctness_delta') is not None:
        corr_delta = comparison['correctness_delta']
        print(f"3. Correctness: Dithering {'improves' if corr_delta > 0 else 'reduces'} answer accuracy by {abs(corr_delta):.1%}")


def main():
    """Main entry point."""
    
    print("=" * 70)
    print("ORDERED DITHERING EVALUATION")
    print("BitNet b1.58 - Baseline vs Dithered Inference")
    print("=" * 70)
    print(f"\nDither intensity: φ/20 = {PHI/20:.4f}")
    print(f"Prompts: {len(EVAL_PROMPTS)}")
    
    # Check model exists
    config = InferenceConfig()
    base_path = Path(__file__).parent.parent.parent
    model_path = base_path / config.model_path
    cli_path = base_path / config.cli_path
    
    if not model_path.exists():
        print(f"\nError: Model not found at {model_path}")
        return
    
    if not cli_path.exists():
        print(f"\nError: CLI not found at {cli_path}")
        return
    
    print(f"\nModel: {model_path}")
    print(f"CLI: {cli_path}")
    
    # Run evaluation
    results = run_evaluation(
        prompts=EVAL_PROMPTS,
        n_runs=3,
        seeds=[42, 123, 456],
        dither_intensities=[0.0, PHI/20],  # baseline and phi/20
        config=config
    )
    
    # Save results
    output_dir = base_path / "eval" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"comparison_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print_results(results)
    
    return results


if __name__ == "__main__":
    main()

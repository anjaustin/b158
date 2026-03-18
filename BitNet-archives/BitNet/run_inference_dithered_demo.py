#!/usr/bin/env python3
"""
BitNet Dithered Inference - Demonstration Engine
=================================================

This script demonstrates the ordered dithering effect on simulated
logit distributions, showing exactly how the algorithm would modify
token selection if we had direct logit access.

For full integration, this would be ported to C++ as a custom sampler
in the llama.cpp sampler chain.

Authors:
    Aaron (Tripp) Josserand-Austin <tripp@anjaustin.com>
    Droid (Λ) <Factory AI>

"It's all about the Deltas!"
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import platform

# Add eval/scripts to path
SCRIPT_DIR = Path(__file__).parent
EVAL_SCRIPTS = SCRIPT_DIR / "eval" / "scripts"
sys.path.insert(0, str(EVAL_SCRIPTS))

import numpy as np

# Import our dithering engine
from ordered_dither import (
    OrderedDither, DitherConfig, BayerMatrix,
    PHI, INTENSITY_MIN, INTENSITY_MAX,
    softmax, top_k_filter, top_p_filter,
    sample_with_dithering
)


# =============================================================================
# Dithering Demonstration
# =============================================================================

class DitherDemo:
    """
    Demonstrates exactly how ordered dithering affects token selection.
    
    Uses synthetic logits to show the mechanism, then applies analysis
    to actual model outputs.
    """
    
    def __init__(
        self,
        intensity: float = PHI / 20,
        matrix_size: int = 8,
        vocab_size: int = 32000
    ):
        self.config = DitherConfig(
            intensity=intensity,
            matrix_size=matrix_size
        )
        self.dither = OrderedDither(self.config)
        self.vocab_size = vocab_size
    
    def demonstrate_single_step(
        self,
        scenario: str = "close_race",
        position: int = 0
    ) -> Dict[str, Any]:
        """
        Demonstrate dithering effect on a single token selection.
        
        Shows before/after logits and probabilities.
        """
        # Generate scenario logits
        np.random.seed(42)
        logits = np.random.randn(self.vocab_size).astype(np.float32) * 2
        
        if scenario == "close_race":
            # Two tokens competing closely
            logits[100] = 5.00
            logits[200] = 4.95
            focus_tokens = [100, 200]
        elif scenario == "clear_winner":
            logits[100] = 10.0
            focus_tokens = [100]
        elif scenario == "multi_way_tie":
            for i in range(100, 110):
                logits[i] = 5.0 + np.random.randn() * 0.05
            focus_tokens = list(range(100, 110))
        else:
            focus_tokens = list(range(10))
        
        # Set position for dithering
        self.dither.state.sequence_position = position
        
        # Get thresholds for focus tokens
        thresholds = {}
        for tid in focus_tokens:
            thresholds[tid] = self.dither.get_threshold(tid)
        
        # Apply dithering
        dithered_logits = self.dither.apply(logits.copy())
        
        # Compute probabilities
        original_probs = softmax(logits, temperature=0.528)
        dithered_probs = softmax(dithered_logits, temperature=0.528)
        
        # Get rankings
        original_ranking = np.argsort(logits)[::-1]
        dithered_ranking = np.argsort(dithered_logits)[::-1]
        
        result = {
            'scenario': scenario,
            'position': position,
            'focus_tokens': {},
        }
        
        for tid in focus_tokens:
            orig_rank = np.where(original_ranking == tid)[0][0]
            dith_rank = np.where(dithered_ranking == tid)[0][0]
            
            result['focus_tokens'][tid] = {
                'original_logit': float(logits[tid]),
                'dithered_logit': float(dithered_logits[tid]),
                'delta': float(dithered_logits[tid] - logits[tid]),
                'threshold': float(thresholds[tid]),
                'original_prob': float(original_probs[tid]),
                'dithered_prob': float(dithered_probs[tid]),
                'original_rank': int(orig_rank),
                'dithered_rank': int(dith_rank),
                'rank_change': int(orig_rank - dith_rank),
            }
        
        result['top1_changed'] = original_ranking[0] != dithered_ranking[0]
        result['original_top1'] = int(original_ranking[0])
        result['dithered_top1'] = int(dithered_ranking[0])
        
        return result
    
    def demonstrate_sequence(
        self,
        n_positions: int = 16,
        scenario: str = "close_race"
    ) -> List[Dict[str, Any]]:
        """
        Demonstrate how dithering evolves over a sequence.
        
        Shows the Bayer pattern effect across positions.
        """
        self.dither.reset()
        results = []
        
        for pos in range(n_positions):
            result = self.demonstrate_single_step(scenario, pos)
            results.append(result)
            self.dither.accept(result['dithered_top1'])
        
        return results
    
    def visualize_bayer_effect(self) -> str:
        """Create ASCII visualization of Bayer pattern on token selection."""
        lines = []
        lines.append("=" * 70)
        lines.append("BAYER PATTERN EFFECT ON TOKEN SELECTION")
        lines.append("=" * 70)
        lines.append("")
        lines.append("Position (X) vs Token ID mod 8 (Y)")
        lines.append("Values show dither adjustment direction (+ boost, - suppress)")
        lines.append("")
        
        matrix = BayerMatrix.get(8)
        
        # Header
        lines.append("     " + " ".join(f"P{i}" for i in range(8)))
        lines.append("    +" + "-" * 31 + "+")
        
        for y in range(8):
            row = f"T{y} |"
            for x in range(8):
                val = matrix[y, x] - 0.5
                if val > 0.2:
                    char = "++"
                elif val > 0:
                    char = "+ "
                elif val < -0.2:
                    char = "--"
                elif val < 0:
                    char = "- "
                else:
                    char = "  "
                row += f" {char}"
            row += " |"
            lines.append(row)
        
        lines.append("    +" + "-" * 31 + "+")
        lines.append("")
        lines.append("Legend: ++ strong boost, + mild boost, - mild suppress, -- strong suppress")
        lines.append("")
        
        return "\n".join(lines)


# =============================================================================
# CLI Integration
# =============================================================================

def run_cli_inference(
    prompt: str,
    model_path: str,
    n_predict: int = 256,
    temperature: float = 0.528,
    top_k: int = 42,
    top_p: float = 0.963,
    seed: int = 42,
    threads: int = 4
) -> Dict[str, Any]:
    """Run inference using the CLI."""
    
    build_dir = "build"
    if platform.system() == "Windows":
        cli_path = os.path.join(build_dir, "bin", "Release", "llama-cli.exe")
        if not os.path.exists(cli_path):
            cli_path = os.path.join(build_dir, "bin", "llama-cli")
    else:
        cli_path = os.path.join(build_dir, "bin", "llama-cli")
    
    if not os.path.exists(cli_path):
        return {'error': f'CLI not found at {cli_path}'}
    
    command = [
        cli_path,
        '-m', model_path,
        '-n', str(n_predict),
        '-t', str(threads),
        '-p', prompt,
        '-ngl', '0',
        '-c', '4096',
        '--temp', str(temperature),
        '--top-k', str(top_k),
        '--top-p', str(top_p),
        '--seed', str(seed),
        '-b', '1',
    ]
    
    start = time.time()
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300
        )
        elapsed = time.time() - start
        
        output = result.stdout
        if prompt in output:
            output = output[output.find(prompt) + len(prompt):]
        
        return {
            'text': output.strip(),
            'elapsed': elapsed,
            'seed': seed,
        }
    except Exception as e:
        return {'error': str(e)}


def compare_outputs(
    prompt: str,
    model_path: str,
    n_runs: int = 5
) -> Dict[str, Any]:
    """
    Compare multiple inference runs to analyze output stability.
    
    This demonstrates what dithering would affect:
    runs with same seed should be identical,
    runs with different seeds show natural variation.
    """
    
    print("=" * 70)
    print("OUTPUT STABILITY ANALYSIS")
    print("=" * 70)
    print(f"Prompt: {prompt[:50]}...")
    print(f"Runs: {n_runs}")
    print()
    
    # Same seed runs (should be identical)
    print("Phase 1: Same seed (baseline stability)...")
    same_seed_outputs = []
    for i in range(n_runs):
        result = run_cli_inference(prompt, model_path, seed=42, n_predict=128)
        if 'error' not in result:
            same_seed_outputs.append(result['text'][:200])
            print(f"  Run {i+1}: {len(result['text'])} chars, {result['elapsed']:.1f}s")
    
    # Different seed runs (natural variation)
    print("\nPhase 2: Different seeds (natural variation)...")
    diff_seed_outputs = []
    for i in range(n_runs):
        result = run_cli_inference(prompt, model_path, seed=42 + i * 100, n_predict=128)
        if 'error' not in result:
            diff_seed_outputs.append(result['text'][:200])
            print(f"  Run {i+1}: {len(result['text'])} chars, {result['elapsed']:.1f}s")
    
    # Analysis
    same_seed_unique = len(set(same_seed_outputs))
    diff_seed_unique = len(set(diff_seed_outputs))
    
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print(f"Same seed unique outputs: {same_seed_unique}/{n_runs} (should be 1)")
    print(f"Diff seed unique outputs: {diff_seed_unique}/{n_runs}")
    print()
    print("INTERPRETATION:")
    print("  • Same-seed outputs being identical confirms deterministic baseline")
    print("  • Different-seed variation shows the 'decision space' dithering would affect")
    print("  • Ordered dithering would create STRUCTURED variation (not random)")
    print()
    
    return {
        'same_seed_unique': same_seed_unique,
        'diff_seed_unique': diff_seed_unique,
        'same_seed_outputs': same_seed_outputs,
        'diff_seed_outputs': diff_seed_outputs,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='BitNet Ordered Dithering Demonstration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Show Bayer pattern effect
    python run_inference_dithered_demo.py --demo pattern
    
    # Demonstrate single-step dithering
    python run_inference_dithered_demo.py --demo single --scenario close_race
    
    # Demonstrate sequence evolution
    python run_inference_dithered_demo.py --demo sequence
    
    # Compare actual model outputs
    python run_inference_dithered_demo.py --compare -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -p "What is 2+2?"
        """
    )
    
    parser.add_argument("--demo", choices=["pattern", "single", "sequence"],
                        help="Run demonstration mode")
    parser.add_argument("--scenario", choices=["close_race", "clear_winner", "multi_way_tie"],
                        default="close_race", help="Scenario for demo")
    parser.add_argument("--intensity", type=float, default=PHI/20,
                        help="Dither intensity")
    parser.add_argument("--compare", action="store_true",
                        help="Compare actual model outputs")
    parser.add_argument("-m", "--model", type=str,
                        default="models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf")
    parser.add_argument("-p", "--prompt", type=str,
                        default="The meaning of life is")
    parser.add_argument("-n", "--n-runs", type=int, default=5)
    
    args = parser.parse_args()
    
    # Header
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "ORDERED DITHERING DEMO" + " " * 26 + "║")
    print("║" + " " * 18 + "\"It's all about the Deltas!\"" + " " * 21 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    demo = DitherDemo(intensity=args.intensity)
    
    if args.demo == "pattern":
        print(demo.visualize_bayer_effect())
        
    elif args.demo == "single":
        result = demo.demonstrate_single_step(args.scenario, position=0)
        
        print(f"Scenario: {result['scenario']}")
        print(f"Position: {result['position']}")
        print(f"Top-1 changed: {result['top1_changed']}")
        print()
        print(f"{'Token':<8} {'Orig Logit':>12} {'Dith Logit':>12} {'Delta':>10} {'Threshold':>10}")
        print("-" * 60)
        
        for tid, data in result['focus_tokens'].items():
            print(f"{tid:<8} {data['original_logit']:>12.4f} {data['dithered_logit']:>12.4f} "
                  f"{data['delta']:>+10.4f} {data['threshold']:>10.4f}")
        
    elif args.demo == "sequence":
        results = demo.demonstrate_sequence(n_positions=16, scenario=args.scenario)
        
        print(f"Sequence demonstration: {args.scenario}")
        print()
        print(f"{'Pos':>4} {'Top-1 Changed':>14} {'Original':>10} {'Dithered':>10}")
        print("-" * 45)
        
        changes = 0
        for r in results:
            changed = "YES" if r['top1_changed'] else "no"
            if r['top1_changed']:
                changes += 1
            print(f"{r['position']:>4} {changed:>14} {r['original_top1']:>10} {r['dithered_top1']:>10}")
        
        print("-" * 45)
        print(f"Total changes: {changes}/{len(results)} ({100*changes/len(results):.0f}%)")
        
    elif args.compare:
        if not os.path.exists(args.model):
            print(f"Model not found: {args.model}")
            print("Please provide a valid model path with -m")
            return
        
        compare_outputs(args.prompt, args.model, n_runs=args.n_runs)
    
    else:
        # Default: show all demos
        print(demo.visualize_bayer_effect())
        print()
        
        print("=" * 70)
        print("SINGLE-STEP DEMONSTRATION: close_race scenario")
        print("=" * 70)
        result = demo.demonstrate_single_step("close_race", position=0)
        print(f"Top-1 changed: {result['top1_changed']}")
        for tid, data in result['focus_tokens'].items():
            print(f"  Token {tid}: logit {data['original_logit']:.2f} → {data['dithered_logit']:.2f} "
                  f"(Δ={data['delta']:+.4f})")


if __name__ == "__main__":
    main()

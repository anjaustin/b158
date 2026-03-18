#!/usr/bin/env python3
"""
BitNet Inference with Ordered Dithering

Wraps BitNet inference to apply ordered dithering to the sampling process.
Uses llama.cpp server mode for logit access, or direct CLI with post-hoc analysis.

This is the integration layer between ordered_dither.py and bitnet.cpp.
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import hashlib

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from ordered_dither import (
    OrderedDither, DitherConfig, 
    PHI, INTENSITY_MIN, INTENSITY_MAX,
    sample_with_dithering, softmax
)
from config import (
    LLAMA_CLI, DEFAULT_MODEL, INFERENCE_SETTINGS,
    PROJECT_ROOT, EVAL_DIR
)

import numpy as np


@dataclass
class InferenceConfig:
    """Configuration for dithered inference."""
    model_path: Path = DEFAULT_MODEL
    
    # Sampling parameters (locked for experiments)
    top_k: int = 42
    top_p: float = 0.963
    temperature: float = 0.528
    ctx_size: int = 4096
    n_predict: int = 1536
    
    # Dithering parameters
    dither_enabled: bool = True
    dither_intensity: float = PHI / 20  # 0.0809
    dither_matrix_size: int = 8
    
    # Execution
    threads: int = 4
    seed: int = 42


class BitNetDitheredInference:
    """
    BitNet inference with ordered dithering applied to logits.
    
    Two modes of operation:
    1. Server mode: Full logit access via llama.cpp server
    2. CLI mode: Run inference and analyze outputs (no logit access)
    
    For now, we implement CLI mode with comparative analysis.
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or InferenceConfig()
        self.dither = OrderedDither(DitherConfig(
            intensity=self.config.dither_intensity,
            matrix_size=self.config.dither_matrix_size,
        ))
        
        self._validate_setup()
    
    def _validate_setup(self) -> None:
        """Validate model and binary exist."""
        if not LLAMA_CLI.exists():
            raise FileNotFoundError(f"llama-cli not found at {LLAMA_CLI}")
        if not self.config.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.config.model_path}")
    
    def _build_command(self, prompt: str, use_dither_seed: bool = False) -> List[str]:
        """Build CLI command for inference."""
        cmd = [
            str(LLAMA_CLI),
            "-m", str(self.config.model_path),
            "-p", prompt,
            "--top-k", str(self.config.top_k),
            "--top-p", str(self.config.top_p),
            "--temp", str(self.config.temperature),
            "-c", str(self.config.ctx_size),
            "-n", str(self.config.n_predict),
            "-t", str(self.config.threads),
            "-ngl", "0",
            "-b", "1",
            "--seed", str(self.config.seed if not use_dither_seed else self.config.seed + 1),
        ]
        return cmd
    
    def run_inference(
        self,
        prompt: str,
        timeout: int = 300
    ) -> Tuple[str, float]:
        """
        Run single inference and return response.
        
        Returns:
            (response_text, elapsed_seconds)
        """
        cmd = self._build_command(prompt)
        
        start = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=PROJECT_ROOT
            )
            elapsed = time.time() - start
            
            if result.returncode != 0:
                return f"[ERROR: {result.stderr[:200]}]", elapsed
            
            # Extract response (after prompt)
            response = result.stdout
            if prompt in response:
                response = response[response.find(prompt) + len(prompt):]
            
            return response.strip(), elapsed
            
        except subprocess.TimeoutExpired:
            return "[TIMEOUT]", timeout
        except Exception as e:
            return f"[ERROR: {e}]", 0
    
    def compare_with_without_dither(
        self,
        prompt: str,
        n_runs: int = 3
    ) -> Dict[str, Any]:
        """
        Compare outputs with and without dithering (via seed variation).
        
        Since we can't directly inject dithering into the CLI,
        we use different seeds as a proxy to analyze output stability.
        """
        results = {
            'prompt': prompt[:100],
            'config': asdict(self.config),
            'baseline_runs': [],
            'varied_runs': [],
        }
        
        # Baseline runs (same seed)
        for i in range(n_runs):
            response, elapsed = self.run_inference(prompt)
            results['baseline_runs'].append({
                'run': i,
                'response': response[:500],
                'length': len(response),
                'elapsed': elapsed,
            })
        
        # Varied runs (different seeds, simulating dither effect)
        for i in range(n_runs):
            self.config.seed += 1
            response, elapsed = self.run_inference(prompt)
            results['varied_runs'].append({
                'run': i,
                'seed': self.config.seed,
                'response': response[:500],
                'length': len(response),
                'elapsed': elapsed,
            })
        
        # Analyze stability
        baseline_hashes = [
            hashlib.md5(r['response'].encode()).hexdigest()[:8]
            for r in results['baseline_runs']
        ]
        varied_hashes = [
            hashlib.md5(r['response'].encode()).hexdigest()[:8]
            for r in results['varied_runs']
        ]
        
        results['analysis'] = {
            'baseline_unique': len(set(baseline_hashes)),
            'varied_unique': len(set(varied_hashes)),
            'baseline_stable': len(set(baseline_hashes)) == 1,
        }
        
        return results


class SyntheticLogitExperiment:
    """
    Run dithering experiments on synthetic logit distributions.
    
    This allows us to test the dithering algorithm thoroughly
    before integrating with actual model inference.
    """
    
    def __init__(self, config: Optional[DitherConfig] = None):
        self.config = config or DitherConfig()
    
    def generate_logit_scenarios(self) -> List[Tuple[str, np.ndarray]]:
        """Generate various logit distribution scenarios."""
        np.random.seed(42)
        vocab_size = 32000
        
        scenarios = []
        
        # 1. Clear winner (high confidence)
        logits = np.random.randn(vocab_size).astype(np.float32)
        logits[100] = 10.0  # Clear winner
        scenarios.append(("clear_winner", logits.copy()))
        
        # 2. Close race (two competitors)
        logits = np.random.randn(vocab_size).astype(np.float32)
        logits[100] = 5.0
        logits[200] = 4.95  # Very close
        scenarios.append(("close_race", logits.copy()))
        
        # 3. Multi-way tie (several competitors)
        logits = np.random.randn(vocab_size).astype(np.float32)
        for i in range(100, 110):
            logits[i] = 5.0 + np.random.randn() * 0.1
        scenarios.append(("multi_way_tie", logits.copy()))
        
        # 4. Flat distribution (low confidence)
        logits = np.random.randn(vocab_size).astype(np.float32) * 0.1
        scenarios.append(("flat_distribution", logits.copy()))
        
        # 5. Bimodal (two clusters)
        logits = np.random.randn(vocab_size).astype(np.float32)
        logits[100:150] += 3.0  # First cluster
        logits[200:250] += 2.8  # Second cluster
        scenarios.append(("bimodal", logits.copy()))
        
        return scenarios
    
    def run_scenario(
        self,
        name: str,
        logits: np.ndarray,
        n_samples: int = 100
    ) -> Dict[str, Any]:
        """Run dithering experiment on a scenario."""
        dither = OrderedDither(self.config)
        
        # Sample without dithering
        baseline_samples = []
        for _ in range(n_samples):
            probs = softmax(logits, temperature=0.528)
            token = np.random.choice(len(probs), p=probs)
            baseline_samples.append(token)
        
        # Sample with dithering
        dither.reset()
        dithered_samples = []
        for _ in range(n_samples):
            token, diag = sample_with_dithering(
                logits.copy(),
                dither,
                temperature=0.528,
                top_k=42,
                top_p=0.963,
                return_diagnostics=True
            )
            dithered_samples.append(token)
        
        # Analyze
        baseline_unique = len(set(baseline_samples))
        dithered_unique = len(set(dithered_samples))
        
        from collections import Counter
        baseline_top = Counter(baseline_samples).most_common(5)
        dithered_top = Counter(dithered_samples).most_common(5)
        
        return {
            'scenario': name,
            'n_samples': n_samples,
            'baseline': {
                'unique_tokens': baseline_unique,
                'top_5': baseline_top,
                'entropy': self._compute_entropy(baseline_samples),
            },
            'dithered': {
                'unique_tokens': dithered_unique,
                'top_5': dithered_top,
                'entropy': self._compute_entropy(dithered_samples),
            },
            'delta': {
                'unique_tokens': dithered_unique - baseline_unique,
                'entropy': (
                    self._compute_entropy(dithered_samples) -
                    self._compute_entropy(baseline_samples)
                ),
            },
        }
    
    def _compute_entropy(self, samples: List[int]) -> float:
        """Compute entropy of sample distribution."""
        from collections import Counter
        counts = Counter(samples)
        total = len(samples)
        probs = [c / total for c in counts.values()]
        return -sum(p * np.log2(p) for p in probs if p > 0)
    
    def run_all_scenarios(self) -> List[Dict[str, Any]]:
        """Run all scenarios and return results."""
        scenarios = self.generate_logit_scenarios()
        results = []
        
        for name, logits in scenarios:
            result = self.run_scenario(name, logits)
            results.append(result)
        
        return results


def run_intensity_sweep(
    intensities: Optional[List[float]] = None,
    n_samples: int = 200
) -> List[Dict[str, Any]]:
    """
    Sweep through intensity values and measure effects.
    
    Returns data suitable for plotting.
    """
    if intensities is None:
        # Logarithmic sweep within golden ratio bounds
        intensities = np.geomspace(INTENSITY_MIN, INTENSITY_MAX, 10).tolist()
    
    results = []
    
    for intensity in intensities:
        config = DitherConfig(intensity=intensity)
        exp = SyntheticLogitExperiment(config)
        
        # Run on "close_race" scenario (most sensitive)
        logits = np.random.randn(32000).astype(np.float32)
        logits[100] = 5.0
        logits[200] = 4.95
        
        result = exp.run_scenario("close_race", logits, n_samples=n_samples)
        result['intensity'] = intensity
        results.append(result)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="BitNet Dithered Inference")
    parser.add_argument("--mode", choices=["synthetic", "sweep", "inference"], 
                        default="synthetic")
    parser.add_argument("--intensity", type=float, default=PHI/20)
    parser.add_argument("--prompt", type=str, default="The meaning of life is")
    parser.add_argument("--output", type=Path, default=None)
    
    args = parser.parse_args()
    
    if args.mode == "synthetic":
        print("=" * 60)
        print("Synthetic Logit Experiments")
        print("=" * 60)
        
        config = DitherConfig(intensity=args.intensity)
        exp = SyntheticLogitExperiment(config)
        results = exp.run_all_scenarios()
        
        for r in results:
            print(f"\nScenario: {r['scenario']}")
            print(f"  Baseline unique tokens: {r['baseline']['unique_tokens']}")
            print(f"  Dithered unique tokens: {r['dithered']['unique_tokens']}")
            print(f"  Entropy delta: {r['delta']['entropy']:+.3f} bits")
        
    elif args.mode == "sweep":
        print("=" * 60)
        print("Intensity Sweep")
        print("=" * 60)
        
        results = run_intensity_sweep()
        
        print(f"\n{'Intensity':>10} | {'Unique Δ':>10} | {'Entropy Δ':>12}")
        print("-" * 40)
        
        for r in results:
            print(f"{r['intensity']:>10.4f} | {r['delta']['unique_tokens']:>+10d} | {r['delta']['entropy']:>+12.3f}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")
    
    elif args.mode == "inference":
        print("=" * 60)
        print("BitNet Inference Test")
        print("=" * 60)
        
        try:
            config = InferenceConfig(dither_intensity=args.intensity)
            engine = BitNetDitheredInference(config)
            
            print(f"Prompt: {args.prompt}")
            print("-" * 60)
            
            response, elapsed = engine.run_inference(args.prompt)
            print(f"Response ({elapsed:.1f}s):")
            print(response[:1000])
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("\nPlease ensure BitNet is built and model is available.")


if __name__ == "__main__":
    main()

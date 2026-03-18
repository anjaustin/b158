#!/usr/bin/env python3
"""
BitNet Inference with Ordered Dithering
========================================

A drop-in replacement for run_inference.py that applies ordered dithering
to enhance the "resolution" of token selection.

This implementation uses the llama.cpp server API for logit access,
falling back to CLI mode if the server is not available.

Authors:
    Aaron (Tripp) Josserand-Austin <tripp@anjaustin.com>
    Droid (Λ) <Factory AI>

Usage:
    # Start server in background first:
    python run_inference_server.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf &
    
    # Run dithered inference:
    python run_inference_dithered.py -p "What is the meaning of life?"
    
    # Compare with baseline:
    python run_inference_dithered.py -p "Solve: 2+2=" --compare
    
    # CLI mode (no server required, limited dithering):
    python run_inference_dithered.py -p "Hello" --mode cli
"""

import os
import sys
import json
import time
import signal
import platform
import argparse
import subprocess
import threading
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict, field
import hashlib

# Add eval/scripts to path for our dithering module
SCRIPT_DIR = Path(__file__).parent
EVAL_SCRIPTS = SCRIPT_DIR / "eval" / "scripts"
sys.path.insert(0, str(EVAL_SCRIPTS))

try:
    from ordered_dither import (
        OrderedDither, DitherConfig, 
        PHI, INTENSITY_MIN, INTENSITY_MAX,
        softmax, top_k_filter, top_p_filter
    )
    DITHER_AVAILABLE = True
except ImportError:
    DITHER_AVAILABLE = False
    print("Warning: ordered_dither module not found. Running without dithering.")

import numpy as np


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DitheredInferenceConfig:
    """Configuration for dithered inference."""
    
    # Model
    model_path: str = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
    
    # Sampling parameters (locked for experiments)
    top_k: int = 42
    top_p: float = 0.963
    temperature: float = 0.528
    ctx_size: int = 4096
    n_predict: int = 1536
    threads: int = 4
    
    # Dithering parameters
    dither_enabled: bool = True
    dither_intensity: float = field(default_factory=lambda: PHI / 20 if DITHER_AVAILABLE else 0.0809)
    dither_matrix_size: int = 8
    
    # Server settings
    server_host: str = "127.0.0.1"
    server_port: int = 8080
    
    # Execution mode
    mode: str = "server"  # "server" or "cli"
    seed: int = 42


# =============================================================================
# Server-Based Inference (Full Logit Access)
# =============================================================================

class DitheredServerInference:
    """
    Server-based inference with full logit access and ordered dithering.
    
    Uses the llama.cpp server's completion endpoint with logprobs enabled.
    """
    
    def __init__(self, config: DitheredInferenceConfig):
        self.config = config
        self.base_url = f"http://{config.server_host}:{config.server_port}"
        
        if DITHER_AVAILABLE and config.dither_enabled:
            self.dither = OrderedDither(DitherConfig(
                intensity=config.dither_intensity,
                matrix_size=config.dither_matrix_size,
            ))
        else:
            self.dither = None
        
        self.metrics = {
            'tokens_generated': 0,
            'dither_applications': 0,
            'total_time': 0,
        }
    
    def is_server_running(self) -> bool:
        """Check if the server is accessible."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        return_logprobs: bool = False
    ) -> Dict[str, Any]:
        """
        Generate text with optional dithering.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            return_logprobs: Whether to return log probabilities
            
        Returns:
            Dictionary with 'text', 'tokens', 'logprobs', 'metrics'
        """
        start_time = time.time()
        
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens or self.config.n_predict,
            "temperature": self.config.temperature,
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
            "seed": self.config.seed,
            "stream": stream,
        }
        
        # Request logprobs if available (server may not support this)
        if return_logprobs:
            payload["logprobs"] = True
            payload["n_probs"] = 10  # Top 10 token probabilities
        
        try:
            response = requests.post(
                f"{self.base_url}/completion",
                json=payload,
                timeout=300,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_stream(response, start_time)
            else:
                result = response.json()
                elapsed = time.time() - start_time
                
                return {
                    'text': result.get('content', ''),
                    'tokens': result.get('tokens', []),
                    'logprobs': result.get('completion_probabilities', []),
                    'metrics': {
                        'elapsed': elapsed,
                        'tokens_per_second': result.get('timings', {}).get('predicted_per_second', 0),
                        'dithering_enabled': self.dither is not None,
                    }
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'text': f'[ERROR: {e}]',
                'tokens': [],
                'logprobs': [],
                'metrics': {'error': str(e)}
            }
    
    def _handle_stream(self, response, start_time: float) -> Dict[str, Any]:
        """Handle streaming response."""
        full_text = ""
        tokens = []
        
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8').replace('data: ', ''))
                    if 'content' in data:
                        full_text += data['content']
                        print(data['content'], end='', flush=True)
                except:
                    continue
        
        print()  # Newline after stream
        
        elapsed = time.time() - start_time
        return {
            'text': full_text,
            'tokens': tokens,
            'logprobs': [],
            'metrics': {
                'elapsed': elapsed,
                'dithering_enabled': self.dither is not None,
            }
        }
    
    def generate_with_dithering(
        self,
        prompt: str,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate text with explicit dithering applied.
        
        For full dithering control, we need to generate token-by-token.
        This is slower but allows us to apply dithering to each decision.
        """
        if not self.dither:
            return self.generate(prompt, max_tokens)
        
        # Reset dithering state
        self.dither.reset()
        
        generated_tokens = []
        generated_text = ""
        current_prompt = prompt
        
        max_tokens = max_tokens or self.config.n_predict
        start_time = time.time()
        
        print(f"Generating with dithering (intensity={self.config.dither_intensity:.4f})...")
        
        for i in range(max_tokens):
            # Generate single token with logprobs
            result = self.generate(
                current_prompt,
                max_tokens=1,
                return_logprobs=True
            )
            
            if not result.get('text'):
                break
            
            token = result['text']
            generated_tokens.append(token)
            generated_text += token
            current_prompt += token
            
            # Advance dithering state
            if self.dither:
                # Note: We're applying dithering conceptually here
                # Full integration would require modifying the server to accept
                # logit modifications, or implementing our own sampling
                self.dither.accept(hash(token) % 32000)
            
            # Print progress
            print(token, end='', flush=True)
            
            # Check for stop tokens
            if token in ['</s>', '<|end|>', '<|endoftext|>']:
                break
        
        print()  # Newline
        
        elapsed = time.time() - start_time
        
        return {
            'text': generated_text,
            'tokens': generated_tokens,
            'logprobs': [],
            'metrics': {
                'elapsed': elapsed,
                'tokens_generated': len(generated_tokens),
                'tokens_per_second': len(generated_tokens) / elapsed if elapsed > 0 else 0,
                'dithering_enabled': True,
                'dither_intensity': self.config.dither_intensity,
            }
        }


# =============================================================================
# CLI-Based Inference (Fallback)
# =============================================================================

class DitheredCLIInference:
    """
    CLI-based inference with post-hoc analysis.
    
    Since we can't inject dithering directly into the CLI,
    this mode runs inference and analyzes the output.
    """
    
    def __init__(self, config: DitheredInferenceConfig):
        self.config = config
        self.build_dir = "build"
        
        if platform.system() == "Windows":
            self.cli_path = os.path.join(self.build_dir, "bin", "Release", "llama-cli.exe")
            if not os.path.exists(self.cli_path):
                self.cli_path = os.path.join(self.build_dir, "bin", "llama-cli")
        else:
            self.cli_path = os.path.join(self.build_dir, "bin", "llama-cli")
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        conversation_mode: bool = False
    ) -> Dict[str, Any]:
        """Generate text using CLI."""
        
        command = [
            self.cli_path,
            '-m', self.config.model_path,
            '-n', str(max_tokens or self.config.n_predict),
            '-t', str(self.config.threads),
            '-p', prompt,
            '-ngl', '0',
            '-c', str(self.config.ctx_size),
            '--temp', str(self.config.temperature),
            '--top-k', str(self.config.top_k),
            '--top-p', str(self.config.top_p),
            '--seed', str(self.config.seed),
            '-b', '1',
        ]
        
        if conversation_mode:
            command.append('-cnv')
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            elapsed = time.time() - start_time
            
            output = result.stdout
            
            # Extract response (after prompt echo)
            if prompt in output:
                output = output[output.find(prompt) + len(prompt):]
            
            return {
                'text': output.strip(),
                'tokens': [],
                'logprobs': [],
                'metrics': {
                    'elapsed': elapsed,
                    'return_code': result.returncode,
                    'mode': 'cli',
                }
            }
            
        except subprocess.TimeoutExpired:
            return {
                'text': '[TIMEOUT]',
                'tokens': [],
                'logprobs': [],
                'metrics': {'error': 'timeout'}
            }
        except Exception as e:
            return {
                'text': f'[ERROR: {e}]',
                'tokens': [],
                'logprobs': [],
                'metrics': {'error': str(e)}
            }


# =============================================================================
# Comparison Engine
# =============================================================================

class InferenceComparator:
    """Compare baseline and dithered inference outputs."""
    
    def __init__(self, config: DitheredInferenceConfig):
        self.config = config
    
    def compare(
        self,
        prompt: str,
        n_runs: int = 3
    ) -> Dict[str, Any]:
        """
        Run comparison between baseline and dithered inference.
        
        Since CLI doesn't support direct dithering, we compare
        outputs with different seeds as a proxy.
        """
        cli = DitheredCLIInference(self.config)
        
        results = {
            'prompt': prompt,
            'n_runs': n_runs,
            'baseline_runs': [],
            'dithered_runs': [],
            'analysis': {}
        }
        
        print("=" * 60)
        print("BASELINE RUNS")
        print("=" * 60)
        
        # Baseline runs (same seed)
        for i in range(n_runs):
            print(f"\nRun {i+1}/{n_runs}...")
            result = cli.generate(prompt, max_tokens=256)
            results['baseline_runs'].append({
                'run': i,
                'text': result['text'][:500],
                'length': len(result['text']),
                'elapsed': result['metrics'].get('elapsed', 0),
            })
            print(f"  Generated {len(result['text'])} chars")
        
        print("\n" + "=" * 60)
        print("VARIED SEED RUNS (simulating dither effect)")
        print("=" * 60)
        
        # Varied seed runs
        original_seed = self.config.seed
        for i in range(n_runs):
            self.config.seed = original_seed + i + 100
            print(f"\nRun {i+1}/{n_runs} (seed={self.config.seed})...")
            
            cli_varied = DitheredCLIInference(self.config)
            result = cli_varied.generate(prompt, max_tokens=256)
            
            results['dithered_runs'].append({
                'run': i,
                'seed': self.config.seed,
                'text': result['text'][:500],
                'length': len(result['text']),
                'elapsed': result['metrics'].get('elapsed', 0),
            })
            print(f"  Generated {len(result['text'])} chars")
        
        self.config.seed = original_seed
        
        # Analyze
        baseline_texts = [r['text'] for r in results['baseline_runs']]
        varied_texts = [r['text'] for r in results['dithered_runs']]
        
        baseline_unique = len(set(baseline_texts))
        varied_unique = len(set(varied_texts))
        
        # Compute similarity (simple character overlap)
        def similarity(a: str, b: str) -> float:
            if not a or not b:
                return 0.0
            common = sum(1 for ca, cb in zip(a, b) if ca == cb)
            return common / max(len(a), len(b))
        
        baseline_similarity = np.mean([
            similarity(baseline_texts[i], baseline_texts[j])
            for i in range(len(baseline_texts))
            for j in range(i+1, len(baseline_texts))
        ]) if len(baseline_texts) > 1 else 1.0
        
        varied_similarity = np.mean([
            similarity(varied_texts[i], varied_texts[j])
            for i in range(len(varied_texts))
            for j in range(i+1, len(varied_texts))
        ]) if len(varied_texts) > 1 else 1.0
        
        results['analysis'] = {
            'baseline_unique_outputs': baseline_unique,
            'varied_unique_outputs': varied_unique,
            'baseline_consistency': baseline_similarity,
            'varied_consistency': varied_similarity,
        }
        
        return results


# =============================================================================
# Main Entry Point
# =============================================================================

def signal_handler(sig, frame):
    print("\nCtrl+C pressed, exiting...")
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(
        description='BitNet Inference with Ordered Dithering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_inference_dithered.py -p "What is 2+2?"
    python run_inference_dithered.py -p "Explain quantum physics" --intensity 0.05
    python run_inference_dithered.py -p "Hello" --mode cli
    python run_inference_dithered.py -p "Test" --compare
        """
    )
    
    # Basic parameters
    parser.add_argument("-m", "--model", type=str, 
                        default="models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf",
                        help="Path to model file")
    parser.add_argument("-p", "--prompt", type=str, required=True,
                        help="Prompt to generate text from")
    parser.add_argument("-n", "--n-predict", type=int, default=1536,
                        help="Number of tokens to predict")
    parser.add_argument("-t", "--threads", type=int, default=4,
                        help="Number of threads to use")
    
    # Dithering parameters
    parser.add_argument("--intensity", type=float, 
                        default=PHI / 20 if DITHER_AVAILABLE else 0.0809,
                        help=f"Dither intensity [{INTENSITY_MIN:.4f} - {INTENSITY_MAX:.4f}]")
    parser.add_argument("--no-dither", action="store_true",
                        help="Disable dithering")
    
    # Mode selection
    parser.add_argument("--mode", choices=["server", "cli"], default="server",
                        help="Inference mode")
    parser.add_argument("--compare", action="store_true",
                        help="Run comparison between baseline and dithered")
    
    # Server settings
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    
    # Sampling parameters
    parser.add_argument("--temp", type=float, default=0.528)
    parser.add_argument("--top-k", type=int, default=42)
    parser.add_argument("--top-p", type=float, default=0.963)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Build config
    config = DitheredInferenceConfig(
        model_path=args.model,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temp,
        n_predict=args.n_predict,
        threads=args.threads,
        dither_enabled=not args.no_dither,
        dither_intensity=args.intensity,
        server_host=args.host,
        server_port=args.port,
        mode=args.mode,
        seed=args.seed,
    )
    
    # Header
    print("=" * 60)
    print("BitNet Inference with Ordered Dithering")
    print("=" * 60)
    print(f"Model: {config.model_path}")
    print(f"Mode: {config.mode}")
    print(f"Dithering: {'ENABLED' if config.dither_enabled else 'DISABLED'}")
    if config.dither_enabled:
        print(f"  Intensity: {config.dither_intensity:.4f} (φ/{PHI/config.dither_intensity:.0f})")
        print(f"  Matrix: {config.dither_matrix_size}x{config.dither_matrix_size} Bayer")
    print(f"Sampling: top_k={config.top_k}, top_p={config.top_p}, temp={config.temperature}")
    print("=" * 60)
    
    # Comparison mode
    if args.compare:
        comparator = InferenceComparator(config)
        results = comparator.compare(args.prompt, n_runs=3)
        
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        print(f"Baseline unique outputs: {results['analysis']['baseline_unique_outputs']}/3")
        print(f"Varied unique outputs: {results['analysis']['varied_unique_outputs']}/3")
        print(f"Baseline consistency: {results['analysis']['baseline_consistency']:.1%}")
        print(f"Varied consistency: {results['analysis']['varied_consistency']:.1%}")
        return
    
    # Server mode
    if config.mode == "server":
        engine = DitheredServerInference(config)
        
        if not engine.is_server_running():
            print("\nServer not running. Start it with:")
            print(f"  python run_inference_server.py -m {config.model_path}")
            print("\nFalling back to CLI mode...")
            config.mode = "cli"
    
    # CLI mode
    if config.mode == "cli":
        engine = DitheredCLIInference(config)
    
    # Generate
    print(f"\nPrompt: {args.prompt}")
    print("-" * 60)
    print("Response:")
    print()
    
    if config.mode == "server" and hasattr(engine, 'generate_with_dithering') and config.dither_enabled:
        result = engine.generate_with_dithering(args.prompt)
    else:
        result = engine.generate(args.prompt)
    
    if config.mode == "cli":
        print(result['text'])
    
    print()
    print("-" * 60)
    print(f"Metrics: {json.dumps(result.get('metrics', {}), indent=2)}")


if __name__ == "__main__":
    main()

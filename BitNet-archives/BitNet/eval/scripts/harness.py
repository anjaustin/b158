#!/usr/bin/env python3
"""
BitNet Evaluation Harness
Core infrastructure for running benchmarks against BitNet models.

Usage:
    python harness.py --benchmark mmlu --samples 100
    python harness.py --benchmark gsm8k --difficulty extreme
    python harness.py --all --output results/
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import hashlib

from config import (
    LLAMA_CLI, DEFAULT_MODEL, INFERENCE_SETTINGS,
    BENCHMARKS, SAMPLING_CONFIG, EVAL_DIR, PROJECT_ROOT
)


@dataclass
class InferenceResult:
    """Result from a single inference call."""
    prompt: str
    response: str
    tokens_generated: int
    time_seconds: float
    tokens_per_second: float
    
    
@dataclass
class BenchmarkResult:
    """Result from a benchmark evaluation."""
    benchmark: str
    timestamp: str
    samples_total: int
    samples_easy: int
    samples_hard: int
    accuracy_overall: float
    accuracy_easy: float
    accuracy_hard: float
    mean_time_per_sample: float
    raw_results: List[Dict]


class BitNetHarness:
    """Core harness for running BitNet inference and benchmarks."""
    
    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL,
        settings: Any = INFERENCE_SETTINGS,
        verbose: bool = True
    ):
        self.model_path = Path(model_path)
        self.settings = settings
        self.verbose = verbose
        
        self._validate_setup()
    
    def _validate_setup(self) -> None:
        """Validate that all required components are available."""
        if not LLAMA_CLI.exists():
            raise FileNotFoundError(
                f"llama-cli not found at {LLAMA_CLI}. "
                f"Please build BitNet first: python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s"
            )
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                f"Please download: huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T"
            )
    
    def _log(self, msg: str) -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    
    def get_model_checksum(self) -> str:
        """Compute SHA256 checksum of model file (first 1MB for speed)."""
        sha256 = hashlib.sha256()
        with open(self.model_path, "rb") as f:
            sha256.update(f.read(1024 * 1024))  # First 1MB
        return sha256.hexdigest()[:16]
    
    def run_inference(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        timeout: int = 300
    ) -> InferenceResult:
        """
        Run single inference with the BitNet model.
        
        Args:
            prompt: The input prompt
            max_tokens: Override n_predict (optional)
            timeout: Maximum seconds to wait
            
        Returns:
            InferenceResult with response and timing
        """
        cmd = [
            str(LLAMA_CLI),
            "-m", str(self.model_path),
            "-p", prompt,
        ] + self.settings.to_cli_args()
        
        if max_tokens:
            # Override n_predict
            cmd = [c for i, c in enumerate(cmd) if cmd[i-1] != "-n"]
            cmd.extend(["-n", str(max_tokens)])
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=PROJECT_ROOT
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode != 0:
                self._log(f"Warning: Non-zero return code: {result.returncode}")
                self._log(f"Stderr: {result.stderr[:500]}")
            
            response = result.stdout.strip()
            
            # Extract just the generated portion (after prompt)
            if prompt in response:
                response = response[response.find(prompt) + len(prompt):].strip()
            
            # Estimate tokens (rough: ~4 chars per token)
            tokens_generated = max(1, len(response) // 4)
            tps = tokens_generated / elapsed if elapsed > 0 else 0
            
            return InferenceResult(
                prompt=prompt,
                response=response,
                tokens_generated=tokens_generated,
                time_seconds=elapsed,
                tokens_per_second=tps
            )
            
        except subprocess.TimeoutExpired:
            self._log(f"Timeout after {timeout}s")
            return InferenceResult(
                prompt=prompt,
                response="[TIMEOUT]",
                tokens_generated=0,
                time_seconds=timeout,
                tokens_per_second=0
            )
        except Exception as e:
            self._log(f"Error: {e}")
            return InferenceResult(
                prompt=prompt,
                response=f"[ERROR: {e}]",
                tokens_generated=0,
                time_seconds=0,
                tokens_per_second=0
            )
    
    def run_batch(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        progress: bool = True
    ) -> List[InferenceResult]:
        """Run inference on a batch of prompts."""
        results = []
        total = len(prompts)
        
        for i, prompt in enumerate(prompts):
            if progress:
                self._log(f"Processing {i+1}/{total}")
            
            result = self.run_inference(prompt, max_tokens)
            results.append(result)
        
        return results


class BenchmarkRunner:
    """Run standardized benchmarks with extreme sampling."""
    
    def __init__(self, harness: BitNetHarness):
        self.harness = harness
        self.datasets: Dict[str, Any] = {}
    
    def load_dataset(self, benchmark_name: str) -> List[Dict]:
        """
        Load and cache a benchmark dataset.
        
        Requires: pip install datasets
        """
        if benchmark_name in self.datasets:
            return self.datasets[benchmark_name]
        
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Please install datasets: pip install datasets"
            )
        
        config = BENCHMARKS.get(benchmark_name)
        if not config:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        self.harness._log(f"Loading {config.name} from {config.dataset_id}...")
        
        if config.subset:
            ds = load_dataset(config.dataset_id, config.subset, split=config.split)
        else:
            ds = load_dataset(config.dataset_id, split=config.split)
        
        self.datasets[benchmark_name] = list(ds)
        self.harness._log(f"Loaded {len(self.datasets[benchmark_name])} samples")
        
        return self.datasets[benchmark_name]
    
    def stratify_by_difficulty(
        self,
        samples: List[Dict],
        benchmark_name: str
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Split samples into easiest and hardest subsets.
        
        Returns:
            (easy_samples, hard_samples)
        """
        config = BENCHMARKS[benchmark_name]
        n_per_tail = SAMPLING_CONFIG.samples_per_tail
        
        if config.difficulty_field and config.difficulty_field in samples[0]:
            # Use explicit difficulty field
            if benchmark_name == "math":
                # MATH has levels 1-5
                easy = [s for s in samples if s.get("level", 3) == 1]
                hard = [s for s in samples if s.get("level", 3) == 5]
            elif benchmark_name == "gsm8k":
                # GSM8K: count steps in solution
                def count_steps(s):
                    return s.get("answer", "").count("\n")
                
                sorted_samples = sorted(samples, key=count_steps)
                easy = sorted_samples[:n_per_tail]
                hard = sorted_samples[-n_per_tail:]
            else:
                # Generic: sort by difficulty field
                sorted_samples = sorted(
                    samples,
                    key=lambda x: x.get(config.difficulty_field, 0)
                )
                easy = sorted_samples[:n_per_tail]
                hard = sorted_samples[-n_per_tail:]
        else:
            # Empirical difficulty: will be computed after baseline run
            # For now, random sample
            import random
            random.seed(SAMPLING_CONFIG.random_seed)
            shuffled = random.sample(samples, min(len(samples), n_per_tail * 2))
            easy = shuffled[:n_per_tail]
            hard = shuffled[n_per_tail:n_per_tail * 2]
        
        # Ensure we have enough samples
        easy = easy[:n_per_tail]
        hard = hard[:n_per_tail]
        
        self.harness._log(f"Stratified: {len(easy)} easy, {len(hard)} hard")
        
        return easy, hard
    
    def format_prompt(self, sample: Dict, benchmark_name: str) -> str:
        """Format a sample into a prompt using the benchmark template."""
        config = BENCHMARKS[benchmark_name]
        template = config.prompt_template
        
        # Handle different benchmark formats
        if benchmark_name == "mmlu":
            choices = sample.get("choices", ["", "", "", ""])
            return template.format(
                question=sample.get("question", ""),
                choice_a=choices[0] if len(choices) > 0 else "",
                choice_b=choices[1] if len(choices) > 1 else "",
                choice_c=choices[2] if len(choices) > 2 else "",
                choice_d=choices[3] if len(choices) > 3 else "",
            )
        
        elif benchmark_name == "hellaswag":
            endings = sample.get("endings", ["", "", "", ""])
            return template.format(
                context=sample.get("ctx", ""),
                choice_a=endings[0] if len(endings) > 0 else "",
                choice_b=endings[1] if len(endings) > 1 else "",
                choice_c=endings[2] if len(endings) > 2 else "",
                choice_d=endings[3] if len(endings) > 3 else "",
            )
        
        elif benchmark_name == "arc_challenge":
            choices = sample.get("choices", {})
            choice_texts = choices.get("text", [])
            choice_labels = choices.get("label", [])
            formatted_choices = "\n".join(
                f"{label}) {text}" 
                for label, text in zip(choice_labels, choice_texts)
            )
            return template.format(
                question=sample.get("question", ""),
                choices=formatted_choices
            )
        
        elif benchmark_name == "winogrande":
            return template.format(
                sentence=sample.get("sentence", ""),
                option1=sample.get("option1", ""),
                option2=sample.get("option2", ""),
            )
        
        elif benchmark_name in ["gsm8k", "math"]:
            field = "problem" if benchmark_name == "math" else "question"
            return template.format(
                question=sample.get(field, ""),
                problem=sample.get(field, ""),
            )
        
        elif benchmark_name == "gpqa":
            # GPQA has specific column names
            return template.format(
                question=sample.get("Question", ""),
                choice_a=sample.get("Incorrect Answer 1", ""),
                choice_b=sample.get("Incorrect Answer 2", ""),
                choice_c=sample.get("Incorrect Answer 3", ""),
                choice_d=sample.get("Correct Answer", ""),
            )
        
        elif benchmark_name == "sciq":
            return template.format(
                question=sample.get("question", ""),
                support=sample.get("support", ""),
            )
        
        # Fallback: simple question format
        return sample.get(config.question_field, str(sample))
    
    def evaluate_response(
        self,
        response: str,
        sample: Dict,
        benchmark_name: str
    ) -> bool:
        """Check if response is correct."""
        config = BENCHMARKS[benchmark_name]
        
        # Normalize response
        response = response.strip().upper()
        
        # Get expected answer
        expected = str(sample.get(config.answer_field, "")).strip().upper()
        
        if benchmark_name in ["mmlu", "hellaswag", "arc_challenge", "gpqa"]:
            # Multiple choice: check if letter matches
            response_letter = ""
            for char in response:
                if char in "ABCD":
                    response_letter = char
                    break
            
            # Map expected to letter if needed
            if expected.isdigit():
                expected = chr(ord('A') + int(expected))
            
            return response_letter == expected
        
        elif benchmark_name == "winogrande":
            # Binary choice: 1 or 2
            response_num = ""
            for char in response:
                if char in "12":
                    response_num = char
                    break
            return response_num == expected
        
        elif benchmark_name in ["gsm8k", "math"]:
            # Math: extract final numerical answer
            import re
            
            # Try to find "The answer is X" pattern
            answer_match = re.search(r"answer\s+is\s*:?\s*\$?([0-9,.\-]+)", response.lower())
            if answer_match:
                response_num = answer_match.group(1).replace(",", "")
            else:
                # Find last number in response
                numbers = re.findall(r"[\-]?[0-9,]+\.?[0-9]*", response)
                response_num = numbers[-1].replace(",", "") if numbers else ""
            
            # Extract expected number
            expected_match = re.search(r"####\s*([0-9,.\-]+)", expected)
            if expected_match:
                expected_num = expected_match.group(1).replace(",", "")
            else:
                expected_num = expected.replace(",", "")
            
            try:
                return abs(float(response_num) - float(expected_num)) < 0.01
            except:
                return response_num == expected_num
        
        elif benchmark_name == "sciq":
            # Check if correct answer is in response
            correct = sample.get("correct_answer", "").lower()
            return correct in response.lower()
        
        # Fallback: exact match
        return response == expected
    
    def run_benchmark(
        self,
        benchmark_name: str,
        difficulty: str = "extreme",  # "extreme", "all", "easy", "hard"
        max_samples: Optional[int] = None
    ) -> BenchmarkResult:
        """
        Run a complete benchmark evaluation.
        
        Args:
            benchmark_name: Name of benchmark to run
            difficulty: Sampling strategy
            max_samples: Override sample count
        """
        self.harness._log(f"=== Running {benchmark_name} ({difficulty}) ===")
        
        # Load data
        samples = self.load_dataset(benchmark_name)
        
        # Stratify
        easy_samples, hard_samples = self.stratify_by_difficulty(samples, benchmark_name)
        
        if difficulty == "extreme":
            test_samples = [("easy", s) for s in easy_samples] + [("hard", s) for s in hard_samples]
        elif difficulty == "easy":
            test_samples = [("easy", s) for s in easy_samples]
        elif difficulty == "hard":
            test_samples = [("hard", s) for s in hard_samples]
        else:  # all
            import random
            random.seed(SAMPLING_CONFIG.random_seed)
            n = max_samples or 100
            test_samples = [("mixed", s) for s in random.sample(samples, min(n, len(samples)))]
        
        if max_samples:
            test_samples = test_samples[:max_samples]
        
        # Run evaluation
        results = []
        correct_easy = 0
        correct_hard = 0
        total_easy = 0
        total_hard = 0
        total_time = 0
        
        for i, (diff_label, sample) in enumerate(test_samples):
            self.harness._log(f"[{i+1}/{len(test_samples)}] {diff_label}")
            
            prompt = self.format_prompt(sample, benchmark_name)
            inference = self.harness.run_inference(prompt, max_tokens=512)
            is_correct = self.evaluate_response(inference.response, sample, benchmark_name)
            
            results.append({
                "difficulty": diff_label,
                "correct": is_correct,
                "prompt": prompt[:200],
                "response": inference.response[:500],
                "time": inference.time_seconds,
            })
            
            total_time += inference.time_seconds
            
            if diff_label == "easy":
                total_easy += 1
                if is_correct:
                    correct_easy += 1
            elif diff_label == "hard":
                total_hard += 1
                if is_correct:
                    correct_hard += 1
        
        # Compute metrics
        total = len(test_samples)
        correct_total = correct_easy + correct_hard
        
        return BenchmarkResult(
            benchmark=benchmark_name,
            timestamp=datetime.utcnow().isoformat() + "Z",
            samples_total=total,
            samples_easy=total_easy,
            samples_hard=total_hard,
            accuracy_overall=correct_total / total if total > 0 else 0,
            accuracy_easy=correct_easy / total_easy if total_easy > 0 else 0,
            accuracy_hard=correct_hard / total_hard if total_hard > 0 else 0,
            mean_time_per_sample=total_time / total if total > 0 else 0,
            raw_results=results,
        )


def main():
    parser = argparse.ArgumentParser(description="BitNet Evaluation Harness")
    parser.add_argument("--benchmark", "-b", type=str, help="Benchmark to run")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--difficulty", "-d", default="extreme",
                        choices=["extreme", "easy", "hard", "all"])
    parser.add_argument("--samples", "-n", type=int, help="Max samples")
    parser.add_argument("--output", "-o", type=Path, help="Output directory")
    parser.add_argument("--model", "-m", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--quiet", "-q", action="store_true")
    
    args = parser.parse_args()
    
    # Initialize harness
    harness = BitNetHarness(
        model_path=args.model,
        verbose=not args.quiet
    )
    
    runner = BenchmarkRunner(harness)
    
    # Determine benchmarks to run
    if args.all:
        benchmarks = list(BENCHMARKS.keys())
    elif args.benchmark:
        benchmarks = [args.benchmark]
    else:
        print("Specify --benchmark NAME or --all")
        print(f"Available: {', '.join(BENCHMARKS.keys())}")
        sys.exit(1)
    
    # Run benchmarks
    all_results = {}
    
    for benchmark in benchmarks:
        try:
            result = runner.run_benchmark(
                benchmark,
                difficulty=args.difficulty,
                max_samples=args.samples
            )
            all_results[benchmark] = asdict(result)
            
            print(f"\n{benchmark}:")
            print(f"  Overall: {result.accuracy_overall:.1%}")
            print(f"  Easy: {result.accuracy_easy:.1%} ({result.samples_easy} samples)")
            print(f"  Hard: {result.accuracy_hard:.1%} ({result.samples_hard} samples)")
            
        except Exception as e:
            print(f"Error running {benchmark}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = args.output / f"results_{timestamp}.json"
        
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

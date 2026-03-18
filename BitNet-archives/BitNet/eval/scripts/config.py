"""
BitNet Evaluation Configuration
Centralized settings for all evaluation scripts.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import platform
import os

# ============================================================
# PATHS
# ============================================================

EVAL_DIR = Path(__file__).parent.parent
PROJECT_ROOT = EVAL_DIR.parent
BUILD_DIR = PROJECT_ROOT / "build"
MODELS_DIR = PROJECT_ROOT / "models"

# Inference binary
if platform.system() == "Windows":
    LLAMA_CLI = BUILD_DIR / "bin" / "Release" / "llama-cli.exe"
    if not LLAMA_CLI.exists():
        LLAMA_CLI = BUILD_DIR / "bin" / "llama-cli.exe"
else:
    LLAMA_CLI = BUILD_DIR / "bin" / "llama-cli"

# Default model
DEFAULT_MODEL = MODELS_DIR / "BitNet-b1.58-2B-4T" / "ggml-model-i2_s.gguf"


# ============================================================
# INFERENCE SETTINGS (LOCKED FOR EXPERIMENTS)
# ============================================================

@dataclass
class InferenceSettings:
    """Locked inference settings for experimental consistency."""
    top_k: int = 42
    top_p: float = 0.963
    temperature: float = 0.528
    ctx_size: int = 4096
    n_predict: int = 1536
    threads: int = field(default_factory=lambda: os.cpu_count() or 4)
    n_gpu_layers: int = 0  # CPU inference
    batch_size: int = 1
    
    def to_cli_args(self) -> List[str]:
        """Convert settings to CLI arguments."""
        return [
            "--top-k", str(self.top_k),
            "--top-p", str(self.top_p),
            "--temp", str(self.temperature),
            "-c", str(self.ctx_size),
            "-n", str(self.n_predict),
            "-t", str(self.threads),
            "-ngl", str(self.n_gpu_layers),
            "-b", str(self.batch_size),
        ]

# Singleton instance
INFERENCE_SETTINGS = InferenceSettings()


# ============================================================
# BENCHMARK DEFINITIONS
# ============================================================

@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark."""
    name: str
    dataset_id: str
    subset: Optional[str] = None
    split: str = "test"
    metric: str = "accuracy"
    difficulty_field: Optional[str] = None
    prompt_template: str = ""
    answer_field: str = "answer"
    question_field: str = "question"
    choices_field: Optional[str] = None
    
BENCHMARKS: Dict[str, BenchmarkConfig] = {
    # Tier 1: Standard ML Evals
    "mmlu": BenchmarkConfig(
        name="MMLU",
        dataset_id="cais/mmlu",
        subset="all",
        split="test",
        metric="accuracy",
        question_field="question",
        choices_field="choices",
        answer_field="answer",
        prompt_template="""Question: {question}

Choices:
A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

Answer with just the letter (A, B, C, or D):"""
    ),
    
    "hellaswag": BenchmarkConfig(
        name="HellaSwag",
        dataset_id="Rowan/hellaswag",
        split="validation",
        metric="accuracy",
        question_field="ctx",
        choices_field="endings",
        answer_field="label",
        prompt_template="""Complete the following sentence in the most logical way.

Context: {context}

Options:
A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

Answer with just the letter (A, B, C, or D):"""
    ),
    
    "arc_challenge": BenchmarkConfig(
        name="ARC-Challenge",
        dataset_id="allenai/ai2_arc",
        subset="ARC-Challenge",
        split="test",
        metric="accuracy",
        question_field="question",
        choices_field="choices",
        answer_field="answerKey",
        prompt_template="""Question: {question}

{choices}

Answer with just the letter:"""
    ),
    
    "winogrande": BenchmarkConfig(
        name="WinoGrande",
        dataset_id="allenai/winogrande",
        subset="winogrande_xl",
        split="validation",
        metric="accuracy",
        question_field="sentence",
        answer_field="answer",
        prompt_template="""Fill in the blank with either option 1 or option 2.

Sentence: {sentence}
Option 1: {option1}
Option 2: {option2}

Answer with just the number (1 or 2):"""
    ),
    
    # Tier 2: Mathematics
    "gsm8k": BenchmarkConfig(
        name="GSM8K",
        dataset_id="openai/gsm8k",
        subset="main",
        split="test",
        metric="accuracy",
        question_field="question",
        answer_field="answer",
        difficulty_field="answer",  # Longer solutions = harder
        prompt_template="""Solve the following math problem step by step. End your response with "The answer is [NUMBER]".

Problem: {question}

Solution:"""
    ),
    
    "math": BenchmarkConfig(
        name="MATH",
        dataset_id="lighteval/MATH",
        subset="all",
        split="test",
        metric="accuracy",
        question_field="problem",
        answer_field="solution",
        difficulty_field="level",  # Explicit 1-5 scale
        prompt_template="""Solve the following mathematics problem. Show your work and provide the final answer.

Problem: {problem}

Solution:"""
    ),
    
    # Tier 3: Physics & Science
    "gpqa": BenchmarkConfig(
        name="GPQA",
        dataset_id="Idavidrein/gpqa",
        subset="gpqa_diamond",
        split="train",  # GPQA only has train split
        metric="accuracy",
        question_field="Question",
        choices_field="choices",  # Will need custom handling
        answer_field="Correct Answer",
        prompt_template="""Answer the following graduate-level science question.

Question: {question}

Choices:
A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

Answer with just the letter (A, B, C, or D):"""
    ),
    
    "sciq": BenchmarkConfig(
        name="SciQ",
        dataset_id="allenai/sciq",
        split="test",
        metric="accuracy",
        question_field="question",
        answer_field="correct_answer",
        prompt_template="""Answer the following science question.

Question: {question}

Support: {support}

Answer:"""
    ),
}


# ============================================================
# DIFFICULTY SAMPLING
# ============================================================

@dataclass 
class SamplingConfig:
    """Configuration for extreme difficulty sampling."""
    easiest_percentile: int = 10  # Bottom 10%
    toughest_percentile: int = 90  # Top 10%
    samples_per_tail: int = 50
    random_seed: int = 42

SAMPLING_CONFIG = SamplingConfig()


# ============================================================
# SIGNAL METRICS
# ============================================================

@dataclass
class SignalMetricsConfig:
    """Configuration for signal processing metrics."""
    capture_entropy: bool = True
    capture_logits: bool = True
    capture_token_distribution: bool = True
    stability_runs: int = 10
    
SIGNAL_CONFIG = SignalMetricsConfig()

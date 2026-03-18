#!/usr/bin/env python3
"""
Signal Processing Metrics for BitNet Optimization Experiments

Captures entropy, token distribution, stability, and calibration metrics
specifically designed to measure dithering and signal processing effects.
"""

import math
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from collections import Counter


@dataclass
class EntropyMetrics:
    """Entropy measurements across a generation."""
    per_token: List[float] = field(default_factory=list)
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    decay_slope: float = 0.0  # Linear regression slope over positions
    
    @classmethod
    def from_logits(cls, logits_per_position: List[List[float]]) -> "EntropyMetrics":
        """Compute entropy metrics from raw logits."""
        entropies = []
        
        for logits in logits_per_position:
            # Convert logits to probabilities via softmax
            max_logit = max(logits)
            exp_logits = [math.exp(l - max_logit) for l in logits]
            sum_exp = sum(exp_logits)
            probs = [e / sum_exp for e in exp_logits]
            
            # Compute entropy: -sum(p * log(p))
            entropy = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)
            entropies.append(entropy)
        
        if not entropies:
            return cls()
        
        # Compute decay slope (linear regression)
        n = len(entropies)
        if n > 1:
            x_mean = (n - 1) / 2
            y_mean = sum(entropies) / n
            numerator = sum((i - x_mean) * (e - y_mean) for i, e in enumerate(entropies))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            slope = numerator / denominator if denominator != 0 else 0
        else:
            slope = 0
        
        return cls(
            per_token=entropies,
            mean=sum(entropies) / len(entropies),
            std=np.std(entropies) if len(entropies) > 1 else 0,
            min=min(entropies),
            max=max(entropies),
            decay_slope=slope,
        )


@dataclass
class TokenDistributionMetrics:
    """Token distribution analysis."""
    total_tokens: int = 0
    unique_tokens: int = 0
    vocabulary_utilization: float = 0.0  # unique / total
    top_k_mass: float = 0.0  # Probability mass in top-k
    entropy_rate: float = 0.0  # Bits per token
    most_common: List[Tuple[str, int]] = field(default_factory=list)
    
    @classmethod
    def from_tokens(
        cls,
        tokens: List[str],
        top_k: int = 42,
        vocab_size: int = 32000  # Approximate for most LLMs
    ) -> "TokenDistributionMetrics":
        """Compute distribution metrics from generated tokens."""
        if not tokens:
            return cls()
        
        counter = Counter(tokens)
        total = len(tokens)
        unique = len(counter)
        
        # Vocabulary utilization
        vocab_util = unique / total if total > 0 else 0
        
        # Top-k mass (what fraction of tokens are in top-k most common)
        top_k_tokens = counter.most_common(top_k)
        top_k_count = sum(count for _, count in top_k_tokens)
        top_k_mass = top_k_count / total if total > 0 else 0
        
        # Entropy rate
        probs = [count / total for count in counter.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        
        return cls(
            total_tokens=total,
            unique_tokens=unique,
            vocabulary_utilization=vocab_util,
            top_k_mass=top_k_mass,
            entropy_rate=entropy,
            most_common=top_k_tokens[:10],
        )


@dataclass
class StabilityMetrics:
    """Response stability across multiple runs."""
    n_runs: int = 0
    token_agreement_rate: float = 0.0  # Fraction of positions with same token
    first_divergence_position: float = 0.0  # Mean position of first difference
    semantic_similarity: float = 0.0  # If embeddings available
    answer_consistency: float = 0.0  # For objective questions
    
    @classmethod
    def from_runs(cls, responses: List[str]) -> "StabilityMetrics":
        """Compute stability metrics from multiple runs of same prompt."""
        if len(responses) < 2:
            return cls(n_runs=len(responses))
        
        n_runs = len(responses)
        
        # Tokenize (simple whitespace split)
        tokenized = [r.split() for r in responses]
        min_len = min(len(t) for t in tokenized)
        
        if min_len == 0:
            return cls(n_runs=n_runs)
        
        # Token agreement rate
        agreements = 0
        total_positions = 0
        
        for pos in range(min_len):
            tokens_at_pos = [t[pos] for t in tokenized]
            if len(set(tokens_at_pos)) == 1:
                agreements += 1
            total_positions += 1
        
        agreement_rate = agreements / total_positions if total_positions > 0 else 0
        
        # First divergence position
        divergence_positions = []
        reference = tokenized[0]
        
        for tokens in tokenized[1:]:
            for i, (t1, t2) in enumerate(zip(reference, tokens)):
                if t1 != t2:
                    divergence_positions.append(i)
                    break
            else:
                divergence_positions.append(min(len(reference), len(tokens)))
        
        mean_divergence = sum(divergence_positions) / len(divergence_positions) if divergence_positions else min_len
        
        return cls(
            n_runs=n_runs,
            token_agreement_rate=agreement_rate,
            first_divergence_position=mean_divergence,
            semantic_similarity=0.0,  # Requires embeddings
            answer_consistency=0.0,  # Requires answer extraction
        )


@dataclass
class CalibrationMetrics:
    """Probability calibration metrics."""
    expected_calibration_error: float = 0.0  # ECE
    max_calibration_error: float = 0.0  # MCE
    mean_confidence: float = 0.0
    overconfidence_rate: float = 0.0  # Fraction where confidence > accuracy
    confidence_histogram: List[int] = field(default_factory=list)
    
    @classmethod
    def from_predictions(
        cls,
        confidences: List[float],
        correct: List[bool],
        n_bins: int = 10
    ) -> "CalibrationMetrics":
        """Compute calibration metrics from predictions."""
        if not confidences or not correct:
            return cls()
        
        n = len(confidences)
        
        # Bin predictions by confidence
        bins = [[] for _ in range(n_bins)]
        
        for conf, corr in zip(confidences, correct):
            bin_idx = min(int(conf * n_bins), n_bins - 1)
            bins[bin_idx].append((conf, corr))
        
        # Compute ECE and MCE
        ece = 0.0
        mce = 0.0
        histogram = []
        
        for bin_data in bins:
            if not bin_data:
                histogram.append(0)
                continue
            
            histogram.append(len(bin_data))
            
            avg_confidence = sum(c for c, _ in bin_data) / len(bin_data)
            avg_accuracy = sum(1 for _, c in bin_data if c) / len(bin_data)
            
            bin_error = abs(avg_confidence - avg_accuracy)
            ece += (len(bin_data) / n) * bin_error
            mce = max(mce, bin_error)
        
        # Compute overconfidence rate
        overconfident = sum(1 for c, corr in zip(confidences, correct) if c > 0.5 and not corr)
        
        return cls(
            expected_calibration_error=ece,
            max_calibration_error=mce,
            mean_confidence=sum(confidences) / n,
            overconfidence_rate=overconfident / n,
            confidence_histogram=histogram,
        )


@dataclass
class LogitStatistics:
    """Statistics on raw logit values."""
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    range: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    
    @classmethod
    def from_logits(cls, logits: List[float]) -> "LogitStatistics":
        """Compute statistics from logit values."""
        if not logits:
            return cls()
        
        arr = np.array(logits)
        mean = np.mean(arr)
        std = np.std(arr)
        
        # Skewness and kurtosis
        if std > 0:
            skewness = np.mean(((arr - mean) / std) ** 3)
            kurtosis = np.mean(((arr - mean) / std) ** 4) - 3
        else:
            skewness = 0
            kurtosis = 0
        
        return cls(
            mean=float(mean),
            std=float(std),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            range=float(np.max(arr) - np.min(arr)),
            skewness=float(skewness),
            kurtosis=float(kurtosis),
        )


@dataclass
class SignalMetricsReport:
    """Complete signal metrics report for an experiment."""
    experiment_id: str = ""
    timestamp: str = ""
    
    entropy: Optional[EntropyMetrics] = None
    token_distribution: Optional[TokenDistributionMetrics] = None
    stability: Optional[StabilityMetrics] = None
    calibration: Optional[CalibrationMetrics] = None
    logit_stats: Optional[LogitStatistics] = None
    
    # Composite score
    signal_quality_score: float = 0.0
    
    def compute_signal_quality_score(
        self,
        baseline: Optional["SignalMetricsReport"] = None,
        weights: Dict[str, float] = None
    ) -> float:
        """
        Compute composite Signal Quality Score.
        
        SQS = w1*(1 - entropy_decay) + w2*stability + w3*(1-ECE) + w4*vocab_util
        """
        weights = weights or {
            "entropy": 0.25,
            "stability": 0.30,
            "calibration": 0.25,
            "diversity": 0.20,
        }
        
        score = 0.0
        
        # Entropy component (lower decay is better)
        if self.entropy:
            entropy_score = 1.0 - min(abs(self.entropy.decay_slope), 1.0)
            score += weights["entropy"] * entropy_score
        
        # Stability component
        if self.stability:
            score += weights["stability"] * self.stability.token_agreement_rate
        
        # Calibration component (lower ECE is better)
        if self.calibration:
            cal_score = 1.0 - min(self.calibration.expected_calibration_error, 1.0)
            score += weights["calibration"] * cal_score
        
        # Diversity component
        if self.token_distribution:
            score += weights["diversity"] * min(self.token_distribution.vocabulary_utilization * 5, 1.0)
        
        self.signal_quality_score = score
        return score
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "entropy": asdict(self.entropy) if self.entropy else None,
            "token_distribution": asdict(self.token_distribution) if self.token_distribution else None,
            "stability": asdict(self.stability) if self.stability else None,
            "calibration": asdict(self.calibration) if self.calibration else None,
            "logit_stats": asdict(self.logit_stats) if self.logit_stats else None,
            "signal_quality_score": self.signal_quality_score,
        }
    
    def save(self, path: Path) -> None:
        """Save metrics to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "SignalMetricsReport":
        """Load metrics from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        report = cls(
            experiment_id=data.get("experiment_id", ""),
            timestamp=data.get("timestamp", ""),
        )
        
        if data.get("entropy"):
            report.entropy = EntropyMetrics(**data["entropy"])
        if data.get("token_distribution"):
            # Handle tuple conversion for most_common
            td = data["token_distribution"]
            td["most_common"] = [tuple(x) for x in td.get("most_common", [])]
            report.token_distribution = TokenDistributionMetrics(**td)
        if data.get("stability"):
            report.stability = StabilityMetrics(**data["stability"])
        if data.get("calibration"):
            report.calibration = CalibrationMetrics(**data["calibration"])
        if data.get("logit_stats"):
            report.logit_stats = LogitStatistics(**data["logit_stats"])
        
        report.signal_quality_score = data.get("signal_quality_score", 0.0)
        
        return report


def compare_metrics(
    baseline: SignalMetricsReport,
    experimental: SignalMetricsReport
) -> Dict[str, Any]:
    """
    Compare two signal metrics reports.
    
    Returns delta and interpretation for each metric.
    """
    comparison = {
        "experiment_id": experimental.experiment_id,
        "vs_baseline": baseline.experiment_id,
        "deltas": {},
        "interpretations": {},
    }
    
    # Entropy comparison
    if baseline.entropy and experimental.entropy:
        delta_mean = experimental.entropy.mean - baseline.entropy.mean
        delta_decay = experimental.entropy.decay_slope - baseline.entropy.decay_slope
        
        comparison["deltas"]["entropy_mean"] = delta_mean
        comparison["deltas"]["entropy_decay"] = delta_decay
        
        if delta_decay < -0.01:
            comparison["interpretations"]["entropy"] = "improved (slower decay)"
        elif delta_decay > 0.01:
            comparison["interpretations"]["entropy"] = "degraded (faster decay)"
        else:
            comparison["interpretations"]["entropy"] = "unchanged"
    
    # Stability comparison
    if baseline.stability and experimental.stability:
        delta = experimental.stability.token_agreement_rate - baseline.stability.token_agreement_rate
        comparison["deltas"]["stability"] = delta
        
        if delta > 0.02:
            comparison["interpretations"]["stability"] = "improved"
        elif delta < -0.02:
            comparison["interpretations"]["stability"] = "degraded"
        else:
            comparison["interpretations"]["stability"] = "unchanged"
    
    # Calibration comparison
    if baseline.calibration and experimental.calibration:
        delta = experimental.calibration.expected_calibration_error - baseline.calibration.expected_calibration_error
        comparison["deltas"]["ece"] = delta
        
        if delta < -0.01:
            comparison["interpretations"]["calibration"] = "improved"
        elif delta > 0.01:
            comparison["interpretations"]["calibration"] = "degraded"
        else:
            comparison["interpretations"]["calibration"] = "unchanged"
    
    # Signal quality score comparison
    delta_sqs = experimental.signal_quality_score - baseline.signal_quality_score
    comparison["deltas"]["signal_quality_score"] = delta_sqs
    
    if delta_sqs > 0.05:
        comparison["interpretations"]["overall"] = "significant improvement"
    elif delta_sqs > 0.02:
        comparison["interpretations"]["overall"] = "modest improvement"
    elif delta_sqs < -0.05:
        comparison["interpretations"]["overall"] = "significant degradation"
    elif delta_sqs < -0.02:
        comparison["interpretations"]["overall"] = "modest degradation"
    else:
        comparison["interpretations"]["overall"] = "no significant change"
    
    return comparison


if __name__ == "__main__":
    # Example usage
    print("Signal Metrics Module")
    print("=" * 40)
    
    # Example: compute metrics from dummy data
    tokens = ["the", "quick", "brown", "fox", "the", "lazy", "dog", "the"]
    dist = TokenDistributionMetrics.from_tokens(tokens)
    print(f"Token distribution: {dist}")
    
    # Stability from multiple runs
    responses = [
        "The answer is 42.",
        "The answer is 42.",
        "The answer is forty-two.",
    ]
    stability = StabilityMetrics.from_runs(responses)
    print(f"Stability: {stability}")

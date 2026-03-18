#!/usr/bin/env python3
"""Test if logit variance predicts entropy without computing softmax."""

import numpy as np
import torch
from typing import List, Tuple


def softmax(logits: np.ndarray) -> np.ndarray:
    """Standard softmax."""
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def entropy(probs: np.ndarray) -> float:
    """Shannon entropy in bits."""
    probs = probs[probs > 1e-10]
    return -np.sum(probs * np.log2(probs))


def logit_variance(logits: np.ndarray) -> float:
    """Variance of logits weighted by softmax probabilities."""
    probs = softmax(logits)
    mean = np.sum(probs * logits)
    return np.sum(probs * (logits - mean) ** 2)


def logit_variance_cheap(logits: np.ndarray) -> float:
    """Cheap variance estimate without softmax - just first two moments of raw logits."""
    # This is the variance of the logits themselves, not weighted by probabilities
    return np.var(logits)


def logit_divergence_score(logits: np.ndarray) -> float:
    """LDS = (max - mean) / std"""
    max_logit = np.max(logits)
    mean_logit = np.mean(logits)
    std_logit = np.std(logits)
    if std_logit < 1e-10:
        return 0.0
    return (max_logit - mean_logit) / std_logit


def generate_test_logits(n_vocab: int, n_samples: int = 100):
    """Generate random logits and compute both variance and entropy."""
    results = []

    for _ in range(n_samples):
        # Generate random logits with varying "sharpness"
        sharpness = np.random.uniform(0.1, 10.0)
        logits = np.random.randn(n_vocab) * sharpness

        probs = softmax(logits)
        ent = entropy(probs)
        var_weighted = logit_variance(logits)
        var_cheap = logit_variance_cheap(logits)
        lds = logit_divergence_score(logits)

        results.append([float(ent), float(var_weighted), float(var_cheap), float(lds)])

    return results


def test_correlation():
    """Test if variance/LDS correlates with entropy."""
    n_vocab = 1000
    results = generate_test_logits(n_vocab, n_samples=500)

    entropies = np.array([r[0] for r in results])
    var_weighted = np.array([r[1] for r in results])
    var_cheap = np.array([r[2] for r in results])
    lds = np.array([r[3] for r in results])

    # Compute correlations: negative variance = high variance means low entropy
    # So we expect positive correlation with -variance
    corr_weighted = np.corrcoef(entropies, var_weighted)[0, 1]
    corr_cheap = np.corrcoef(entropies, var_cheap)[0, 1]
    corr_lds = np.corrcoef(entropies, lds)[0, 1]

    print(f"Correlation with entropy (n={n_vocab}, samples={len(results)}):")
    print(
        f"  Weighted variance:  r = {corr_weighted:.4f} (expect: negative = high var → low entropy)"
    )
    print(
        f"  Cheap variance:     r = {corr_cheap:.4f} (expect: negative = high var → low entropy)"
    )
    print(
        f"  LDS (max-mean)/std: r = {corr_lds:.4f} (expect: positive = high LDS → low entropy)"
    )

    # Test with realistic model outputs (sharp distributions)
    print("\n--- Realistic model test ---")
    for sharpness in [0.5, 1.0, 2.0, 5.0, 10.0]:
        logits = np.random.randn(n_vocab) * sharpness
        probs = softmax(logits)
        ent = entropy(probs)
        lds = logit_divergence_score(logits)

        print(f"  sharpness={sharpness:4.1f}: entropy={ent:.2f} bits, LDS={lds:.2f}")


def test_single_example():
    """Detailed single example."""
    n_vocab = 10
    logits = np.array([2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5])

    probs = softmax(logits)
    ent = entropy(probs)
    var_w = logit_variance(logits)
    var_c = logit_variance_cheap(logits)
    lds = logit_divergence_score(logits)

    print(f"Single example (n={n_vocab}):")
    print(f"  Probs:    {probs[:5]}...")
    print(f"  Entropy:  {ent:.3f} bits")
    print(f"  Var(weighted): {var_w:.3f}")
    print(f"  Var(cheap):    {var_c:.3f}")
    print(f"  LDS:          {lds:.3f}")


if __name__ == "__main__":
    test_single_example()
    print()
    test_correlation()

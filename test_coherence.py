#!/usr/bin/env python3
"""
Coherence Test: Measure if dither focusing improves reasoning chain coherence.

Simulates reasoning tasks by tracking token consistency across sequences.
"""

import numpy as np
from collections import Counter

BAYER = (
    np.array(
        [
            [0, 32, 8, 40, 2, 34, 10, 42],
            [48, 16, 56, 24, 50, 18, 58, 26],
            [12, 44, 4, 36, 14, 46, 6, 38],
            [60, 28, 52, 20, 62, 30, 54, 22],
            [3, 35, 11, 43, 1, 33, 9, 41],
            [51, 19, 59, 27, 49, 17, 57, 25],
            [15, 47, 7, 39, 13, 45, 5, 37],
            [63, 31, 55, 23, 61, 29, 53, 21],
        ],
        dtype=np.float32,
    )
    / 64.0
    - 0.5
)


def dither(logits, position, intensity):
    if intensity <= 0:
        return logits
    r = np.max(logits) - np.min(logits)
    if r < 1e-6:
        r = 1.0
    x = position % 8
    d = np.array([intensity * BAYER[tid % 8, x] * 2 * r for tid in range(len(logits))])
    return logits + d


def sample(logits, temp=1.0, top_k=40):
    if temp != 1.0:
        logits = logits / temp
    if top_k > 0 and top_k < len(logits):
        idx = np.argpartition(logits, -top_k)[-top_k:]
        mask = np.full_like(logits, -1e30)
        mask[idx] = logits[idx]
        logits = mask
    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)
    return np.random.choice(len(probs), p=probs)


def generate_sequence(length, intensity, seed=42):
    """Generate a reasoning-like sequence."""
    np.random.seed(seed)

    vocab_size = 1000
    base_logits = np.zeros(vocab_size)

    reasoning_tokens = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    filler_tokens = list(range(20, 100))

    tokens = []
    for pos in range(length):
        logits = base_logits.copy()

        if pos < len(reasoning_tokens):
            for rt in reasoning_tokens[: pos + 1]:
                logits[rt] = 5.0 + np.random.randn() * 0.5
        else:
            for rt in reasoning_tokens:
                logits[rt] = 5.0 + np.random.randn() * 0.5

        for ft in filler_tokens:
            logits[ft] = np.random.randn() * 2

        logits = dither(logits, pos, intensity)

        tok = sample(logits, temp=0.7, top_k=40)
        tokens.append(tok)
        base_logits[tok] += 0.1

    return tokens


def coherence_score(tokens):
    """Measure coherence: how consistently reasoning tokens appear."""
    reasoning_count = sum(1 for t in tokens if 10 <= t <= 19)
    unique_reasoning = len(set(t for t in tokens if 10 <= t <= 19))

    if reasoning_count == 0:
        return 0.0
    return unique_reasoning / reasoning_count


def run_test():
    print("=== COHERENCE TEST ===\n")
    print("Simulating reasoning chain (10 reasoning tokens, 80 filler)")
    print("High coherence = staying on topic (fewer unique reasoning tokens)")
    print("Low coherence = wandering (more unique reasoning tokens)\n")

    lengths = [20, 50, 100]
    intensities = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15]

    print(f"{'Length':<8} {'Intensity':<10} {'Coherence':<12} {'Reasoning':<12}")
    print("-" * 45)

    for length in lengths:
        for intensity in intensities:
            seqs = [generate_sequence(length, intensity, seed=s) for s in range(50)]

            coherences = [coherence_score(s) for s in seqs]
            avg_coherence = np.mean(coherences)

            reasoning_totals = [sum(1 for t in s if 10 <= t <= 19) for s in seqs]
            avg_reasoning = np.mean(reasoning_totals)

            print(
                f"{length:<8} {intensity:<10.2f} {avg_coherence:<12.3f} {avg_reasoning:<12.1f}"
            )

    print("\n=== INTERPRETATION ===")
    print("Lower coherence = MORE focused (fewer unique topic tokens)")
    print("Higher coherence = LESS focused (wandering across more topics)")
    print("\nIf dither works:")
    print("  - Higher intensity → lower coherence → more focused output")
    print("  - This should produce more consistent reasoning chains")


if __name__ == "__main__":
    run_test()

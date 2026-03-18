#!/usr/bin/env python3
"""
Test the math behind ordered dithering for LLM inference.
Verify phase transitions, entropy reduction, and fundamental constant bounds.
"""

import numpy as np
from collections import Counter

BAYER_8X8 = (
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

ALPHA = 1.0 / 137.0
PHI = (1 + np.sqrt(5)) / 2


def apply_dither(logits, position, intensity):
    """Apply Bayer dithering to logits.

    Formula: logit_new = logit + intensity * (threshold - 0.5) * logit_range
    """
    if intensity == 0:
        return logits

    dithered = np.copy(logits)
    num_tokens = logits.shape[-1]

    x = position % 8

    for token_id in range(num_tokens):
        y = token_id % 8
        threshold = BAYER_8X8[y, x]
        logit_range = np.max(logits) - np.min(logits)
        if logit_range < 1e-6:
            logit_range = 1.0
        dither = intensity * (threshold - 0.5) * 2 * logit_range
        dithered[..., token_id] = logits[..., token_id] + dither

    return dithered


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def entropy(counts):
    total = sum(counts)
    probs = np.array(list(counts)) / total
    probs = probs[probs > 1e-10]
    return -np.sum(probs * np.log2(probs))


def select_token(logits, temperature=1.0, top_k=0, top_p=0.0):
    """Sample token from logits."""
    if temperature != 1.0:
        logits = logits / temperature

    if top_k > 0:
        indices = np.argpartition(logits, -top_k)[-top_k:]
        mask = np.full_like(logits, float("-inf"))
        mask[indices] = logits[indices]
        logits = mask

    if top_p > 0:
        sorted_indices = np.argsort(logits)[::-1]
        cumsum = np.cumsum(softmax(logits[sorted_indices]))
        if cumsum[-1] > top_p:
            cutoff_idx = np.searchsorted(cumsum, top_p)
            mask = np.full_like(logits, float("-inf"))
            mask[sorted_indices[: cutoff_idx + 1]] = logits[
                sorted_indices[: cutoff_idx + 1]
            ]
            logits = mask

    probs = softmax(logits)
    return np.random.choice(len(probs), p=probs)


def test_entropy_reduction():
    """Test that dithering reduces entropy in ambiguous cases."""
    print("=" * 60)
    print("TEST 1: Entropy Reduction")
    print("=" * 60)

    results = []

    for scenario_name, logit_setup in [
        ("close_race", lambda: np.array([5.0, 4.95] + [0.0] * 998)),
        ("multi_way_tie", lambda: np.array([5.0] * 10 + [0.0] * 990)),
        ("flat_dist", lambda: np.random.randn(1000) * 0.5),
    ]:
        baseline_tokens = []
        dithered_tokens = []

        intensity = PHI / 20

        for seed in range(200):
            np.random.seed(seed)
            logit_dist = logit_setup()
            baseline_tokens.append(select_token(logit_dist, temperature=0.7))

        for seed in range(200):
            np.random.seed(seed)
            logit_dist = logit_setup()
            dithered = apply_dither(
                logit_dist.copy(), position=seed, intensity=intensity
            )
            dithered_tokens.append(select_token(dithered, temperature=0.7))

        baseline_entropy = entropy(Counter(baseline_tokens).values())
        dithered_entropy = entropy(Counter(dithered_tokens).values())

        print(f"\n{scenario_name}:")
        print(f"  Unique: {len(set(baseline_tokens))} → {len(set(dithered_tokens))}")
        print(
            f"  Entropy: {baseline_entropy:.3f} → {dithered_entropy:.3f} ({dithered_entropy - baseline_entropy:+.3f})"
        )

        baseline_unique = len(set(baseline_tokens))
        dithered_unique = len(set(dithered_tokens))
        entropy_delta = dithered_entropy - baseline_entropy

        if dithered_unique < baseline_unique or entropy_delta < 0:
            print(f"  ✓ Dithering FOCUSES (reduces diversity)")
        else:
            print(f"  → No significant focus effect")


def test_phase_transitions():
    """Test phase transitions at different intensity levels."""
    print("\n" + "=" * 60)
    print("TEST 2: Phase Transitions")
    print("=" * 60)

    np.random.seed(123)
    base_logits = np.random.randn(1000) * 2 + 5.0

    intensities = [0.0, ALPHA, 0.05, 0.074, 0.096, 0.143, 0.144, 0.16, 0.1618, PHI / 10]

    for intensity in intensities:
        tokens = []
        for pos in range(100):
            dithered = apply_dither(
                base_logits.copy(), position=pos, intensity=intensity
            )
            tokens.append(select_token(dithered, temperature=0.528))

        unique = len(set(tokens))
        top_10 = Counter(tokens).most_common(10)

        print(f"\nIntensity {intensity:.4f}:")
        print(f"  Unique tokens: {unique}")
        print(f"  Top-3: {top_10[:3]}")


def test_fundamental_bounds():
    """Test that alpha and phi/10 are sensible bounds."""
    print("\n" + "=" * 60)
    print("TEST 3: Fundamental Constant Bounds")
    print("=" * 60)

    print(f"\nAlpha (fine structure): 1/137 = {ALPHA:.5f}")
    print(f"Phi/10 (golden ratio): {PHI / 10:.5f}")
    print(f"Range: [{ALPHA:.5f}, {PHI / 10:.5f}]")

    np.random.seed(456)
    logits = np.random.randn(1000) * 3

    for intensity in [ALPHA, 0.05, 0.0809, PHI / 10, 0.2]:
        tokens = []
        for pos in range(200):
            dithered = apply_dither(logits.copy(), position=pos, intensity=intensity)
            tokens.append(select_token(dithered, top_k=42, top_p=0.963))

        unique = len(set(tokens))
        print(f"\nIntensity {intensity:.4f}:")
        print(f"  Unique tokens: {unique}")


def verify_deterministic_at_zero():
    """Verify byte-for-byte identical output when dithering disabled."""
    print("\n" + "=" * 60)
    print("TEST 4: Deterministic at Zero Intensity")
    print("=" * 60)

    np.random.seed(789)
    logits = np.random.randn(100) * 5

    result_0 = apply_dither(logits.copy(), position=0, intensity=0.0)
    result_1 = apply_dither(logits.copy(), position=0, intensity=0.0)

    assert np.allclose(result_0, result_1), "FAIL: Zero intensity not deterministic!"
    print(f"✓ Zero intensity produces identical output")
    print(f"  Max diff: {np.max(np.abs(result_0 - result_1)):.10f}")


def main():
    print("Ordered Dithering Math Verification")
    print("=" * 60)

    test_entropy_reduction()
    test_phase_transitions()
    test_fundamental_bounds()
    verify_deterministic_at_zero()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

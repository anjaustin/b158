#!/usr/bin/env python3
"""
Ordered Dithering for LLM Logit Space

Applies Bayer matrix dithering patterns to logit distributions
to enhance the "resolution" of token selection in BitNet b1.58.

Based on classic ordered dithering from digital halftoning,
adapted for the discrete decision space of language models.

Intensity bounds: [φ/100, φ/10] = [0.01618, 0.1618]
where φ = golden ratio ≈ 1.618033988749895
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

# Golden ratio bounds
PHI = 1.618033988749895
INTENSITY_MIN = PHI / 100  # 0.01618
INTENSITY_MAX = PHI / 10   # 0.1618


class BayerMatrix:
    """
    Bayer ordered dithering matrices.
    
    These are threshold matrices with blue noise properties -
    the values are distributed to minimize low-frequency patterns.
    """
    
    # 2x2 Bayer matrix
    BAYER_2 = np.array([
        [0, 2],
        [3, 1],
    ], dtype=np.float32) / 4.0
    
    # 4x4 Bayer matrix
    BAYER_4 = np.array([
        [ 0,  8,  2, 10],
        [12,  4, 14,  6],
        [ 3, 11,  1,  9],
        [15,  7, 13,  5],
    ], dtype=np.float32) / 16.0
    
    # 8x8 Bayer matrix (classic)
    BAYER_8 = np.array([
        [ 0, 32,  8, 40,  2, 34, 10, 42],
        [48, 16, 56, 24, 50, 18, 58, 26],
        [12, 44,  4, 36, 14, 46,  6, 38],
        [60, 28, 52, 20, 62, 30, 54, 22],
        [ 3, 35, 11, 43,  1, 33,  9, 41],
        [51, 19, 59, 27, 49, 17, 57, 25],
        [15, 47,  7, 39, 13, 45,  5, 37],
        [63, 31, 55, 23, 61, 29, 53, 21],
    ], dtype=np.float32) / 64.0
    
    @classmethod
    def get(cls, size: int = 8) -> np.ndarray:
        """Get Bayer matrix of specified size."""
        if size == 2:
            return cls.BAYER_2
        elif size == 4:
            return cls.BAYER_4
        elif size == 8:
            return cls.BAYER_8
        else:
            raise ValueError(f"Unsupported Bayer matrix size: {size}. Use 2, 4, or 8.")


@dataclass
class DitherConfig:
    """Configuration for ordered dithering."""
    
    intensity: float = PHI / 20  # Default: mid-range (0.0809)
    matrix_size: int = 8
    
    # Coordinate mapping for tokens
    use_token_id_as_y: bool = True  # token_id % matrix_size
    
    # Scaling mode
    scale_to_logit_range: bool = True  # Scale dither to local logit range
    
    # Bounds
    intensity_min: float = INTENSITY_MIN
    intensity_max: float = INTENSITY_MAX
    
    def __post_init__(self):
        self.intensity = np.clip(self.intensity, self.intensity_min, self.intensity_max)
        self.matrix = BayerMatrix.get(self.matrix_size)
    
    def set_intensity(self, value: float) -> None:
        """Set intensity within golden ratio bounds."""
        self.intensity = np.clip(value, self.intensity_min, self.intensity_max)


@dataclass
class TokenData:
    """Represents a token with its logit and probability."""
    id: int
    logit: float
    prob: float = 0.0


@dataclass
class DitherState:
    """Tracks state across sequence generation."""
    sequence_position: int = 0
    tokens_generated: List[int] = field(default_factory=list)
    
    # Metrics
    total_dither_applied: float = 0.0
    decisions_affected: int = 0
    
    def advance(self, token_id: int) -> None:
        """Advance state after token selection."""
        self.sequence_position += 1
        self.tokens_generated.append(token_id)
    
    def reset(self) -> None:
        """Reset state for new sequence."""
        self.sequence_position = 0
        self.tokens_generated = []
        self.total_dither_applied = 0.0
        self.decisions_affected = 0


class OrderedDither:
    """
    Ordered dithering sampler for LLM logit space.
    
    Applies Bayer matrix patterns to logit distributions to create
    structured (non-random) perturbations that may enhance the
    "resolution" of token selection decisions.
    
    Usage:
        dither = OrderedDither(intensity=0.05)
        modified_logits = dither.apply(logits, token_ids)
        dither.accept(selected_token_id)
    """
    
    def __init__(self, config: Optional[DitherConfig] = None):
        self.config = config or DitherConfig()
        self.state = DitherState()
    
    @property
    def name(self) -> str:
        return f"ordered-dither-{self.config.matrix_size}x{self.config.matrix_size}"
    
    def get_threshold(self, token_id: int) -> float:
        """
        Get Bayer threshold for current position and token.
        
        X coordinate: sequence position (time dimension)
        Y coordinate: token ID (vocabulary dimension)
        """
        size = self.config.matrix_size
        pos_x = self.state.sequence_position % size
        pos_y = token_id % size
        return self.config.matrix[pos_y, pos_x]
    
    def compute_dither_value(self, threshold: float, logit_range: float) -> float:
        """
        Compute dither value from threshold.
        
        Maps threshold [0, 1] to dither [-0.5, 0.5] * intensity * range
        """
        # Center threshold around 0
        centered = threshold - 0.5
        
        # Scale by intensity and logit range
        if self.config.scale_to_logit_range:
            dither = centered * self.config.intensity * logit_range
        else:
            dither = centered * self.config.intensity
        
        return dither
    
    def apply(
        self,
        logits: np.ndarray,
        token_ids: Optional[np.ndarray] = None,
        return_diagnostics: bool = False
    ) -> np.ndarray:
        """
        Apply ordered dithering to logits.
        
        Args:
            logits: Array of logit values [vocab_size] or [batch, vocab_size]
            token_ids: Token IDs corresponding to logits (default: range(len(logits)))
            return_diagnostics: If True, return (modified_logits, diagnostics)
        
        Returns:
            Modified logits with dithering applied
        """
        logits = np.asarray(logits, dtype=np.float32)
        is_batched = logits.ndim == 2
        
        if not is_batched:
            logits = logits.reshape(1, -1)
        
        batch_size, vocab_size = logits.shape
        
        if token_ids is None:
            token_ids = np.arange(vocab_size)
        
        # Compute logit range for scaling
        logit_max = logits.max(axis=1, keepdims=True)
        logit_min = logits.min(axis=1, keepdims=True)
        logit_range = logit_max - logit_min
        logit_range = np.maximum(logit_range, 1e-8)  # Prevent division by zero
        
        # Get thresholds for all tokens
        size = self.config.matrix_size
        pos_x = self.state.sequence_position % size
        pos_y = token_ids % size
        thresholds = self.config.matrix[pos_y, pos_x]
        
        # Compute dither values
        centered = thresholds - 0.5
        dither_values = centered * self.config.intensity * logit_range.flatten()
        
        # Apply dithering
        modified_logits = logits + dither_values
        
        # Update metrics
        self.state.total_dither_applied += np.abs(dither_values).sum()
        
        if not is_batched:
            modified_logits = modified_logits.flatten()
        
        if return_diagnostics:
            diagnostics = {
                'position': self.state.sequence_position,
                'logit_range': float(logit_range.mean()),
                'dither_magnitude': float(np.abs(dither_values).mean()),
                'thresholds_used': thresholds.tolist()[:10],  # First 10
            }
            return modified_logits, diagnostics
        
        return modified_logits
    
    def accept(self, token_id: int) -> None:
        """
        Accept a selected token and advance state.
        
        Called after token selection to update sequence position.
        """
        self.state.advance(token_id)
    
    def reset(self) -> None:
        """Reset state for new sequence."""
        self.state.reset()
    
    def get_metrics(self) -> dict:
        """Get accumulated metrics."""
        return {
            'sequence_length': self.state.sequence_position,
            'total_dither_applied': self.state.total_dither_applied,
            'mean_dither_per_token': (
                self.state.total_dither_applied / max(1, self.state.sequence_position)
            ),
        }


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute softmax probabilities from logits."""
    scaled = logits / temperature
    exp_logits = np.exp(scaled - scaled.max())
    return exp_logits / exp_logits.sum()


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """Zero out logits outside top-k."""
    if k <= 0 or k >= len(logits):
        return logits
    
    indices = np.argpartition(logits, -k)[-k:]
    mask = np.zeros_like(logits, dtype=bool)
    mask[indices] = True
    
    filtered = logits.copy()
    filtered[~mask] = -np.inf
    return filtered


def top_p_filter(probs: np.ndarray, p: float) -> np.ndarray:
    """Zero out probabilities outside nucleus (top-p)."""
    if p >= 1.0:
        return probs
    
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumsum = np.cumsum(sorted_probs)
    
    # Find cutoff
    cutoff_idx = np.searchsorted(cumsum, p) + 1
    
    mask = np.zeros_like(probs, dtype=bool)
    mask[sorted_indices[:cutoff_idx]] = True
    
    filtered = probs.copy()
    filtered[~mask] = 0
    return filtered / filtered.sum()  # Renormalize


def sample_with_dithering(
    logits: np.ndarray,
    dither: OrderedDither,
    temperature: float = 0.528,
    top_k: int = 42,
    top_p: float = 0.963,
    return_diagnostics: bool = False
) -> int:
    """
    Sample a token using ordered dithering.
    
    Pipeline:
    1. Apply ordered dithering to logits
    2. Apply temperature
    3. Apply top-k filtering
    4. Convert to probabilities (softmax)
    5. Apply top-p filtering
    6. Sample from distribution
    
    Args:
        logits: Raw logit values
        dither: OrderedDither instance
        temperature: Sampling temperature
        top_k: Top-k filtering parameter
        top_p: Nucleus sampling parameter
        return_diagnostics: Return diagnostic info
    
    Returns:
        Selected token ID (or (token_id, diagnostics) if return_diagnostics)
    """
    # 1. Apply ordered dithering
    if return_diagnostics:
        dithered_logits, dither_diag = dither.apply(logits, return_diagnostics=True)
    else:
        dithered_logits = dither.apply(logits)
        dither_diag = {}
    
    # 2. Apply top-k filtering (before temperature for efficiency)
    filtered_logits = top_k_filter(dithered_logits, top_k)
    
    # 3. Apply temperature and convert to probabilities
    probs = softmax(filtered_logits, temperature)
    
    # 4. Apply top-p filtering
    probs = top_p_filter(probs, top_p)
    
    # 5. Sample
    token_id = np.random.choice(len(probs), p=probs)
    
    # 6. Accept (advance dither state)
    dither.accept(token_id)
    
    if return_diagnostics:
        # Find how dithering affected the top candidates
        original_top = np.argsort(logits)[-5:][::-1]
        dithered_top = np.argsort(dithered_logits)[-5:][::-1]
        
        diagnostics = {
            **dither_diag,
            'original_top5': original_top.tolist(),
            'dithered_top5': dithered_top.tolist(),
            'top5_changed': not np.array_equal(original_top, dithered_top),
            'selected_token': int(token_id),
            'selected_prob': float(probs[token_id]),
        }
        return token_id, diagnostics
    
    return token_id


# =============================================================================
# Testing and Visualization
# =============================================================================

def visualize_bayer_pattern(size: int = 8) -> str:
    """Create ASCII visualization of Bayer matrix."""
    matrix = BayerMatrix.get(size)
    chars = " ░▒▓█"
    
    lines = []
    lines.append(f"Bayer {size}x{size} Matrix:")
    lines.append("+" + "-" * (size * 2 + 1) + "+")
    
    for row in matrix:
        line = "| "
        for val in row:
            idx = int(val * (len(chars) - 1))
            line += chars[idx] + " "
        line += "|"
        lines.append(line)
    
    lines.append("+" + "-" * (size * 2 + 1) + "+")
    return "\n".join(lines)


def analyze_dither_effect(
    n_tokens: int = 1000,
    vocab_size: int = 32000,
    intensity: float = PHI / 20,
    seed: int = 42
) -> dict:
    """
    Analyze the effect of dithering on synthetic logit distributions.
    
    Returns statistics on how dithering affects token selection.
    """
    np.random.seed(seed)
    
    dither = OrderedDither(DitherConfig(intensity=intensity))
    
    # Metrics
    rank_changes = []
    top1_changes = 0
    top5_changes = 0
    
    for _ in range(n_tokens):
        # Generate synthetic logits (normal distribution)
        logits = np.random.randn(vocab_size).astype(np.float32)
        
        # Get original ranking
        original_order = np.argsort(logits)[::-1]
        
        # Apply dithering
        dithered = dither.apply(logits)
        dithered_order = np.argsort(dithered)[::-1]
        
        # Measure rank changes
        original_top1 = original_order[0]
        dithered_top1 = dithered_order[0]
        
        if original_top1 != dithered_top1:
            top1_changes += 1
        
        if not np.array_equal(original_order[:5], dithered_order[:5]):
            top5_changes += 1
        
        # Track how far top token moved
        new_rank = np.where(dithered_order == original_top1)[0][0]
        rank_changes.append(new_rank)
        
        dither.accept(int(dithered_top1))
    
    return {
        'intensity': intensity,
        'n_tokens': n_tokens,
        'vocab_size': vocab_size,
        'top1_change_rate': top1_changes / n_tokens,
        'top5_change_rate': top5_changes / n_tokens,
        'mean_rank_change': np.mean(rank_changes),
        'max_rank_change': np.max(rank_changes),
        'metrics': dither.get_metrics(),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Ordered Dithering for LLM Logit Space")
    print("=" * 60)
    print()
    
    # Visualize Bayer matrices
    for size in [2, 4, 8]:
        print(visualize_bayer_pattern(size))
        print()
    
    # Test different intensities
    print("Effect Analysis (synthetic logits):")
    print("-" * 60)
    
    intensities = [INTENSITY_MIN, PHI/50, PHI/20, PHI/10, INTENSITY_MAX]
    
    for intensity in intensities:
        result = analyze_dither_effect(intensity=intensity, n_tokens=500)
        print(f"Intensity: {intensity:.4f}")
        print(f"  Top-1 change rate: {result['top1_change_rate']:.1%}")
        print(f"  Top-5 change rate: {result['top5_change_rate']:.1%}")
        print(f"  Mean rank change:  {result['mean_rank_change']:.2f}")
        print()
    
    print("=" * 60)
    print(f"Intensity bounds: [{INTENSITY_MIN:.5f}, {INTENSITY_MAX:.5f}]")
    print(f"Based on φ (golden ratio) = {PHI:.15f}")
    print("=" * 60)

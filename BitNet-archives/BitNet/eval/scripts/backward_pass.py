#!/usr/bin/env python3
"""
Backward Pass Dithering
=======================

Pass 1: DIVERGE - Generate raw creative material (high intensity)
Pass 2: CONVERGE - Refine with structure (low intensity)

Like diffusion, but for token selection.

Authors: Aaron (Tripp) Josserand-Austin & Droid (Λ)
"""

import subprocess
import sys
from pathlib import Path

BASE_PATH = Path(__file__).parent.parent.parent
CLI = BASE_PATH / "build/bin/llama-cli"
MODEL = BASE_PATH / "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"

# Basin intensities from our topology mapping
INTENSITY_DIVERGE = 0.16    # Basin D - Philosopher mode
INTENSITY_CONVERGE = 0.10   # Basin B - Curator mode


def generate(prompt: str, n_tokens: int, intensity: float, seed: int = 42) -> str:
    """Run inference with given intensity."""
    cmd = [
        str(CLI),
        "-m", str(MODEL),
        "-p", prompt,
        "-n", str(n_tokens),
        "--temp", "0.528",
        "--top-k", "42",
        "--top-p", "0.963",
        "--seed", str(seed),
        "--dither", str(intensity),
        "-ngl", "0",
        "--no-display-prompt",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=str(BASE_PATH))
    return result.stdout.strip()


def backward_pass(prompt: str, n_tokens: int = 64, seed: int = 42) -> dict:
    """
    Two-pass generation:
    1. Diverge: Generate exploratory draft
    2. Converge: Use draft as context, refine
    """
    
    print("=" * 60)
    print("BACKWARD PASS DITHERING")
    print("=" * 60)
    print(f"Prompt: {prompt}")
    print()
    
    # Pass 1: DIVERGE
    print(f"[PASS 1] DIVERGE (intensity={INTENSITY_DIVERGE})")
    print("-" * 40)
    divergent = generate(prompt, n_tokens, INTENSITY_DIVERGE, seed)
    print(divergent)
    print()
    
    # Pass 2: CONVERGE - use divergent output as additional context
    # The idea: "Given this exploratory draft, now synthesize..."
    converge_prompt = f"""{prompt}

[Draft thoughts: {divergent[:200]}...]

Now, synthesizing the above into a clear answer:"""
    
    print(f"[PASS 2] CONVERGE (intensity={INTENSITY_CONVERGE})")
    print("-" * 40)
    convergent = generate(converge_prompt, n_tokens, INTENSITY_CONVERGE, seed)
    print(convergent)
    print()
    
    # Also show baseline for comparison
    print(f"[BASELINE] No dither")
    print("-" * 40)
    baseline = generate(prompt, n_tokens, 0.0, seed)
    print(baseline)
    print()
    
    return {
        "prompt": prompt,
        "divergent": divergent,
        "convergent": convergent,
        "baseline": baseline,
    }


def backward_pass_v2(prompt: str, n_tokens: int = 64, seed: int = 42) -> dict:
    """
    Alternative: Generate both, then synthesize.
    
    1. Generate with DIVERGE
    2. Generate with CONVERGE  
    3. Ask model to synthesize both perspectives
    """
    
    print("=" * 60)
    print("BACKWARD PASS v2: SYNTHESIS")
    print("=" * 60)
    print(f"Prompt: {prompt}")
    print()
    
    # Generate both perspectives
    print(f"[DIVERGE] intensity={INTENSITY_DIVERGE}")
    print("-" * 40)
    divergent = generate(prompt, n_tokens, INTENSITY_DIVERGE, seed)
    print(divergent[:300])
    print()
    
    print(f"[CONVERGE] intensity={INTENSITY_CONVERGE}")
    print("-" * 40)
    convergent = generate(prompt, n_tokens, INTENSITY_CONVERGE, seed)
    print(convergent[:300])
    print()
    
    # Synthesize
    synth_prompt = f"""Two perspectives on "{prompt}"

Perspective A (exploratory):
{divergent[:300]}

Perspective B (structured):
{convergent[:300]}

Synthesizing both perspectives into a unified answer:"""
    
    print("[SYNTHESIS] Combining both...")
    print("-" * 40)
    synthesis = generate(synth_prompt, n_tokens, 0.13, seed)  # Middle ground
    print(synthesis)
    print()
    
    return {
        "divergent": divergent,
        "convergent": convergent,
        "synthesis": synthesis,
    }


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The meaning of life is"
    
    print("\n" + "=" * 60)
    print("VERSION 1: Draft → Refine")
    print("=" * 60 + "\n")
    backward_pass(prompt)
    
    print("\n" + "=" * 60)
    print("VERSION 2: Dual Generate → Synthesize")  
    print("=" * 60 + "\n")
    backward_pass_v2(prompt)

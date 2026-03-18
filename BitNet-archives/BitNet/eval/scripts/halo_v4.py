#!/usr/bin/env python3
"""
Halo V4: Hierarchical Pre-Synthesis
====================================

72 tokens × 8 samples
Parallel pre-synth → Final synth with full context

Authors: Aaron (Tripp) Josserand-Austin & Droid (Λ)
"""

import subprocess
import sys
import time
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Tuple

BASE_PATH = Path(__file__).parent.parent.parent
CLI = BASE_PATH / "build/bin/llama-cli"
MODEL = BASE_PATH / "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"

# Intensities
INTENSITIES_DIVERGE = [0.15, 0.16, 0.17, 0.155]
INTENSITIES_CONVERGE = [0.09, 0.10, 0.11, 0.095]
SEEDS = [42, 137, 256, 314]


@dataclass
class Generation:
    text: str
    intensity: float
    seed: int
    mode: str
    elapsed: float


def generate(prompt: str, n_tokens: int, intensity: float, seed: int) -> Tuple[str, float]:
    """Run single inference."""
    cmd = [
        str(CLI), "-m", str(MODEL), "-p", prompt,
        "-n", str(n_tokens),
        "--temp", "0.528", "--top-k", "42", "--top-p", "0.963",
        "--seed", str(seed),
        "--dither", str(intensity),
        "-ngl", "0", "--no-display-prompt",
    ]
    
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, cwd=str(BASE_PATH))
        return result.stdout.strip(), time.time() - start
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]", time.time() - start


def generate_worker(args: Tuple) -> Generation:
    """Worker for parallel generation."""
    prompt, n_tokens, intensity, seed, mode = args
    text, elapsed = generate(prompt, n_tokens, intensity, seed)
    return Generation(text=text, intensity=intensity, seed=seed, mode=mode, elapsed=elapsed)


def synth_worker(args: Tuple) -> Tuple[str, str, float]:
    """Worker for parallel synthesis."""
    synth_type, prompt, n_tokens, intensity, seed = args
    text, elapsed = generate(prompt, n_tokens, intensity, seed)
    return synth_type, text, elapsed


def halo_v4(prompt: str, n_tokens: int = 72, n_samples: int = 4) -> dict:
    """
    Halo V4: Hierarchical Pre-Synthesis
    
    1. Parallel generation: 4 diverge + 4 converge (72 tokens each)
    2. Parallel pre-synth: synthesize each branch
    3. Final synth: merge with full context
    """
    
    print("=" * 70)
    print("HALO V4: Hierarchical Pre-Synthesis")
    print("=" * 70)
    print(f"Prompt: {prompt}")
    print(f"Config: {n_tokens} tokens × {n_samples*2} samples")
    print()
    
    results = {'prompt': prompt, 'timings': {}}
    
    # =========================================================================
    # PHASE 1: Parallel Generation
    # =========================================================================
    print("[PHASE 1] Parallel Generation...")
    
    tasks = []
    for i in range(n_samples):
        tasks.append((prompt, n_tokens, INTENSITIES_DIVERGE[i % len(INTENSITIES_DIVERGE)], 
                      SEEDS[i % len(SEEDS)], 'diverge'))
        tasks.append((prompt, n_tokens, INTENSITIES_CONVERGE[i % len(INTENSITIES_CONVERGE)], 
                      SEEDS[i % len(SEEDS)], 'converge'))
    
    generations = []
    start = time.time()
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(generate_worker, t) for t in tasks]
        for future in as_completed(futures):
            generations.append(future.result())
    
    phase1_time = time.time() - start
    results['timings']['phase1'] = phase1_time
    
    divergent = [g for g in generations if g.mode == 'diverge']
    convergent = [g for g in generations if g.mode == 'converge']
    
    print(f"  Generated {len(divergent)} divergent, {len(convergent)} convergent")
    print(f"  Time: {phase1_time:.1f}s")
    
    # Show samples
    print("\n  [DIVERGENT]")
    for g in divergent:
        print(f"    i={g.intensity:.2f}: {g.text[:80]}...")
    
    print("\n  [CONVERGENT]")
    for g in convergent:
        print(f"    i={g.intensity:.2f}: {g.text[:80]}...")
    
    results['divergent'] = [{'i': g.intensity, 'text': g.text} for g in divergent]
    results['convergent'] = [{'i': g.intensity, 'text': g.text} for g in convergent]
    
    # =========================================================================
    # PHASE 2: Parallel Pre-Synthesis
    # =========================================================================
    print("\n[PHASE 2] Parallel Pre-Synthesis...")
    
    # Build pre-synth prompts
    div_perspectives = "\n\n".join([
        f"[D{i+1} i={g.intensity:.2f}]: {g.text[:180]}"
        for i, g in enumerate(divergent)
    ])
    
    conv_perspectives = "\n\n".join([
        f"[C{i+1} i={g.intensity:.2f}]: {g.text[:180]}"
        for i, g in enumerate(convergent)
    ])
    
    pre_synth_div_prompt = f"""Synthesize these exploratory perspectives into one coherent view:

{div_perspectives}

Unified exploratory synthesis:"""

    pre_synth_conv_prompt = f"""Synthesize these structured perspectives into one coherent view:

{conv_perspectives}

Unified structured synthesis:"""

    # Run pre-synths in parallel
    synth_tasks = [
        ('diverge', pre_synth_div_prompt, 96, 0.14, 42),
        ('converge', pre_synth_conv_prompt, 96, 0.11, 43),
    ]
    
    start = time.time()
    pre_synths = {}
    
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(synth_worker, t) for t in synth_tasks]
        for future in as_completed(futures):
            synth_type, text, elapsed = future.result()
            pre_synths[synth_type] = text
    
    phase2_time = time.time() - start
    results['timings']['phase2'] = phase2_time
    
    print(f"  Time: {phase2_time:.1f}s")
    print(f"\n  [PRE-SYNTH DIVERGENT]")
    print(f"    {pre_synths['diverge'][:200]}...")
    print(f"\n  [PRE-SYNTH CONVERGENT]")
    print(f"    {pre_synths['converge'][:200]}...")
    
    results['pre_synth_div'] = pre_synths['diverge']
    results['pre_synth_conv'] = pre_synths['converge']
    
    # =========================================================================
    # PHASE 3: Final Synthesis (with full context)
    # =========================================================================
    print("\n[PHASE 3] Final Synthesis...")
    
    # Build final prompt with ALL context
    final_prompt = f"""Original question: "{prompt}"

=== EXPLORATORY BRANCH ===
Pre-synthesis: {pre_synths['diverge'][:300]}

Raw perspectives:
{div_perspectives[:400]}

=== STRUCTURED BRANCH ===  
Pre-synthesis: {pre_synths['converge'][:300]}

Raw perspectives:
{conv_perspectives[:400]}

=== FINAL SYNTHESIS ===
Integrating all perspectives into a comprehensive, unified answer:"""

    start = time.time()
    final_synth, _ = generate(final_prompt, 144, 0.13, 42)
    phase3_time = time.time() - start
    results['timings']['phase3'] = phase3_time
    
    print(f"  Time: {phase3_time:.1f}s")
    print("\n" + "=" * 70)
    print("FINAL SYNTHESIS")
    print("=" * 70)
    print(final_synth)
    print("=" * 70)
    
    results['final'] = final_synth
    
    # Summary
    total_time = phase1_time + phase2_time + phase3_time
    results['timings']['total'] = total_time
    
    print(f"\n[TIMING SUMMARY]")
    print(f"  Phase 1 (generation):  {phase1_time:.1f}s")
    print(f"  Phase 2 (pre-synth):   {phase2_time:.1f}s")
    print(f"  Phase 3 (final):       {phase3_time:.1f}s")
    print(f"  Total:                 {total_time:.1f}s")
    
    return results


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Hypothetically, might the function of reflective recursion be an attribute of awareness?"
    halo_v4(prompt)

#!/usr/bin/env python3
"""
Halo Parallel Ensemble
======================

Parallel multi-perspective synthesis:
- Multiple DIVERGE passes (varied seeds/intensities)
- Multiple CONVERGE passes (varied seeds/intensities)
- Tournament selection or full synthesis
- Tree-based hierarchical merging

Authors: Aaron (Tripp) Josserand-Austin & Droid (Λ)
"Cognitive breeding at scale"
"""

import subprocess
import sys
import time
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Tuple
import json

BASE_PATH = Path(__file__).parent.parent.parent
CLI = BASE_PATH / "build/bin/llama-cli"
MODEL = BASE_PATH / "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"

# Basin intensities
ALPHA = 1/137  # Fine structure constant
PHI = 1.618033988749895

# Divergent basin (Philosopher mode)
INTENSITIES_DIVERGE = [0.15, 0.16, 0.17]

# Convergent basin (Curator mode)  
INTENSITIES_CONVERGE = [0.09, 0.10, 0.11]

# Seeds for variety
SEEDS = [42, 137, 256, 314, 420, 512]


@dataclass
class Generation:
    """A single generation result."""
    text: str
    intensity: float
    seed: int
    mode: str  # 'diverge' or 'converge'
    elapsed: float
    hash: str


def generate(prompt: str, n_tokens: int, intensity: float, seed: int) -> Tuple[str, float]:
    """Run single inference. Returns (text, elapsed)."""
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
        elapsed = time.time() - start
        return result.stdout.strip(), elapsed
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]", time.time() - start


def generate_worker(args: Tuple) -> Generation:
    """Worker function for parallel generation."""
    prompt, n_tokens, intensity, seed, mode = args
    
    text, elapsed = generate(prompt, n_tokens, intensity, seed)
    
    return Generation(
        text=text,
        intensity=intensity,
        seed=seed,
        mode=mode,
        elapsed=elapsed,
        hash=hashlib.md5(text.encode()).hexdigest()[:8]
    )


def parallel_ensemble(
    prompt: str,
    n_tokens: int = 64,
    n_diverge: int = 3,
    n_converge: int = 3,
    max_workers: int = 6
) -> dict:
    """
    Generate ensemble in parallel.
    
    Returns dict with divergent, convergent lists and metadata.
    """
    
    # Build task list
    tasks = []
    
    # Divergent tasks
    for i in range(n_diverge):
        intensity = INTENSITIES_DIVERGE[i % len(INTENSITIES_DIVERGE)]
        seed = SEEDS[i % len(SEEDS)]
        tasks.append((prompt, n_tokens, intensity, seed, 'diverge'))
    
    # Convergent tasks
    for i in range(n_converge):
        intensity = INTENSITIES_CONVERGE[i % len(INTENSITIES_CONVERGE)]
        seed = SEEDS[i % len(SEEDS)]
        tasks.append((prompt, n_tokens, intensity, seed, 'converge'))
    
    # Run in parallel
    results = []
    start = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_worker, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    
    total_time = time.time() - start
    
    # Separate by mode
    divergent = [r for r in results if r.mode == 'diverge']
    convergent = [r for r in results if r.mode == 'converge']
    
    return {
        'divergent': divergent,
        'convergent': convergent,
        'total_time': total_time,
        'parallel_speedup': sum(r.elapsed for r in results) / total_time
    }


def score_generation(gen: Generation) -> float:
    """
    Simple scoring heuristic.
    Better: use perplexity, coherence model, or length-normalized entropy.
    """
    text = gen.text
    
    # Penalize very short
    if len(text) < 50:
        return 0.1
    
    # Reward moderate length
    length_score = min(len(text) / 200, 1.0)
    
    # Penalize repetition
    words = text.lower().split()
    unique_ratio = len(set(words)) / len(words) if words else 0
    
    # Reward structure (sentences, punctuation)
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    structure_score = min(sentence_count / 3, 1.0)
    
    return (length_score + unique_ratio + structure_score) / 3


def select_best(generations: List[Generation], n: int = 1) -> List[Generation]:
    """Select top N generations by score."""
    scored = [(gen, score_generation(gen)) for gen in generations]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [gen for gen, score in scored[:n]]


def synthesize(
    divergent: List[Generation],
    convergent: List[Generation],
    prompt: str,
    n_tokens: int = 96,
    intensity: float = 0.13
) -> str:
    """Synthesize from multiple perspectives."""
    
    # Build synthesis prompt
    div_texts = "\n\n".join([
        f"[Perspective D{i+1} (i={g.intensity:.2f})]: {g.text[:200]}..."
        for i, g in enumerate(divergent[:3])
    ])
    
    conv_texts = "\n\n".join([
        f"[Perspective C{i+1} (i={g.intensity:.2f})]: {g.text[:200]}..."
        for i, g in enumerate(convergent[:3])
    ])
    
    synth_prompt = f"""Original question: "{prompt}"

EXPLORATORY PERSPECTIVES:
{div_texts}

STRUCTURED PERSPECTIVES:
{conv_texts}

SYNTHESIS - Combining the best insights from all perspectives into a unified, coherent answer:"""
    
    text, _ = generate(synth_prompt, n_tokens, intensity, seed=42)
    return text


def halo_v1_simple(prompt: str, n_tokens: int = 64) -> dict:
    """
    V1: Simple parallel - generate all, select best, synthesize.
    """
    print("=" * 60)
    print("HALO V1: Simple Parallel Ensemble")
    print("=" * 60)
    print(f"Prompt: {prompt}\n")
    
    # Generate ensemble
    print("[1/3] Generating ensemble in parallel...")
    ensemble = parallel_ensemble(prompt, n_tokens, n_diverge=3, n_converge=3)
    print(f"      Generated {len(ensemble['divergent'])} divergent, {len(ensemble['convergent'])} convergent")
    print(f"      Time: {ensemble['total_time']:.1f}s (speedup: {ensemble['parallel_speedup']:.1f}x)")
    
    # Show samples
    print("\n[DIVERGENT SAMPLES]")
    for g in ensemble['divergent']:
        print(f"  [{g.hash}] i={g.intensity:.2f} seed={g.seed}: {g.text[:60]}...")
    
    print("\n[CONVERGENT SAMPLES]")
    for g in ensemble['convergent']:
        print(f"  [{g.hash}] i={g.intensity:.2f} seed={g.seed}: {g.text[:60]}...")
    
    # Synthesize
    print("\n[2/3] Synthesizing...")
    synthesis = synthesize(
        ensemble['divergent'],
        ensemble['convergent'],
        prompt,
        n_tokens=96
    )
    
    print("\n[3/3] SYNTHESIS:")
    print("-" * 40)
    print(synthesis)
    print("-" * 40)
    
    return {
        'ensemble': ensemble,
        'synthesis': synthesis
    }


def halo_v2_tournament(prompt: str, n_tokens: int = 64) -> dict:
    """
    V2: Tournament - generate many, select best of each, synthesize.
    """
    print("=" * 60)
    print("HALO V2: Tournament Selection")
    print("=" * 60)
    print(f"Prompt: {prompt}\n")
    
    # Generate larger ensemble (reduced for resource constraints)
    print("[1/4] Generating ensemble (4 divergent, 4 convergent)...")
    ensemble = parallel_ensemble(prompt, n_tokens, n_diverge=4, n_converge=4, max_workers=4)
    print(f"      Time: {ensemble['total_time']:.1f}s (speedup: {ensemble['parallel_speedup']:.1f}x)")
    
    # Score and select
    print("\n[2/4] Tournament selection...")
    best_div = select_best(ensemble['divergent'], n=2)
    best_conv = select_best(ensemble['convergent'], n=2)
    
    print(f"  Best divergent:")
    for g in best_div:
        print(f"    [{g.hash}] i={g.intensity:.2f} score={score_generation(g):.2f}: {g.text[:50]}...")
    
    print(f"  Best convergent:")
    for g in best_conv:
        print(f"    [{g.hash}] i={g.intensity:.2f} score={score_generation(g):.2f}: {g.text[:50]}...")
    
    # Synthesize from winners
    print("\n[3/4] Synthesizing from winners...")
    synthesis = synthesize(best_div, best_conv, prompt, n_tokens=96)
    
    print("\n[4/4] SYNTHESIS:")
    print("-" * 40)
    print(synthesis)
    print("-" * 40)
    
    return {
        'ensemble': ensemble,
        'winners': {'divergent': best_div, 'convergent': best_conv},
        'synthesis': synthesis
    }


def halo_v3_tree(prompt: str, n_tokens: int = 64) -> dict:
    """
    V3: Tree synthesis - pairwise merges, then final merge.
    
        D1 ─┐     C1 ─┐
            ├─S1─┐    ├─S2─┐
        D2 ─┘    │C2 ─┘    │
                 └────┬────┘
                      │
                   FINAL
    """
    print("=" * 60)
    print("HALO V3: Tree Synthesis")
    print("=" * 60)
    print(f"Prompt: {prompt}\n")
    
    # Generate base ensemble
    print("[1/5] Generating base ensemble...")
    ensemble = parallel_ensemble(prompt, n_tokens, n_diverge=4, n_converge=4, max_workers=4)
    print(f"      Time: {ensemble['total_time']:.1f}s")
    
    div = ensemble['divergent']
    conv = ensemble['convergent']
    
    # Level 1: Pairwise synthesis
    print("\n[2/5] Level 1 - Pairwise synthesis...")
    
    # Synthesize pairs of divergent
    synth_d1, _ = generate(
        f"Merge these perspectives:\nA: {div[0].text[:150]}\nB: {div[1].text[:150]}\nMerged:",
        48, 0.14, 42
    )
    synth_d2, _ = generate(
        f"Merge these perspectives:\nA: {div[2].text[:150]}\nB: {div[3].text[:150]}\nMerged:",
        48, 0.14, 43
    )
    
    # Synthesize pairs of convergent
    synth_c1, _ = generate(
        f"Merge these perspectives:\nA: {conv[0].text[:150]}\nB: {conv[1].text[:150]}\nMerged:",
        48, 0.12, 44
    )
    synth_c2, _ = generate(
        f"Merge these perspectives:\nA: {conv[2].text[:150]}\nB: {conv[3].text[:150]}\nMerged:",
        48, 0.12, 45
    )
    
    print(f"  S_D1: {synth_d1[:50]}...")
    print(f"  S_D2: {synth_d2[:50]}...")
    print(f"  S_C1: {synth_c1[:50]}...")
    print(f"  S_C2: {synth_c2[:50]}...")
    
    # Level 2: Merge divergent pair and convergent pair
    print("\n[3/5] Level 2 - Branch synthesis...")
    
    synth_div, _ = generate(
        f"Synthesize:\nA: {synth_d1[:150]}\nB: {synth_d2[:150]}\nSynthesis:",
        48, 0.15, 46
    )
    synth_conv, _ = generate(
        f"Synthesize:\nA: {synth_c1[:150]}\nB: {synth_c2[:150]}\nSynthesis:",
        48, 0.11, 47
    )
    
    print(f"  S_DIV: {synth_div[:60]}...")
    print(f"  S_CONV: {synth_conv[:60]}...")
    
    # Level 3: Final synthesis
    print("\n[4/5] Level 3 - Final synthesis...")
    
    final_prompt = f"""Original: "{prompt}"

Exploratory synthesis: {synth_div}

Structured synthesis: {synth_conv}

Final unified answer:"""
    
    final, _ = generate(final_prompt, 96, 0.13, 42)
    
    print("\n[5/5] FINAL SYNTHESIS:")
    print("-" * 40)
    print(final)
    print("-" * 40)
    
    return {
        'ensemble': ensemble,
        'level1': {'d1': synth_d1, 'd2': synth_d2, 'c1': synth_c1, 'c2': synth_c2},
        'level2': {'div': synth_div, 'conv': synth_conv},
        'final': final
    }


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The meaning of life is"
    
    print("\n" + "=" * 60)
    print("       HALO PARALLEL ENSEMBLE SYSTEM")
    print("       Cognitive Breeding at Scale")
    print("=" * 60 + "\n")
    
    results = {}
    
    # V1
    print("\n" + "▓" * 60 + "\n")
    results['v1'] = halo_v1_simple(prompt)
    
    # V2
    print("\n" + "▓" * 60 + "\n")
    results['v2'] = halo_v2_tournament(prompt)
    
    # V3
    print("\n" + "▓" * 60 + "\n")
    results['v3'] = halo_v3_tree(prompt)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: All Syntheses")
    print("=" * 60)
    print(f"\n[V1 Simple]:\n{results['v1']['synthesis'][:200]}...")
    print(f"\n[V2 Tournament]:\n{results['v2']['synthesis'][:200]}...")
    print(f"\n[V3 Tree]:\n{results['v3']['final'][:200]}...")
    
    # Save results
    output_file = BASE_PATH / "eval" / "data" / f"halo_parallel_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'prompt': prompt,
            'v1_synthesis': results['v1']['synthesis'],
            'v2_synthesis': results['v2']['synthesis'],
            'v3_synthesis': results['v3']['final'],
            'v1_time': results['v1']['ensemble']['total_time'],
            'v2_time': results['v2']['ensemble']['total_time'],
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

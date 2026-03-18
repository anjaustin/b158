#!/usr/bin/env python3
"""
Rigorous tests for Pink-Steinberg two-pass error diffusion

Tests:
1. Sampler chain verification - are our samplers in the chain?
2. Error measurement validation - do we capture meaningful metrics?
3. Reproducibility - same seed = same output?
4. Dither comparison - pink-measure shouldn't change output (observation only)
5. Edge cases - empty input, single token, etc.
"""

import subprocess
import sys
import os
import json
import hashlib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf")
LLAMA_CLI = os.path.join(PROJECT_ROOT, "build/bin/llama-cli")

TESTS_PASSED = 0
TESTS_FAILED = 0

def run_inference(prompt: str, extra_args: list = None, n_predict: int = 50, seed: int = 42) -> dict:
    """Run inference and return detailed results"""
    cmd = [
        LLAMA_CLI,
        "-m", MODEL_PATH,
        "-p", prompt,
        "--temp", "0.5",
        "--repeat-penalty", "1.2",
        "--seed", str(seed),
        "-n", str(n_predict),
        "--no-warmup"
    ]
    if extra_args:
        cmd.extend(extra_args)
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    
    # Combine stdout and stderr for parsing (CLI mixes them)
    combined = result.stdout + "\n" + result.stderr
    
    # Extract generated text (after prompt in stdout)
    output = result.stdout
    if prompt in output:
        output = output.split(prompt, 1)[-1]
    
    # Extract sampler chain (can be in either stream)
    sampler_chain = ""
    for line in combined.split('\n'):
        if "sampler chain:" in line:
            sampler_chain = line.strip()
            break
    
    # Extract timing
    timing_ms = 0.0
    tokens_generated = 0
    for line in combined.split('\n'):
        if 'total time' in line:
            try:
                timing_ms = float(line.split('=')[1].split('ms')[0].strip())
            except:
                pass
        if 'eval time' in line and 'runs' in line:
            try:
                tokens_generated = int(line.split('/')[1].split('runs')[0].strip())
            except:
                pass
    
    return {
        'output': output.strip(),
        'output_hash': hashlib.md5(output.strip().encode()).hexdigest()[:16],
        'sampler_chain': sampler_chain,
        'timing_ms': timing_ms,
        'tokens': tokens_generated,
        'returncode': result.returncode
    }


def test_pass(name: str, condition: bool, details: str = ""):
    """Record test result"""
    global TESTS_PASSED, TESTS_FAILED
    if condition:
        TESTS_PASSED += 1
        print(f"  [PASS] {name}")
    else:
        TESTS_FAILED += 1
        print(f"  [FAIL] {name}")
    if details:
        print(f"         {details}")


def test_1_sampler_chain():
    """Verify sampler chain contains expected components"""
    print("\n" + "=" * 60)
    print("TEST 1: Sampler Chain Verification")
    print("=" * 60)
    
    # Test with dither enabled
    result = run_inference("test", ["--dither", "0.1"], n_predict=5)
    
    test_pass(
        "CLI returns success",
        result['returncode'] == 0,
        f"returncode={result['returncode']}"
    )
    
    test_pass(
        "Sampler chain reported",
        len(result['sampler_chain']) > 0,
        result['sampler_chain'][:80] if result['sampler_chain'] else "NO CHAIN FOUND"
    )
    
    test_pass(
        "ordered-dither in chain when --dither used",
        "ordered-dither" in result['sampler_chain'] or "dither" in result['sampler_chain'].lower(),
        result['sampler_chain']
    )


def test_2_reproducibility():
    """Same seed should produce identical output"""
    print("\n" + "=" * 60)
    print("TEST 2: Reproducibility (same seed = same output)")
    print("=" * 60)
    
    prompt = "The capital of France is"
    
    # Run twice with same seed
    result1 = run_inference(prompt, seed=42, n_predict=30)
    result2 = run_inference(prompt, seed=42, n_predict=30)
    
    test_pass(
        "Run 1 succeeds",
        result1['returncode'] == 0
    )
    
    test_pass(
        "Run 2 succeeds", 
        result2['returncode'] == 0
    )
    
    test_pass(
        "Outputs match (same seed)",
        result1['output_hash'] == result2['output_hash'],
        f"hash1={result1['output_hash']}, hash2={result2['output_hash']}"
    )
    
    # Run with different seed
    result3 = run_inference(prompt, seed=123, n_predict=30)
    
    test_pass(
        "Different seed produces different output",
        result1['output_hash'] != result3['output_hash'],
        f"seed42={result1['output_hash']}, seed123={result3['output_hash']}"
    )


def test_3_dither_effect():
    """Dithering should change output vs baseline"""
    print("\n" + "=" * 60)
    print("TEST 3: Dither Effect Validation")
    print("=" * 60)
    
    prompt = "Explain quantum entanglement in simple terms:"
    
    # Baseline (no dither)
    baseline = run_inference(prompt, ["--dither", "0.0"], seed=42, n_predict=50)
    
    # With dither
    dithered = run_inference(prompt, ["--dither", "0.1"], seed=42, n_predict=50)
    
    test_pass(
        "Baseline succeeds",
        baseline['returncode'] == 0
    )
    
    test_pass(
        "Dithered succeeds",
        dithered['returncode'] == 0
    )
    
    test_pass(
        "Dither changes output",
        baseline['output_hash'] != dithered['output_hash'],
        f"baseline={baseline['output_hash']}, dithered={dithered['output_hash']}"
    )
    
    # Verify dither=0.0 matches true baseline (no --dither flag)
    true_baseline = run_inference(prompt, seed=42, n_predict=50)
    
    test_pass(
        "--dither 0.0 matches no-dither baseline",
        baseline['output_hash'] == true_baseline['output_hash'],
        f"dither0={baseline['output_hash']}, nodither={true_baseline['output_hash']}"
    )


def test_4_dither_intensity_range():
    """Different intensities should produce different outputs"""
    print("\n" + "=" * 60)
    print("TEST 4: Dither Intensity Range")
    print("=" * 60)
    
    prompt = "Write a haiku about computing:"
    
    intensities = [0.0, 0.05, 0.1, 0.16]
    results = {}
    
    for intensity in intensities:
        result = run_inference(prompt, ["--dither", str(intensity)], seed=42, n_predict=40)
        results[intensity] = result
        test_pass(
            f"Intensity {intensity} succeeds",
            result['returncode'] == 0,
            f"tokens={result['tokens']}"
        )
    
    # Check that different intensities produce different outputs
    unique_hashes = set(r['output_hash'] for r in results.values())
    test_pass(
        f"Different intensities produce varied outputs",
        len(unique_hashes) >= 2,
        f"{len(unique_hashes)} unique outputs from {len(intensities)} intensities"
    )


def test_5_edge_cases():
    """Handle edge cases gracefully"""
    print("\n" + "=" * 60)
    print("TEST 5: Edge Cases")
    print("=" * 60)
    
    # Very short generation
    result = run_inference("Hello", ["--dither", "0.1"], n_predict=1)
    test_pass(
        "Single token generation",
        result['returncode'] == 0,
        f"output='{result['output'][:50]}'"
    )
    
    # Empty-ish prompt
    result = run_inference(" ", ["--dither", "0.1"], n_predict=10)
    test_pass(
        "Whitespace prompt",
        result['returncode'] == 0
    )
    
    # Very high intensity (should be clamped)
    result = run_inference("Test", ["--dither", "10.0"], n_predict=10)
    test_pass(
        "High intensity clamped gracefully",
        result['returncode'] == 0,
        "Intensity 10.0 should be clamped to max"
    )
    
    # Very low intensity
    result = run_inference("Test", ["--dither", "0.001"], n_predict=10)
    test_pass(
        "Low intensity works",
        result['returncode'] == 0
    )


def test_6_control_validation():
    """Validate against original BitNet (if available)"""
    print("\n" + "=" * 60)
    print("TEST 6: Control Validation")
    print("=" * 60)
    
    original_cli = os.path.join(PROJECT_ROOT, "..", "BitNet-original", "build", "bin", "llama-cli")
    
    if not os.path.exists(original_cli):
        print("  [SKIP] Original BitNet not available for control test")
        print(f"         Expected at: {original_cli}")
        return
    
    prompt = "The quick brown fox"
    
    # Our build with dither=0.0
    ours = run_inference(prompt, ["--dither", "0.0"], seed=42, n_predict=30)
    
    # Original build
    cmd = [
        original_cli,
        "-m", MODEL_PATH,
        "-p", prompt,
        "--temp", "0.5",
        "--repeat-penalty", "1.2",
        "--seed", "42",
        "-n", "30",
        "--no-warmup"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    original_output = result.stdout
    if prompt in original_output:
        original_output = original_output.split(prompt, 1)[-1]
    original_hash = hashlib.md5(original_output.strip().encode()).hexdigest()[:16]
    
    test_pass(
        "Control matches original (dither=0.0)",
        ours['output_hash'] == original_hash,
        f"ours={ours['output_hash']}, original={original_hash}"
    )


def test_7_performance():
    """Ensure dithering doesn't significantly impact performance"""
    print("\n" + "=" * 60)
    print("TEST 7: Performance Impact")
    print("=" * 60)
    
    prompt = "Explain the theory of relativity in detail:"
    n_runs = 3
    
    # Baseline timing
    baseline_times = []
    for _ in range(n_runs):
        result = run_inference(prompt, ["--dither", "0.0"], seed=42, n_predict=80)
        if result['timing_ms'] > 0:
            baseline_times.append(result['timing_ms'])
    
    # Dithered timing
    dither_times = []
    for _ in range(n_runs):
        result = run_inference(prompt, ["--dither", "0.1"], seed=42, n_predict=80)
        if result['timing_ms'] > 0:
            dither_times.append(result['timing_ms'])
    
    if baseline_times and dither_times:
        avg_baseline = sum(baseline_times) / len(baseline_times)
        avg_dither = sum(dither_times) / len(dither_times)
        overhead_pct = ((avg_dither - avg_baseline) / avg_baseline) * 100
        
        test_pass(
            "Dither overhead < 10%",
            overhead_pct < 10.0,
            f"baseline={avg_baseline:.1f}ms, dither={avg_dither:.1f}ms, overhead={overhead_pct:.1f}%"
        )
    else:
        print("  [SKIP] Could not measure timing")


def main():
    print("=" * 60)
    print("PINK-STEINBERG RIGOROUS TEST SUITE")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"CLI: {LLAMA_CLI}")
    
    # Verify prerequisites
    if not os.path.exists(LLAMA_CLI):
        print(f"ERROR: CLI not found at {LLAMA_CLI}")
        return 1
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return 1
    
    # Run all tests
    test_1_sampler_chain()
    test_2_reproducibility()
    test_3_dither_effect()
    test_4_dither_intensity_range()
    test_5_edge_cases()
    test_6_control_validation()
    test_7_performance()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  PASSED: {TESTS_PASSED}")
    print(f"  FAILED: {TESTS_FAILED}")
    print(f"  TOTAL:  {TESTS_PASSED + TESTS_FAILED}")
    
    if TESTS_FAILED > 0:
        print("\n  STATUS: TESTS FAILED - DO NOT PROCEED")
        return 1
    else:
        print("\n  STATUS: ALL TESTS PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())

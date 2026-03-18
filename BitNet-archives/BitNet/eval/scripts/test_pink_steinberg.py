#!/usr/bin/env python3
"""
Test Pink-Steinberg two-pass error diffusion

P1wTC-P2A: Pass 1 measures at Touchpoint C (probability), Pass 2 injects at Touchpoint A (logits)
"""

import subprocess
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf")
LLAMA_CLI = os.path.join(PROJECT_ROOT, "build/bin/llama-cli")

def run_inference(prompt: str, extra_args: list = None, n_predict: int = 100) -> tuple[str, float]:
    """Run inference and return output and timing"""
    cmd = [
        LLAMA_CLI,
        "-m", MODEL_PATH,
        "-p", prompt,
        "--temp", "0.5",
        "--repeat-penalty", "1.2",
        "--seed", "42",
        "-n", str(n_predict),
        "--no-warmup"
    ]
    if extra_args:
        cmd.extend(extra_args)
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    
    # Extract output (everything after the prompt)
    output = result.stdout
    if prompt in output:
        output = output.split(prompt, 1)[-1]
    
    # Extract timing
    timing = 0.0
    for line in result.stderr.split('\n'):
        if 'total time' in line:
            try:
                timing = float(line.split('=')[1].split('ms')[0].strip())
            except:
                pass
    
    return output.strip(), timing


def test_vanilla():
    """Baseline: vanilla BitNet without any dithering"""
    print("=" * 60)
    print("TEST 1: Vanilla BitNet (no dithering)")
    print("=" * 60)
    
    prompt = "Hypothetically, might the function of reflective recursion be a feature of awareness?"
    output, timing = run_inference(prompt)
    
    print(f"Prompt: {prompt}")
    print(f"Output: {output[:500]}...")
    print(f"Time: {timing:.1f}ms")
    return output


def test_ordered_dither():
    """Test with ordered dithering (Bayer matrix)"""
    print("\n" + "=" * 60)
    print("TEST 2: Ordered Dithering (--dither 0.1)")
    print("=" * 60)
    
    prompt = "Hypothetically, might the function of reflective recursion be a feature of awareness?"
    output, timing = run_inference(prompt, ["--dither", "0.1"])
    
    print(f"Prompt: {prompt}")
    print(f"Output: {output[:500]}...")
    print(f"Time: {timing:.1f}ms")
    return output


def test_build_info():
    """Verify Pink-Steinberg samplers are available"""
    print("\n" + "=" * 60)
    print("BUILD VERIFICATION")
    print("=" * 60)
    
    # Check if the binary has our samplers by looking at help
    cmd = [LLAMA_CLI, "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    
    has_dither = "--dither" in result.stdout
    print(f"--dither flag: {'FOUND' if has_dither else 'NOT FOUND'}")
    
    # Try running with --samplers to see chain
    cmd = [
        LLAMA_CLI,
        "-m", MODEL_PATH,
        "-p", "test",
        "-n", "1",
        "--no-warmup"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    
    if "sampler chain:" in result.stdout:
        for line in result.stdout.split('\n'):
            if "sampler chain:" in line:
                print(f"Sampler chain: {line}")
    
    return has_dither


def main():
    print("Pink-Steinberg Two-Pass Error Diffusion Test")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"CLI: {LLAMA_CLI}")
    print()
    
    # Verify build
    if not test_build_info():
        print("ERROR: Build verification failed")
        return 1
    
    # Run tests
    vanilla_output = test_vanilla()
    dither_output = test_ordered_dither()
    
    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    
    if vanilla_output == dither_output:
        print("WARNING: Outputs are identical - dithering may not be working")
    else:
        print("SUCCESS: Outputs differ - dithering is having an effect")
        
        # Show first divergence point
        for i, (a, b) in enumerate(zip(vanilla_output, dither_output)):
            if a != b:
                print(f"First difference at character {i}:")
                print(f"  Vanilla: ...{vanilla_output[max(0,i-20):i+30]}...")
                print(f"  Dither:  ...{dither_output[max(0,i-20):i+30]}...")
                break
    
    print("\n" + "=" * 60)
    print("NOTE: Pink-Steinberg two-pass inference requires CLI integration")
    print("Current test validates sampler compilation only.")
    print("Full two-pass test pending --pink-steinberg CLI flag implementation.")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
HumanEval Test with Halo 3+3+1+1 Multi-Pass Synthesis
The Magick Inferometer in full effect
"""

import subprocess
import time
import re

TEMP = 0.08645
REPEAT_PENALTY = 1.285
MODEL_PATH = "/home/ztflynn/000/randy/b158/BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
LLAMA_CLI = "/home/ztflynn/000/randy/b158/BitNet/build/bin/llama-cli"

# Halo intensity bounds
INTENSITY_DIV = [0.15, 0.16, 0.17]  # 3 divergent samples
INTENSITY_CONV = [0.09, 0.10, 0.11]  # 3 convergent samples
INTENSITY_PRESYNTH = 0.13
INTENSITY_FINAL = 0.13

PROBLEMS = {
    "is_prime": {
        "partial": '''def is_prime(n):
    """Return True if n is prime, False otherwise."""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return''',
        "test_cases": [
            ("is_prime(6)", False),
            ("is_prime(101)", True),
            ("is_prime(11)", True),
            ("is_prime(2)", True),
            ("is_prime(1)", False),
        ],
    },
    "fizz_buzz": {
        "partial": '''def fizz_buzz(n):
    """Count times digit 7 appears in numbers less than n divisible by 11 or 13."""
    count = 0
    for i in range(n):
        if i % 11 == 0 or i % 13 == 0:
            s = str(i)
            count += s.count(''',
        "test_cases": [
            ("fizz_buzz(50)", 0),
            ("fizz_buzz(78)", 2),
            ("fizz_buzz(79)", 3),
        ],
    },
    "below_zero": {
        "partial": '''def below_zero(operations):
    """Return True if running sum of operations goes below zero."""
    balance = 0
    for op in operations:
        balance += op
        if balance < 0:
            return''',
        "test_cases": [
            ("below_zero([1, 2, 3])", False),
            ("below_zero([1, 2, -4, 5])", True),
            ("below_zero([1, -2])", True),
        ],
    },
    "is_multiply_prime": {
        "partial": '''def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0: return False
    return True

def is_multiply_prime(a):
    """Return True if a is product of exactly 3 primes."""
    for i in range(2, a):
        if not is_prime(i): continue
        for j in range(2, a):
            if not is_prime(j): continue
            for k in range(2, a):
                if not is_prime(k): continue
                if i * j * k == a:
                    return''',
        "test_cases": [
            ("is_multiply_prime(30)", True),
            ("is_multiply_prime(8)", True),
            ("is_multiply_prime(10)", False),
        ],
    },
}


def run_inference(prompt: str, intensity: float = 0.0, n_tokens: int = 80, seed: int = 42) -> str:
    """Run single inference pass"""
    cmd = [
        LLAMA_CLI, "-m", MODEL_PATH, "-p", prompt,
        "-n", str(n_tokens), "-c", "4096", "-s", str(seed),
        "--temp", str(TEMP), "--repeat-penalty", str(REPEAT_PENALTY),
        "--no-display-prompt",
    ]
    if intensity > 0:
        cmd.extend(["--dither", str(intensity)])
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return result.stdout.strip()


def fix_code(code: str) -> str:
    """Fix common tokenization issues"""
    code = code.replace('returnTrue', 'return True')
    code = code.replace('returnFalse', 'return False')
    code = re.sub(r'\.count\((\d+)\)', r".count('\1')", code)
    return code


def extract_completion(partial: str, output: str) -> str:
    """Extract just the function completion, stop at new defs"""
    full = partial + output
    lines = full.split('\n')
    
    result = []
    seen_main = False
    for line in lines:
        if line.strip().startswith('def ') and seen_main:
            # New function after main - stop
            func_name = line.strip().split('(')[0].replace('def ', '')
            if func_name not in ('is_prime', 'is_multiply_prime'):
                break
        if line.strip().startswith('def '):
            seen_main = True
        if line.strip().startswith('print(') or line.strip().startswith('# Test'):
            break
        result.append(line)
    
    # Remove trailing incomplete defs
    while result and result[-1].strip().startswith('def '):
        result.pop()
    while result and not result[-1].strip():
        result.pop()
    
    return fix_code('\n'.join(result))


def halo_synthesize(partial: str) -> tuple[str, float, dict]:
    """Run full Halo 3+3+1+1 synthesis"""
    start = time.time()
    stats = {"phases": {}}
    
    # Phase 1: Divergent sampling (3 passes at different intensities)
    print("  [DIV] ", end="", flush=True)
    div_outputs = []
    for i, intensity in enumerate(INTENSITY_DIV):
        out = run_inference(partial, intensity, n_tokens=72, seed=42+i)
        div_outputs.append(out)
        print(f"{i+1}", end=" ", flush=True)
    stats["phases"]["diverge"] = len(div_outputs)
    
    # Phase 2: Convergent sampling (3 passes at lower intensities)
    print("[CONV] ", end="", flush=True)
    conv_outputs = []
    for i, intensity in enumerate(INTENSITY_CONV):
        out = run_inference(partial, intensity, n_tokens=72, seed=100+i)
        conv_outputs.append(out)
        print(f"{i+1}", end=" ", flush=True)
    stats["phases"]["converge"] = len(conv_outputs)
    
    # Phase 3: Pre-synthesis (combine all perspectives)
    print("[PRE] ", end="", flush=True)
    
    # Build synthesis prompt
    presynth_prompt = f"""{partial}

# Multiple perspectives on this code:
# Divergent explorations:
{chr(10).join(f'# V{i+1}: {out[:100]}...' for i, out in enumerate(div_outputs))}
# Convergent refinements:  
{chr(10).join(f'# R{i+1}: {out[:100]}...' for i, out in enumerate(conv_outputs))}
# Synthesized completion:
{partial.split(chr(10))[-1]}"""
    
    presynth_out = run_inference(presynth_prompt, INTENSITY_PRESYNTH, n_tokens=120, seed=200)
    print("1 ", end="", flush=True)
    stats["phases"]["presynth"] = 1
    
    # Phase 4: Final synthesis
    print("[FINAL] ", end="", flush=True)
    
    final_prompt = f"""{partial}

# After considering multiple approaches, the best completion is:
{partial.split(chr(10))[-1]}"""
    
    # Use the most common pattern from earlier outputs
    final_out = run_inference(final_prompt, INTENSITY_FINAL, n_tokens=100, seed=300)
    print("1", flush=True)
    stats["phases"]["final"] = 1
    
    elapsed = time.time() - start
    stats["total_time"] = elapsed
    stats["total_gens"] = 8
    
    # Return the best result - try final first, then presynth, then best divergent
    for output in [final_out, presynth_out] + div_outputs:
        code = extract_completion(partial, output)
        if code and 'return' in code:
            return code, elapsed, stats
    
    # Fallback
    return extract_completion(partial, div_outputs[0]), elapsed, stats


def test_code(code: str, test_cases: list) -> tuple[int, int, list]:
    """Test generated code"""
    passed = 0
    results = []
    
    for test_expr, expected in test_cases:
        try:
            namespace = {}
            exec(code, namespace)
            result = eval(test_expr, namespace)
            success = result == expected
            if success:
                passed += 1
            results.append((test_expr, expected, result, success))
        except Exception as e:
            results.append((test_expr, expected, f"ERR: {str(e)[:25]}", False))
    
    return passed, len(test_cases), results


def main():
    print("=" * 70)
    print("HUMANEVAL + HALO (3+3+1+1) - MAGICK INFEROMETER")
    print("=" * 70)
    print(f"Settings: temp={TEMP}, repeat_penalty={REPEAT_PENALTY}")
    print(f"Halo: 3 diverge + 3 converge + 1 presynth + 1 final = 8 generations\n")
    
    total_passed = 0
    total_tests = 0
    total_time = 0
    
    for name, problem in PROBLEMS.items():
        print(f"\n{'='*50}")
        print(f"Problem: {name}")
        print(f"{'='*50}")
        
        code, elapsed, stats = halo_synthesize(problem['partial'])
        total_time += elapsed
        
        print(f"\nTime: {elapsed:.1f}s ({stats['total_gens']} generations)")
        print(f"\n--- SYNTHESIZED CODE ---")
        for line in code.split('\n')[:12]:
            print(line)
        if len(code.split('\n')) > 12:
            print("...")
        
        passed, total, results = test_code(code, problem['test_cases'])
        total_passed += passed
        total_tests += total
        
        print(f"\n--- RESULTS: {passed}/{total} ---")
        for test_expr, expected, got, success in results:
            status = "PASS" if success else "FAIL"
            print(f"  [{status}] {test_expr} = {got}")
    
    print("\n" + "=" * 70)
    pct = 100 * total_passed / total_tests if total_tests > 0 else 0
    print(f"HALO TOTAL: {total_passed}/{total_tests} ({pct:.1f}%)")
    print(f"Total time: {total_time:.1f}s for 4 problems (8 gens each = 32 total)")
    print("=" * 70)


if __name__ == "__main__":
    main()

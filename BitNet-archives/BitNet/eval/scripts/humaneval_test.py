#!/usr/bin/env python3
"""
HumanEval Random Sample Test - Magick Inferometer Edition
Tests 4 random HumanEval problems with vanilla BitNet vs Halo

Problems selected:
1. HumanEval/31: is_prime - Check if number is prime
2. HumanEval/36: fizz_buzz - Count 7s in numbers divisible by 11 or 13
3. HumanEval/75: is_multiply_prime - Check if number is product of 3 primes  
4. HumanEval/3: below_zero - Check if running sum goes below zero
"""

import subprocess
import time
import json
from pathlib import Path

# Optimal settings from math testing
TEMP = 0.08645
REPEAT_PENALTY = 1.285
TOP_K = 37
TOP_P = 0.963
CTX = 4096
N_PREDICT = 512

MODEL_PATH = "/home/ztflynn/000/randy/b158/BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
LLAMA_CLI = "/home/ztflynn/000/randy/b158/BitNet/build/bin/llama-cli"

# HumanEval problems in standard format
PROBLEMS = {
    "HumanEval/31": {
        "name": "is_prime",
        "prompt": '''def is_prime(n):
    """Return true if a given number is prime, and false otherwise.
    >>> is_prime(6)
    False
    >>> is_prime(101)
    True
    >>> is_prime(11)
    True
    >>> is_prime(13441)
    True
    >>> is_prime(61)
    True
    >>> is_prime(4)
    False
    >>> is_prime(1)
    False
    """
''',
        "test_cases": [
            ("is_prime(6)", False),
            ("is_prime(101)", True),
            ("is_prime(11)", True),
            ("is_prime(61)", True),
            ("is_prime(4)", False),
            ("is_prime(1)", False),
            ("is_prime(2)", True),
            ("is_prime(17)", True),
        ],
        "canonical": '''    if n < 2:
        return False
    for k in range(2, n - 1):
        if n % k == 0:
            return False
    return True'''
    },
    
    "HumanEval/36": {
        "name": "fizz_buzz",
        "prompt": '''def fizz_buzz(n: int):
    """Return the number of times the digit 7 appears in integers less than n which are divisible by 11 or 13.
    >>> fizz_buzz(50)
    0
    >>> fizz_buzz(78)
    2
    >>> fizz_buzz(79)
    3
    """
''',
        "test_cases": [
            ("fizz_buzz(50)", 0),
            ("fizz_buzz(78)", 2),
            ("fizz_buzz(79)", 3),
        ],
        "canonical": '''    ns = []
    for i in range(n):
        if i % 11 == 0 or i % 13 == 0:
            ns.append(i)
    s = ''.join(list(map(str, ns)))
    ans = 0
    for c in s:
        ans += (c == '7')
    return ans'''
    },
    
    "HumanEval/75": {
        "name": "is_multiply_prime",
        "prompt": '''def is_multiply_prime(a):
    """Write a function that returns true if the given number is the multiplication of 3 prime numbers
    and false otherwise.
    Knowing that (a) is less than 100. 
    Example:
    is_multiply_prime(30) == True
    30 = 2 * 3 * 5
    """
''',
        "test_cases": [
            ("is_multiply_prime(30)", True),  # 2*3*5
            ("is_multiply_prime(8)", True),   # 2*2*2
            ("is_multiply_prime(10)", False), # 2*5 (only 2 primes)
            ("is_multiply_prime(125)", True), # 5*5*5
            ("is_multiply_prime(6)", False),  # 2*3 (only 2 primes)
        ],
        "canonical": '''    def is_prime(n):
        for j in range(2, n):
            if n % j == 0:
                return False
        return True

    for i in range(2, 101):
        if not is_prime(i): continue
        for j in range(2, 101):
            if not is_prime(j): continue
            for k in range(2, 101):
                if not is_prime(k): continue
                if i*j*k == a: return True
    return False'''
    },
    
    "HumanEval/3": {
        "name": "below_zero",
        "prompt": '''from typing import List

def below_zero(operations: List[int]) -> bool:
    """ You're given a list of deposit and withdrawal operations on a bank account that starts with
    zero balance. Your task is to detect if at any point the balance of account falls below zero, and
    at that point function should return True. Otherwise it should return False.
    >>> below_zero([1, 2, 3])
    False
    >>> below_zero([1, 2, -4, 5])
    True
    """
''',
        "test_cases": [
            ("below_zero([1, 2, 3])", False),
            ("below_zero([1, 2, -4, 5])", True),
            ("below_zero([])", False),
            ("below_zero([1, -1, 2, -2])", False),
            ("below_zero([1, -2])", True),
        ],
        "canonical": '''    balance = 0
    for op in operations:
        balance += op
        if balance < 0:
            return True
    return False'''
    },
}


def run_inference(prompt: str, use_dither: bool = False) -> tuple[str, float]:
    """Run inference and return (output, time_taken)"""
    cmd = [
        LLAMA_CLI,
        "-m", MODEL_PATH,
        "-p", prompt,
        "-n", str(N_PREDICT),
        "-c", str(CTX),
        "--temp", str(TEMP),
        "--repeat-penalty", str(REPEAT_PENALTY),
        "--top-k", str(TOP_K),
        "--top-p", str(TOP_P),
        "--no-display-prompt",
    ]
    
    if use_dither:
        cmd.extend(["--dither", "0.0073"])
    
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        elapsed = time.time() - start
        return result.stdout.strip(), elapsed
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]", 120.0


def extract_code(output: str, function_name: str) -> str:
    """Extract just the function body from the output - stop at next def or unindented code"""
    lines = output.split('\n')
    code_lines = []
    
    for line in lines:
        # Stop at markdown code blocks
        if line.strip().startswith('```'):
            break
        # Stop at a new function definition (not nested)
        if line.strip().startswith('def ') and len(code_lines) > 0:
            break
        # Stop at print() or main() calls (test code)
        if line.strip().startswith('print(') or line.strip().startswith('if __name__'):
            break
        # Stop at comments that look like test headers or explanations
        if line.strip().startswith('# Test') or line.strip().startswith('# Your'):
            break
        code_lines.append(line)
    
    # Normalize indentation: first substantive line should have 4 spaces
    result_lines = code_lines
    
    # Find the indentation of the first non-empty line
    first_indent = 0
    for line in result_lines:
        if line.strip():
            first_indent = len(line) - len(line.lstrip())
            break
    
    # Calculate adjustment needed (we want first line at 4 spaces)
    adjustment = 4 - first_indent
    
    # Apply adjustment to all lines
    fixed_lines = []
    for line in result_lines:
        if not line.strip():
            fixed_lines.append('')
        else:
            current_indent = len(line) - len(line.lstrip())
            new_indent = max(0, current_indent + adjustment)
            fixed_lines.append(' ' * new_indent + line.lstrip())
    
    return '\n'.join(fixed_lines)


def test_code(prompt: str, code: str, test_cases: list) -> tuple[int, int, list]:
    """Test generated code against test cases"""
    # Combine prompt with generated code
    full_code = prompt + code
    
    passed = 0
    total = len(test_cases)
    results = []
    
    for test_expr, expected in test_cases:
        try:
            # Create isolated namespace
            namespace = {}
            exec(full_code, namespace)
            result = eval(test_expr, namespace)
            success = result == expected
            if success:
                passed += 1
            results.append((test_expr, expected, result, success))
        except Exception as e:
            results.append((test_expr, expected, f"ERROR: {e}", False))
    
    return passed, total, results


def main():
    print("=" * 70)
    print("HUMANEVAL RANDOM SAMPLE TEST - MAGICK INFEROMETER")
    print("=" * 70)
    print(f"\nSettings: temp={TEMP}, repeat_penalty={REPEAT_PENALTY}, top_k={TOP_K}")
    print(f"Testing 4 problems with vanilla BitNet\n")
    
    all_results = {}
    total_passed = 0
    total_tests = 0
    
    for problem_id, problem in PROBLEMS.items():
        print("-" * 70)
        print(f"\n{problem_id}: {problem['name']}")
        print("-" * 70)
        
        # Create completion prompt - just the function, model should continue
        prompt = problem['prompt']
        
        print(f"\nGenerating code...")
        output, elapsed = run_inference(prompt, use_dither=False)
        print(f"Time: {elapsed:.1f}s")
        
        # Extract clean code
        clean_code = extract_code(output, problem['name'])
        
        # Show raw and cleaned output
        print(f"\n--- RAW OUTPUT ---")
        print(output[:400] + "..." if len(output) > 400 else output)
        print(f"\n--- CLEANED CODE ---")
        print(clean_code[:300] + "..." if len(clean_code) > 300 else clean_code)
        
        # Try to test the code
        print(f"\n--- TESTING ---")
        
        # Use the cleaned output as code body
        passed, total, results = test_code(problem['prompt'], clean_code, problem['test_cases'])
        
        # If that failed, try with canonical solution for comparison
        print(f"\nGenerated code results: {passed}/{total} tests passed")
        for test_expr, expected, got, success in results:
            status = "PASS" if success else "FAIL"
            print(f"  [{status}] {test_expr} -> expected {expected}, got {got}")
        
        total_passed += passed
        total_tests += total
        
        all_results[problem_id] = {
            "name": problem['name'],
            "output": output,
            "time": elapsed,
            "passed": passed,
            "total": total,
            "results": [(t, str(e), str(g), s) for t, e, g, s in results]
        }
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal: {total_passed}/{total_tests} tests passed ({100*total_passed/total_tests:.1f}%)")
    print(f"\nPer-problem breakdown:")
    for problem_id, result in all_results.items():
        print(f"  {problem_id}: {result['passed']}/{result['total']} ({result['time']:.1f}s)")
    
    # Save results
    output_path = Path("/home/ztflynn/000/randy/b158/BitNet/eval/data/humaneval_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

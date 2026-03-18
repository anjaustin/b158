#!/usr/bin/env python3
"""
HumanEval Test with Chat-Style Prompts
Tests 4 random HumanEval problems with BitNet using chat formatting
"""

import subprocess
import time
import re
import json
from pathlib import Path

TEMP = 0.08645
REPEAT_PENALTY = 1.285
MODEL_PATH = "/home/ztflynn/000/randy/b158/BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
LLAMA_CLI = "/home/ztflynn/000/randy/b158/BitNet/build/bin/llama-cli"

PROBLEMS = {
    "HumanEval/31": {
        "name": "is_prime",
        "chat_prompt": """Write a Python function is_prime(n) that returns True if n is prime, False otherwise.

Examples:
- is_prime(6) returns False
- is_prime(101) returns True  
- is_prime(11) returns True
- is_prime(1) returns False

Just output the Python code:""",
        "test_cases": [
            ("is_prime(6)", False),
            ("is_prime(101)", True),
            ("is_prime(11)", True),
            ("is_prime(61)", True),
            ("is_prime(4)", False),
            ("is_prime(1)", False),
            ("is_prime(2)", True),
        ],
    },
    
    "HumanEval/36": {
        "name": "fizz_buzz",
        "chat_prompt": """Write a Python function fizz_buzz(n) that returns the count of how many times the digit 7 appears in numbers less than n that are divisible by 11 or 13.

Examples:
- fizz_buzz(50) returns 0
- fizz_buzz(78) returns 2
- fizz_buzz(79) returns 3

Just output the Python code:""",
        "test_cases": [
            ("fizz_buzz(50)", 0),
            ("fizz_buzz(78)", 2),
            ("fizz_buzz(79)", 3),
        ],
    },
    
    "HumanEval/75": {
        "name": "is_multiply_prime",
        "chat_prompt": """Write a Python function is_multiply_prime(a) that returns True if the number a is the product of exactly 3 prime numbers (not necessarily distinct), False otherwise. The number a is less than 100.

Examples:
- is_multiply_prime(30) returns True (30 = 2 * 3 * 5)
- is_multiply_prime(8) returns True (8 = 2 * 2 * 2)
- is_multiply_prime(10) returns False (10 = 2 * 5, only 2 primes)

Just output the Python code:""",
        "test_cases": [
            ("is_multiply_prime(30)", True),
            ("is_multiply_prime(8)", True),
            ("is_multiply_prime(10)", False),
            ("is_multiply_prime(6)", False),
        ],
    },
    
    "HumanEval/3": {
        "name": "below_zero",
        "chat_prompt": """Write a Python function below_zero(operations) that takes a list of integers representing deposits (positive) and withdrawals (negative) on a bank account starting at zero balance. Return True if the balance ever goes below zero, False otherwise.

Examples:
- below_zero([1, 2, 3]) returns False
- below_zero([1, 2, -4, 5]) returns True (balance goes to -1 after third operation)

Just output the Python code:""",
        "test_cases": [
            ("below_zero([1, 2, 3])", False),
            ("below_zero([1, 2, -4, 5])", True),
            ("below_zero([])", False),
            ("below_zero([1, -2])", True),
        ],
    },
}


def run_inference(prompt: str) -> tuple[str, float]:
    cmd = [
        LLAMA_CLI, "-m", MODEL_PATH, "-p", prompt,
        "-n", "300", "-c", "4096",
        "--temp", str(TEMP), "--repeat-penalty", str(REPEAT_PENALTY),
        "--no-display-prompt",
    ]
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return result.stdout.strip(), time.time() - start


def extract_python_code(output: str) -> str:
    """Extract Python code from markdown code blocks or raw output"""
    # Try to find ```python ... ``` block
    match = re.search(r'```python\s*(.*?)```', output, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Try to find ``` ... ``` block
    match = re.search(r'```\s*(.*?)```', output, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Look for def ... and extract until we hit explanation text
    lines = output.split('\n')
    code_lines = []
    in_code = False
    for line in lines:
        if line.strip().startswith('def '):
            in_code = True
        if in_code:
            if line.strip() and not line.startswith(' ') and not line.startswith('\t') and not line.startswith('def'):
                break
            code_lines.append(line)
    
    if code_lines:
        return '\n'.join(code_lines).strip()
    
    return output


def test_code(code: str, test_cases: list) -> tuple[int, int, list]:
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
            results.append((test_expr, expected, f"ERROR: {e}", False))
    
    return passed, len(test_cases), results


def main():
    print("=" * 70)
    print("HUMANEVAL CHAT-STYLE TEST - MAGICK INFEROMETER")
    print("=" * 70)
    print(f"\nSettings: temp={TEMP}, repeat_penalty={REPEAT_PENALTY}\n")
    
    all_results = {}
    total_passed = 0
    total_tests = 0
    
    for problem_id, problem in PROBLEMS.items():
        print("-" * 70)
        print(f"{problem_id}: {problem['name']}")
        print("-" * 70)
        
        output, elapsed = run_inference(problem['chat_prompt'])
        print(f"Time: {elapsed:.1f}s")
        
        code = extract_python_code(output)
        print(f"\n--- EXTRACTED CODE ---")
        print(code[:400] + "..." if len(code) > 400 else code)
        
        passed, total, results = test_code(code, problem['test_cases'])
        
        print(f"\n--- RESULTS: {passed}/{total} ---")
        for test_expr, expected, got, success in results:
            status = "PASS" if success else "FAIL"
            print(f"  [{status}] {test_expr} = {got} (expected {expected})")
        
        total_passed += passed
        total_tests += total
        all_results[problem_id] = {"passed": passed, "total": total, "time": elapsed}
    
    print("\n" + "=" * 70)
    print(f"TOTAL: {total_passed}/{total_tests} ({100*total_passed/total_tests:.1f}%)")
    print("=" * 70)
    
    # Save
    with open("/home/ztflynn/000/randy/b158/BitNet/eval/data/humaneval_chat_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()

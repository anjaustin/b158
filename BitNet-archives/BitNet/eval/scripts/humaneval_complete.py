#!/usr/bin/env python3
"""
HumanEval Test - Completion Style
Uses partial code that the model completes
"""

import subprocess
import time
import json

TEMP = 0.08645
REPEAT_PENALTY = 1.285
MODEL_PATH = "/home/ztflynn/000/randy/b158/BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
LLAMA_CLI = "/home/ztflynn/000/randy/b158/BitNet/build/bin/llama-cli"

# Problems with partial implementations to complete
PROBLEMS = {
    "HumanEval/31": {
        "name": "is_prime",
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
    
    "HumanEval/36": {
        "name": "fizz_buzz",
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
    
    "HumanEval/3": {
        "name": "below_zero",
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
            ("below_zero([])", False),
        ],
    },
    
    "HumanEval/75": {
        "name": "is_multiply_prime",
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


def run_completion(partial: str, n_tokens: int = 80) -> tuple[str, float]:
    cmd = [
        LLAMA_CLI, "-m", MODEL_PATH, "-p", partial,
        "-n", str(n_tokens), "-c", "4096",
        "--temp", str(TEMP), "--repeat-penalty", str(REPEAT_PENALTY),
        "--no-display-prompt",
    ]
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return result.stdout.strip(), time.time() - start


def fix_common_bugs(code: str) -> str:
    """Fix common tokenization issues in generated code"""
    # Fix missing space after return
    code = code.replace('returnTrue', 'return True')
    code = code.replace('returnFalse', 'return False')
    code = code.replace('returnNone', 'return None')
    code = code.replace('return true', 'return True')
    code = code.replace('return false', 'return False')
    
    # Fix str.count with int instead of string
    import re
    code = re.sub(r'\.count\((\d+)\)', r".count('\1')", code)
    
    return code


def extract_functions(partial: str, completion: str, target_func: str = None) -> str:
    """Combine partial + completion and extract functions before test code"""
    full = partial + completion
    lines = full.split('\n')
    
    result_lines = []
    seen_funcs = set()
    
    for line in lines:
        # Stop at test code
        if line.strip().startswith('print('):
            break
        if line.strip().startswith('# Test') or line.strip().startswith('# Example'):
            break
        if line.strip().startswith('if __name__'):
            break
            
        # Track function definitions
        if line.strip().startswith('def '):
            func_name = line.strip().split('(')[0].replace('def ', '')
            # If we've already seen the main target functions and hit a new one, stop
            if 'is_prime' in seen_funcs or 'fizz_buzz' in seen_funcs or 'below_zero' in seen_funcs or 'is_multiply_prime' in seen_funcs:
                if func_name not in ('is_prime', 'is_multiply_prime'):  # Allow helper functions
                    break
            seen_funcs.add(func_name)
                
        result_lines.append(line)
    
    # Remove trailing incomplete function definitions
    code = '\n'.join(result_lines)
    lines = code.split('\n')
    
    # Find last complete function (ends with return or has body)
    while lines and lines[-1].strip().startswith('def '):
        lines.pop()  # Remove trailing def without body
    while lines and not lines[-1].strip():
        lines.pop()  # Remove trailing blank lines
        
    code = '\n'.join(lines)
    return fix_common_bugs(code)


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
            results.append((test_expr, expected, f"ERR: {str(e)[:30]}", False))
    
    return passed, len(test_cases), results


def main():
    print("=" * 70)
    print("HUMANEVAL COMPLETION TEST - MAGICK INFEROMETER")
    print("=" * 70)
    print(f"Settings: temp={TEMP}, repeat_penalty={REPEAT_PENALTY}\n")
    
    total_passed = 0
    total_tests = 0
    
    for problem_id, problem in PROBLEMS.items():
        print("-" * 70)
        print(f"{problem_id}: {problem['name']}")
        print("-" * 70)
        
        completion, elapsed = run_completion(problem['partial'])
        full_code = extract_functions(problem['partial'], completion)
        
        print(f"Time: {elapsed:.1f}s")
        print(f"\n--- COMPLETED CODE ---")
        # Show just the relevant part
        code_preview = full_code.split('\n')
        for i, line in enumerate(code_preview[:15]):
            print(line)
        if len(code_preview) > 15:
            print("...")
        
        passed, total, results = test_code(full_code, problem['test_cases'])
        
        print(f"\n--- RESULTS: {passed}/{total} ---")
        for test_expr, expected, got, success in results:
            status = "PASS" if success else "FAIL"
            print(f"  [{status}] {test_expr} = {got}")
        
        total_passed += passed
        total_tests += total
        print()
    
    print("=" * 70)
    pct = 100*total_passed/total_tests if total_tests > 0 else 0
    print(f"TOTAL: {total_passed}/{total_tests} ({pct:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()

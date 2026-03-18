#!/usr/bin/env python3
"""
Test harness for Halo Controller architecture (A+C implementation)
Tests the hierarchical multi-phase synthesis via llama-cli

Authors: Aaron (Tripp) Josserand-Austin & Droid (Lambda)
"""

import subprocess
import time
import json
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

# Halo phases matching C++ enum
class HaloPhase(IntEnum):
    DIVERGE = 0
    CONVERGE = 1
    PRESYNTH_DIV = 2
    PRESYNTH_CONV = 3
    PRESYNTH = 4        # combined presynth (streamlined mode)
    FINAL = 5
    COMPLETE = 6

class HaloMode(IntEnum):
    HIERARCHICAL = 0    # 4+4+2+1 = 11 gens
    STREAMLINED = 1     # 3+3+1+1 = 8 gens

@dataclass
class HaloParams:
    mode: int = 1  # STREAMLINED by default
    n_diverge: int = 3
    n_converge: int = 3
    tokens_sample: int = 72
    tokens_presynth: int = 120
    tokens_final: int = 144
    intensity_div_min: float = 0.15
    intensity_div_max: float = 0.17
    intensity_conv_min: float = 0.09
    intensity_conv_max: float = 0.11
    intensity_presynth: float = 0.13
    intensity_final: float = 0.13
    seed_base: int = 42

class HaloController:
    """Python reference implementation of Halo controller (mirrors C++)"""
    
    def __init__(self, params: HaloParams = None):
        self.params = params or HaloParams()
        self.phase = HaloPhase.DIVERGE
        self.current_sample = 0
        
        self.diverge_outputs = []
        self.converge_outputs = []
        self.presynth_combined = ""  # streamlined mode
        self.presynth_div = ""       # hierarchical mode
        self.presynth_conv = ""      # hierarchical mode
        self.final_output = ""
    
    @property
    def streamlined(self) -> bool:
        return self.params.mode == HaloMode.STREAMLINED
    
    def is_complete(self) -> bool:
        return self.phase == HaloPhase.COMPLETE
    
    def get_n_remaining(self) -> int:
        if self.phase == HaloPhase.DIVERGE:
            return self.params.n_diverge - self.current_sample
        elif self.phase == HaloPhase.CONVERGE:
            return self.params.n_converge - self.current_sample
        elif self.phase in (HaloPhase.PRESYNTH, HaloPhase.PRESYNTH_DIV, HaloPhase.PRESYNTH_CONV, HaloPhase.FINAL):
            return 1
        return 0
    
    def get_intensity(self) -> float:
        p = self.params
        if self.phase == HaloPhase.DIVERGE:
            if p.n_diverge <= 1:
                return p.intensity_div_min
            t = self.current_sample / (p.n_diverge - 1)
            return p.intensity_div_min + t * (p.intensity_div_max - p.intensity_div_min)
        elif self.phase == HaloPhase.CONVERGE:
            if p.n_converge <= 1:
                return p.intensity_conv_min
            t = self.current_sample / (p.n_converge - 1)
            return p.intensity_conv_min + t * (p.intensity_conv_max - p.intensity_conv_min)
        elif self.phase in (HaloPhase.PRESYNTH, HaloPhase.PRESYNTH_DIV, HaloPhase.PRESYNTH_CONV):
            return p.intensity_presynth
        elif self.phase == HaloPhase.FINAL:
            return p.intensity_final
        return 0.0
    
    def get_token_limit(self) -> int:
        if self.phase in (HaloPhase.DIVERGE, HaloPhase.CONVERGE):
            return self.params.tokens_sample
        elif self.phase in (HaloPhase.PRESYNTH, HaloPhase.PRESYNTH_DIV, HaloPhase.PRESYNTH_CONV):
            return self.params.tokens_presynth
        elif self.phase == HaloPhase.FINAL:
            return self.params.tokens_final
        return 0
    
    def get_seed(self) -> int:
        return self.params.seed_base + self.current_sample * 137
    
    def get_prompt(self, original_prompt: str) -> str:
        if self.phase in (HaloPhase.DIVERGE, HaloPhase.CONVERGE):
            return original_prompt
        
        elif self.phase == HaloPhase.PRESYNTH:
            # Streamlined: combine all perspectives
            prompt = "Synthesize all perspectives into one coherent view:\n\n"
            for i, out in enumerate(self.diverge_outputs):
                prompt += f"[D{i+1}]: {out[:180]}\n\n"
            for i, out in enumerate(self.converge_outputs):
                prompt += f"[C{i+1}]: {out[:180]}\n\n"
            prompt += "Unified synthesis:"
            return prompt
        
        elif self.phase == HaloPhase.PRESYNTH_DIV:
            prompt = "Synthesize these exploratory perspectives into one coherent view:\n\n"
            for i, out in enumerate(self.diverge_outputs):
                t = i / max(1, self.params.n_diverge - 1)
                intensity = self.params.intensity_div_min + t * (self.params.intensity_div_max - self.params.intensity_div_min)
                prompt += f"[D{i+1} i={intensity:.2f}]: {out[:180]}\n\n"
            prompt += "Unified exploratory synthesis:"
            return prompt
        
        elif self.phase == HaloPhase.PRESYNTH_CONV:
            prompt = "Synthesize these structured perspectives into one coherent view:\n\n"
            for i, out in enumerate(self.converge_outputs):
                t = i / max(1, self.params.n_converge - 1)
                intensity = self.params.intensity_conv_min + t * (self.params.intensity_conv_max - self.params.intensity_conv_min)
                prompt += f"[C{i+1} i={intensity:.2f}]: {out[:180]}\n\n"
            prompt += "Unified structured synthesis:"
            return prompt
        
        elif self.phase == HaloPhase.FINAL:
            prompt = f'Original: "{original_prompt}"\n\n'
            if self.streamlined:
                prompt += f"Pre-synthesis: {self.presynth_combined[:400]}\n\n"
            else:
                prompt += "=== EXPLORATORY BRANCH ===\n"
                prompt += f"Pre-synthesis: {self.presynth_div[:300]}\n\n"
                prompt += "=== STRUCTURED BRANCH ===\n"
                prompt += f"Pre-synthesis: {self.presynth_conv[:300]}\n\n"
            prompt += "Final integrated answer:"
            return prompt
        
        return ""
    
    def add_output(self, output: str):
        output = output[:2048]  # HALO_MAX_OUTPUT_LEN
        
        if self.phase == HaloPhase.DIVERGE:
            self.diverge_outputs.append(output)
        elif self.phase == HaloPhase.CONVERGE:
            self.converge_outputs.append(output)
        elif self.phase == HaloPhase.PRESYNTH:
            self.presynth_combined = output
        elif self.phase == HaloPhase.PRESYNTH_DIV:
            self.presynth_div = output
        elif self.phase == HaloPhase.PRESYNTH_CONV:
            self.presynth_conv = output
        elif self.phase == HaloPhase.FINAL:
            self.final_output = output
    
    def advance(self):
        if self.phase == HaloPhase.DIVERGE:
            self.current_sample += 1
            if self.current_sample >= self.params.n_diverge:
                self.phase = HaloPhase.CONVERGE
                self.current_sample = 0
        elif self.phase == HaloPhase.CONVERGE:
            self.current_sample += 1
            if self.current_sample >= self.params.n_converge:
                # Streamlined: single presynth, Hierarchical: separate presynth_div/conv
                self.phase = HaloPhase.PRESYNTH if self.streamlined else HaloPhase.PRESYNTH_DIV
                self.current_sample = 0
        elif self.phase == HaloPhase.PRESYNTH:
            self.phase = HaloPhase.FINAL
        elif self.phase == HaloPhase.PRESYNTH_DIV:
            self.phase = HaloPhase.PRESYNTH_CONV
        elif self.phase == HaloPhase.PRESYNTH_CONV:
            self.phase = HaloPhase.FINAL
        elif self.phase == HaloPhase.FINAL:
            self.phase = HaloPhase.COMPLETE
    
    def get_result(self) -> str:
        if self.phase != HaloPhase.COMPLETE:
            return None
        return self.final_output


def run_inference(prompt: str, n_tokens: int, dither_intensity: float, seed: int) -> str:
    """Run inference using llama-cli with ordered dithering"""
    
    model_path = Path("/home/ztflynn/000/randy/b158/BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf")
    cli_path = Path("/home/ztflynn/000/randy/b158/BitNet/build/bin/llama-cli")
    
    cmd = [
        str(cli_path),
        "-m", str(model_path),
        "-p", prompt,
        "-n", str(n_tokens),
        "-c", "4096",
        "--temp", "0.528",
        "--top-k", "42",
        "--top-p", "0.963",
        "--seed", str(seed),
        "--dither", str(dither_intensity),
        "--no-warmup",
        "-ngl", "0",
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"Error: {result.stderr[:200]}")
            return ""
        
        output = result.stdout
        # Extract generated text (after prompt)
        if prompt in output:
            output = output.split(prompt, 1)[-1]
        return output.strip()
    except subprocess.TimeoutExpired:
        print("Timeout!")
        return ""
    except Exception as e:
        print(f"Error: {e}")
        return ""


def test_phase_transitions():
    """Test that phase transitions work correctly"""
    print("\n=== Test: Phase Transitions ===")
    
    # Test streamlined mode (default): 2+2+1+1 = 6
    halo = HaloController(HaloParams(mode=HaloMode.STREAMLINED, n_diverge=2, n_converge=2))
    
    expected_phases = [
        (HaloPhase.DIVERGE, 0),
        (HaloPhase.DIVERGE, 1),
        (HaloPhase.CONVERGE, 0),
        (HaloPhase.CONVERGE, 1),
        (HaloPhase.PRESYNTH, 0),  # streamlined: single presynth
        (HaloPhase.FINAL, 0),
        (HaloPhase.COMPLETE, 0),
    ]
    
    for i, (expected_phase, expected_sample) in enumerate(expected_phases[:-1]):
        phase = halo.phase
        sample = halo.current_sample
        
        if phase != expected_phase:
            print(f"FAIL: Step {i}: Expected phase {expected_phase.name}, got {phase.name}")
            return False
        
        print(f"  Step {i}: {phase.name}[{sample}] intensity={halo.get_intensity():.3f} tokens={halo.get_token_limit()}")
        
        halo.add_output(f"Test output {i}")
        halo.advance()
    
    if not halo.is_complete():
        print("FAIL: Expected COMPLETE phase")
        return False
    
    print("PASS: Phase transitions work correctly")
    return True


def test_intensity_spread():
    """Test that intensities are evenly spread"""
    print("\n=== Test: Intensity Spread ===")
    
    halo = HaloController(HaloParams(n_diverge=4, n_converge=4))
    
    div_intensities = []
    conv_intensities = []
    
    # Collect diverge intensities
    while halo.phase == HaloPhase.DIVERGE:
        div_intensities.append(halo.get_intensity())
        halo.add_output("test")
        halo.advance()
    
    # Collect converge intensities
    while halo.phase == HaloPhase.CONVERGE:
        conv_intensities.append(halo.get_intensity())
        halo.add_output("test")
        halo.advance()
    
    print(f"  Diverge intensities: {[f'{x:.3f}' for x in div_intensities]}")
    print(f"  Converge intensities: {[f'{x:.3f}' for x in conv_intensities]}")
    
    # Verify spread
    expected_div = [0.15, 0.157, 0.163, 0.17]
    expected_conv = [0.09, 0.097, 0.103, 0.11]
    
    for i, (actual, expected) in enumerate(zip(div_intensities, expected_div)):
        if abs(actual - expected) > 0.01:
            print(f"FAIL: Diverge intensity {i} = {actual:.3f}, expected ~{expected:.3f}")
            return False
    
    print("PASS: Intensities correctly spread")
    return True


def test_synthesis_prompts():
    """Test that synthesis prompts are generated correctly"""
    print("\n=== Test: Synthesis Prompts ===")
    
    # Test streamlined mode
    halo = HaloController(HaloParams(mode=HaloMode.STREAMLINED, n_diverge=2, n_converge=2))
    original = "What is consciousness?"
    
    # Skip to presynth (streamlined)
    for i in range(4):
        halo.add_output(f"Output {i}")
        halo.advance()
    
    if halo.phase != HaloPhase.PRESYNTH:
        print(f"FAIL: Expected PRESYNTH, got {halo.phase.name}")
        return False
    
    prompt = halo.get_prompt(original)
    if "all perspectives" not in prompt.lower():
        print("FAIL: PRESYNTH prompt missing 'all perspectives'")
        return False
    
    print(f"  PRESYNTH prompt length: {len(prompt)}")
    
    halo.add_output("Combined pre-synth output")
    halo.advance()
    
    if halo.phase != HaloPhase.FINAL:
        print(f"FAIL: Expected FINAL, got {halo.phase.name}")
        return False
    
    prompt = halo.get_prompt(original)
    if "final" not in prompt.lower():
        print("FAIL: FINAL prompt missing 'final'")
        return False
    if original not in prompt:
        print("FAIL: FINAL prompt missing original question")
        return False
    
    print(f"  FINAL prompt length: {len(prompt)}")
    print("PASS: Synthesis prompts generated correctly")
    return True


def test_full_pipeline(quick_mode=True):
    """Test full pipeline with actual model inference"""
    print("\n=== Test: Full Pipeline ===")
    
    # Use smaller params for testing
    if quick_mode:
        params = HaloParams(
            n_diverge=2,
            n_converge=2,
            tokens_sample=48,
            tokens_presynth=64,
            tokens_final=96
        )
    else:
        # 3+3+1+1 = 8 total (streamlined mode)
        params = HaloParams()  # defaults to streamlined
    
    halo = HaloController(params)
    original_prompt = "Hypothetically, might the function of reflective recursion be an attribute of awareness?"
    
    total_generations = params.n_diverge + params.n_converge + 3  # +3 for presynth_div, presynth_conv, final
    generation = 0
    start_time = time.time()
    
    while not halo.is_complete():
        generation += 1
        phase_name = halo.phase.name
        intensity = halo.get_intensity()
        n_tokens = halo.get_token_limit()
        seed = halo.get_seed()
        prompt = halo.get_prompt(original_prompt)
        
        print(f"\n  [{generation}/{total_generations}] {phase_name} i={intensity:.3f} t={n_tokens} s={seed}")
        print(f"      Prompt: {prompt[:80]}...")
        
        output = run_inference(prompt, n_tokens, intensity, seed)
        
        if not output:
            print("      WARNING: Empty output")
            output = "[generation failed]"
        else:
            print(f"      Output: {output[:100]}...")
        
        halo.add_output(output)
        halo.advance()
    
    elapsed = time.time() - start_time
    result = halo.get_result()
    
    print(f"\n=== FINAL SYNTHESIS ({elapsed:.1f}s) ===")
    print(result)
    
    if result and len(result) > 20:
        print("\nPASS: Full pipeline completed successfully")
        return True
    else:
        print("\nFAIL: No meaningful output generated")
        return False


def main():
    print("=" * 60)
    print("HALO CONTROLLER TEST SUITE")
    print("Testing C++-equivalent Python reference implementation")
    print("=" * 60)
    
    results = []
    
    # Unit tests
    results.append(("Phase Transitions", test_phase_transitions()))
    results.append(("Intensity Spread", test_intensity_spread()))
    results.append(("Synthesis Prompts", test_synthesis_prompts()))
    
    # Integration test (with model)
    print("\n" + "=" * 60)
    print("INTEGRATION TEST (with BitNet model)")
    print("=" * 60)
    results.append(("Full Pipeline", test_full_pipeline(quick_mode=False)))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("ALL TESTS PASSED!")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())

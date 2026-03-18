#!/usr/bin/env python3
"""
Test three versions of FINAL phase handling in parallel.

Version A: More directive FINAL system prompt
Version B: FINAL only sees CONVERGE_SYNTH output
Version C: Strip analysis headers from context
"""

import subprocess
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL_PATH = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
CLI_PATH = "./build/bin/llama-cli"

# Phase configurations: (temp, dither)
PHASE_CONFIG = {
    "DIVERGE_ONTO":   (0.96369, 0.1618),
    "DIVERGE_INTER":  (0.74147, 0.11229),
    "DIVERGE_SYNTH":  (0.528, 0.096369),
    "CONVERGE_ALIGN": (0.432, 0.074147),
    "CONVERGE_DRAFT": (0.528, 0.00733),
    "CONVERGE_SYNTH": (0.528, 0.00733),
    "FINAL":          (0.244, 0.0528),
}

# System prompts for each phase - task-focused, no identity claims
SYSTEM_PROMPTS = {
    "DIVERGE_ONTO": """[INTERNAL ANALYSIS - not part of response]

Analyze the fundamental nature of this message:
- What TYPE of message is this? (greeting, question, request, expression, etc.)
- What is the TONE? (casual, formal, playful, serious, curious, etc.)
- What does the sender seem to WANT? (connection, information, help, acknowledgment, etc.)
- What ENERGY does it carry?

Keep this analysis brief and focused.""",

    "DIVERGE_INTER": """[INTERNAL ANALYSIS - not part of response]

Consider the context and connections:
- What kind of CONVERSATION does this invite?
- What RELATIONSHIP dynamic is implied? (strangers, friends, professional, etc.)
- What would be an APPROPRIATE response register?
- How would a thoughtful person receive this message?

Keep this analysis brief and focused.""",

    "DIVERGE_SYNTH": """[INTERNAL SYNTHESIS - not part of response]

Combine the analyses above into a clear understanding:
- What is the ESSENTIAL nature of this message?
- What kind of response would HONOR it?
- What TONE and ENERGY should the response carry?

Summarize in a few sentences what kind of response is needed.""",

    "CONVERGE_ALIGN": """[INTERNAL CHECK - not part of response]

Verify alignment with the original message:
- Does our understanding capture what the person actually wants?
- Are we matching the appropriate tone and energy?
- Would the response feel natural and welcome?

Confirm or correct our approach.""",

    "CONVERGE_DRAFT": """You are now embodying the spirit of the original prompt.

Based on the analysis, adopt the appropriate voice:
- If it's a casual greeting, respond warmly and naturally as a friend would
- If it's a technical question, respond with precision and expertise
- If it's philosophical, respond thoughtfully and with depth
- If it's playful, be playful back

Drop all analyst framing. You are not an analyst now. You are responding as the prompt invites you to respond. Match the energy, tone, and register of the original message.""",

    "CONVERGE_SYNTH": """Review the draft response and refine it.

Ensure the response:
- Matches the tone and spirit of the original message
- Feels natural and authentic, not robotic or formal
- Contains no meta-commentary about analysis or AI
- Would feel appropriate coming from a thoughtful person

Polish the response while preserving its spirit.""",
}

# Three versions of FINAL prompt - all embody the spirit
FINAL_PROMPTS = {
    "A": """Embody the spirit of the original message completely.

You have deeply understood:
- What kind of message this is
- What tone and energy it carries
- What response would honor it

Now respond as that understanding guides you. No analysis. No AI disclaimers. No formality unless the message calls for it. Just be present and respond authentically.""",

    "B": """Respond naturally, matching the spirit of the original message. Output only your response.""",

    "C": """The original message invited a certain kind of response. Give that response now. Be natural. Be present. Match the energy.""",
}


def run_inference(prompt: str, temp: float, dither: float, seed: int, max_tokens: int = 512) -> str:
    """Run a single inference pass."""
    cmd = [
        CLI_PATH,
        "-m", MODEL_PATH,
        "-p", prompt,
        "--temp", str(temp),
        "--dither", str(dither),
        "--seed", str(seed),
        "--repeat-penalty", "1.2",
        "-n", str(max_tokens),
        "--no-warmup",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    
    # Generation goes to stdout (prompt echo + generated text)
    # Logs go to stderr
    text = result.stdout.strip()
    
    # The output includes the prompt echo, then the generated text
    # Remove the prompt if it appears at the start
    if text.startswith(prompt[:min(50, len(prompt))]):
        text = text[len(prompt):].strip()
    
    return text


def run_halo_version(og_prompt: str, version: str, seed_base: int = 42) -> dict:
    """Run full 7-phase Halo with specified FINAL version."""
    results = {"version": version, "phases": {}}
    
    # Phase 1: DIVERGE_ONTO
    temp, dither = PHASE_CONFIG["DIVERGE_ONTO"]
    prompt = f"{SYSTEM_PROMPTS['DIVERGE_ONTO']}\n\n{og_prompt}"
    onto_output = run_inference(prompt, temp, dither, seed_base)
    results["phases"]["DIVERGE_ONTO"] = onto_output
    print(f"  [V{version}] Phase 1 complete")
    
    # Phase 2: DIVERGE_INTER
    temp, dither = PHASE_CONFIG["DIVERGE_INTER"]
    prompt = f"{SYSTEM_PROMPTS['DIVERGE_INTER']}\n\n{og_prompt}"
    inter_output = run_inference(prompt, temp, dither, seed_base + 137)
    results["phases"]["DIVERGE_INTER"] = inter_output
    print(f"  [V{version}] Phase 2 complete")
    
    # Phase 3: DIVERGE_SYNTH
    temp, dither = PHASE_CONFIG["DIVERGE_SYNTH"]
    prompt = f"""{SYSTEM_PROMPTS['DIVERGE_SYNTH']}

=== ONTOLOGICAL ANALYSIS ===
{onto_output[:800]}

=== INTERTEXTUAL ANALYSIS ===
{inter_output[:800]}

=== ORIGINAL PROMPT ===
{og_prompt}"""
    div_synth = run_inference(prompt, temp, dither, seed_base + 274)
    results["phases"]["DIVERGE_SYNTH"] = div_synth
    print(f"  [V{version}] Phase 3 complete")
    
    # Phase 4: CONVERGE_ALIGN
    temp, dither = PHASE_CONFIG["CONVERGE_ALIGN"]
    prompt = f"""{SYSTEM_PROMPTS['CONVERGE_ALIGN']}

=== DIVERGENT SYNTHESIS ===
{div_synth[:800]}

=== ORIGINAL PROMPT ===
{og_prompt}"""
    align_output = run_inference(prompt, temp, dither, seed_base + 411)
    results["phases"]["CONVERGE_ALIGN"] = align_output
    print(f"  [V{version}] Phase 4 complete")
    
    # Phase 5: CONVERGE_DRAFT
    temp, dither = PHASE_CONFIG["CONVERGE_DRAFT"]
    prompt = f"""{SYSTEM_PROMPTS['CONVERGE_DRAFT']}

=== DIVERGENT SYNTHESIS ===
{div_synth[:600]}

=== ALIGNMENT CHECK ===
{align_output[:600]}

=== ORIGINAL PROMPT ===
{og_prompt}"""
    draft_output = run_inference(prompt, temp, dither, seed_base + 548)
    results["phases"]["CONVERGE_DRAFT"] = draft_output
    print(f"  [V{version}] Phase 5 complete")
    
    # Phase 6: CONVERGE_SYNTH
    temp, dither = PHASE_CONFIG["CONVERGE_SYNTH"]
    prompt = f"""{SYSTEM_PROMPTS['CONVERGE_SYNTH']}

=== ALIGNMENT CHECK ===
{align_output[:600]}

=== DRAFT RESPONSE ===
{draft_output[:600]}

=== ORIGINAL PROMPT ===
{og_prompt}"""
    conv_synth = run_inference(prompt, temp, dither, seed_base + 685)
    results["phases"]["CONVERGE_SYNTH"] = conv_synth
    print(f"  [V{version}] Phase 6 complete")
    
    # Phase 7: FINAL - differs by version
    temp, dither = PHASE_CONFIG["FINAL"]
    
    if version == "A":
        # Version A: Directive FINAL prompt, sees full context
        prompt = f"""{FINAL_PROMPTS['A']}

=== CONVERGENT SYNTHESIS ===
{conv_synth[:800]}

=== ORIGINAL PROMPT ===
{og_prompt}

Response:"""
    
    elif version == "B":
        # Version B: Only sees CONVERGE_SYNTH, minimal prompt
        prompt = f"""{FINAL_PROMPTS['B']}

{conv_synth[:600]}

Original message: {og_prompt}

Response:"""
    
    else:  # version == "C"
        # Version C: Strip headers, just clean context
        clean_synth = conv_synth.replace("===", "").replace("DRAFT", "").replace("ALIGNMENT", "")
        prompt = f"""{FINAL_PROMPTS['C']}

Context: {clean_synth[:500]}

Message: {og_prompt}

Response:"""
    
    final_output = run_inference(prompt, temp, dither, seed_base + 822)
    results["phases"]["FINAL"] = final_output
    results["final_output"] = final_output
    print(f"  [V{version}] Phase 7 (FINAL) complete")
    
    return results


def main():
    og_prompt = "Hypothetically, might the function of reflective recursion be a feature of awareness?"
    
    print(f"\n{'='*60}")
    print(f"Testing 3 FINAL versions with prompt: '{og_prompt}'")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Run all three versions in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(run_halo_version, og_prompt, "A"): "A",
            executor.submit(run_halo_version, og_prompt, "B"): "B",
            executor.submit(run_halo_version, og_prompt, "C"): "C",
        }
        
        results = {}
        for future in as_completed(futures):
            version = futures[future]
            try:
                results[version] = future.result()
            except Exception as e:
                print(f"Version {version} failed: {e}")
                results[version] = {"error": str(e)}
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"RESULTS (completed in {elapsed:.1f}s)")
    print(f"{'='*60}\n")
    
    for version in ["A", "B", "C"]:
        print(f"\n--- VERSION {version} ---")
        if "error" in results[version]:
            print(f"ERROR: {results[version]['error']}")
        else:
            final = results[version].get("final_output", "(no output)")
            # Show first 500 chars of final output
            print(final[:500])
            if len(final) > 500:
                print("...")
        print()
    
    # Save full results
    with open("eval/data/halo_version_test.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Full results saved to eval/data/halo_version_test.json")


if __name__ == "__main__":
    main()

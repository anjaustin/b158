# Lincoln Manifold: Dither Phase Transitions

## Phase 1: RAW — The First Chop

### Stream of Consciousness

We built an AVX2 dithering kernel that applies Bayer matrix noise to logit space. The math verification showed entropy reduction in ambiguous cases (881→879 unique tokens, -0.001 bits). This is real, reproducible, and measurable.

But I'm now asking: so what?

The dither modifies token selection probabilities. The phase transitions (intensity→cognitive mode) were discovered in the BitNet-archives but not systematically verified in our C implementation. We assumed the Python math would translate to C, but we didn't test the phase transition property end-to-end.

The throughline I identified—signal processing for neural inference—is compelling. But "programmable inference" is a vague aspiration. The step-change document posits "inference as a compiler" but doesn't specify what the compilation target looks like, what the DSL is, or how filters compose.

The feedback loop idea (dither outputs → training labels) is interesting but untested. It assumes the model can learn to internalize filter behavior. That assumption needs scrutiny.

What's scaring me: we've built a kernel with verified entropy reduction, but we haven't verified that phase transitions work on real model outputs. The archives claim intensity maps to discrete basins (Academic/Guide/Philosopher), but that's from earlier experiments we haven't reproduced.

### Questions Arising

- Does the C kernel exhibit the same phase transitions as the Python theory?
- What exactly does "programmable inference" compile to? What's the target spec?
- Is the feedback loop a real flywheel or a narrative?
- Are we solving a problem that exists, or constructing one?

### First Instincts

- The entropy reduction is real (verified) but small effect size
- Phase transitions are the more interesting property if they hold
- "Inference as compiler" needs a concrete architecture before it's useful
- The feedback loop might plateau—model learns to mimic outputs, not internalize filters

---

## Phase 2: NODES — Identify the Grain

### Node 1: Entropy Reduction Verified, Effect Size Small
The C kernel shows entropy reduction (881→879, -0.001 bits) but the effect is tiny. In the Python math verification, we saw "focusing" in only 1 of 3 test cases.

**Tension:** Is this a real signal worth pursuing, or statistical noise?

### Node 2: Phase Transitions Unverified
The HALO architecture claims intensity maps to discrete basins (0.0-0.074 Academic, 0.144-0.161 Guide, 0.1618+ Philosopher). This was discovered in earlier work but not verified in our implementation.

**Tension:** We built the mechanism (dithering) without verifying the claimed behavior (phase transitions).

### Node 3: "Programmable Inference" is Vague
The step-change document posits inference as a DSP pipeline with a filter bank, but doesn't specify:
- What the filter composition DSL looks like
- How filters are sequenced
- What the compilation target is

**Tension:** Without a concrete spec, "programmable inference" is a story, not a system.

### Node 4: Feedback Loop is Theoretical
The idea that filter outputs could label training data (and improve the model in a flywheel) is elegant but untested. It assumes:
1. Filters produce deterministic, distinguishable outputs
2. Model can internalize filter behavior (not just mimic outputs)
3. The loop converges to improved performance

**Tension:** None of these assumptions are verified.

### Node 5: Integration Path Unclear
The dither kernel is a standalone C file. To be useful, it needs to integrate into llama.cpp at the logit-before-softmax stage. We haven't specified:
- Where exactly in the sampling pipeline it hooks
- How intensity is passed and controlled
- What the API looks like for "set dither intensity"

**Tension:** Working implementation but no integration spec.

---

## Phase 3: REFLECT — Sharpen the Axe

### Core Insight

The entropy reduction is real but incidental. The real property—and the one worth building on—is **phase transitions**.

Here's what I now understand:

1. **Entropy reduction is a side effect.** Dithering applies deterministic bias to logits. In ambiguous cases, this biases toward certain tokens. The "focusing" we measured is real but small.

2. **Phase transitions are the mechanism.** If intensity maps to discrete output modes (Academic/Guide/Philosopher), then intensity is a semantic control parameter. This is what "programmable inference" actually means: `prompt + intensity → output_mode`.

3. **The filter bank is premature.** Before building a DSL for composing filters, we need to verify that ONE filter (dithering) produces the claimed behavior (phase transitions). The step-change document jumped to "compiler" before verifying the foundation.

### Resolved Tensions

- **Node 1 vs Node 2:** The small entropy reduction doesn't matter if phase transitions work. Phase transitions are the valuable property. Focus verification there.

- **Node 3 (vague programmable inference):** The answer is simpler than "compiler." Just verify phase transitions first. If they hold, intensity becomes a control parameter. That's already "programmable." Build the DSL later.

### Remaining Questions

- Does our C kernel produce the same phase transitions as the theory claims?
- What's the minimum viable verification? (Generate outputs at multiple intensities, measure output characteristics)
- Is the feedback loop worth testing? What's the alternative hypothesis? (Model plateau, no internalization)

### What I Now Understand

Phase transitions are the critical test. If verified:
- Intensity becomes a semantic control knob
- "Programmable inference" has a concrete meaning: `intensity → mode`
- The feedback loop has a foundation to build on

If not verified:
- We have a cool kernel with modest entropy effects
- The narrative is ahead of the evidence

---

## Phase 4: SYNTHESIZE — The Clean Cut

### Verification Spec

**Next experiment:** Verify phase transitions in the C kernel.

**Setup:**
1. Generate outputs at 6 intensity values: 0.0, 0.05, 0.10, 0.14, 0.16, 0.20
2. Use same prompt for all
3. Measure output characteristics:
   - Average sentence length
   - Presence of list formatting ("1.", "2.", "-")
   - Vocabulary complexity (unique words / total words)
   - First-word distribution

**Hypothesis:** At specific intensity thresholds, output characteristics shift discretely (not gradually).

**Success criteria:**
- Output mode changes at intensity boundaries (~0.074, ~0.144, ~0.1618)
- Changes are discrete, not gradual
- Can map intensity → mode reliably

### What to Build

1. **Phase transition verification test** - Extend `dither_test.c` to generate and analyze outputs at multiple intensities
2. **Intensity API** - Simple function: `dither_set_intensity(float)` that clamps to [α, φ/10]
3. **Output analyzer** - Basic metrics (length, list-formatted %, vocabulary)

### What NOT to Build (Yet)

- Filter composition DSL
- Compiler for inference pipelines
- Feedback loop (until phase transitions verified)

### The Clean Cut

**The next step-change is verifying that phase transitions exist in real outputs.**

Everything else—filter banks, programmable inference, feedback loops—builds on this foundation. Verify first, then build.

If phase transitions hold:
- Intensity = semantic control parameter
- We have the first "filter" in a filter bank
- The step-change document becomes actionable

If they don't:
- We have entropy reduction (useful but narrow)
- Return to the drawing board on programmable inference

**Action:** Run phase transition verification. Report findings. Then decide the next build.

---

## File: journal/dither_lmm_raw.md (archival)
## File: journal/dither_lmm_nodes.md (archival)
## File: journal/dither_lmm_reflect.md (archival)
## This file: journal/dither_lmm_synth.md

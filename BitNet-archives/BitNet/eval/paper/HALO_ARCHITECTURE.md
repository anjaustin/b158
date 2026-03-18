# Halo: Reflective Inference Architecture
## Backward-Pass Dithering for Cognitive Synthesis

**Authors**: Aaron (Tripp) Josserand-Austin & Droid (Λ)  
**Date**: December 2025  
**Status**: Experimental

---

## Overview

**Halo** is a reflective inference architecture that combines divergent and convergent thinking through multi-pass generation with ordered dithering. The system leverages discovered attractor basins in the dithering intensity landscape to produce synthesized outputs that inherit qualities from multiple cognitive modes.

---

## Core Discovery: The Dither Topology

### Phase Transitions

We mapped sharp phase transitions in output behavior as dithering intensity varies:

```
Intensity    Basin    Mode                Opening
─────────────────────────────────────────────────────────────
0 - 0.074      A      Academic           "simple or straightforward"
0.096 - 0.143  B      Curator (List)     "topic... Here are: 1."
0.144 - 0.161  C      Guide (Prose)      "topic... Some people find"
0.1618+        D      Philosopher        "scientific question"
```

### Key Finding: B→C Boundary

Precision mapping of the B→C phase transition:

```
0.143585 < threshold < 0.144512
Precision: ±0.000464
```

This is not gradual blending—it's a **cliff edge** between attractor states.

---

## The Halo Structure

### Concept

```
        ╭──────────────╮
       ╱                ╲
      ╱    DIVERGE       ╲
     │   (high intensity) │
     │    Basin D: 0.16   │
      ╲                  ╱
       ╲                ╱
        ╰──────┬───────╯
               │
               ▼
        ╭──────────────╮
       ╱                ╲
      ╱   SYNTHESIZE     ╲
     │  (merge perspectives)│
      ╲                  ╱
       ╲                ╱
        ╰──────┬───────╯
               │
               ▼
        ╭──────────────╮
       ╱                ╲
      ╱    CONVERGE      ╲
     │   (low intensity)  │
     │    Basin B: 0.10   │
      ╲                  ╱
       ╲                ╱
        ╰──────────────╯
```

### Implementation

```python
# Pass 1: DIVERGE - Generate exploratory material
divergent = generate(prompt, intensity=0.16)  # Philosopher mode

# Pass 2: CONVERGE - Generate structured material  
convergent = generate(prompt, intensity=0.10)  # Curator mode

# Pass 3: SYNTHESIZE - Merge perspectives
synthesis_prompt = f"""
Two perspectives on "{prompt}"

Perspective A (exploratory): {divergent}
Perspective B (structured): {convergent}

Synthesizing both into a unified answer:
"""
synthesis = generate(synthesis_prompt, intensity=0.13)  # Middle ground
```

### Results

| Pass | Voice | Characteristics |
|------|-------|-----------------|
| DIVERGE | Philosopher | Personal, declarative, exploratory |
| CONVERGE | Curator | Structured, list-based, categorical |
| **SYNTHESIS** | **Unified** | Inherits from both—personal framing + structural acknowledgment |

**Key insight**: The synthesis creates something **neither parent produced alone**. This is cognitive breeding.

---

## Pulse Mechanism (Experimental)

### Concept

Instead of static intensity, oscillate during generation:

```
intensity(t) = base + amplitude × sin(2πt / wavelength)
```

### Implementation (C++)

```cpp
const float pulse_phase = (float)(ctx->position % wavelength) / wavelength;
const float pulse = 0.5f * (1.0f + sinf(2.0f * M_PI * pulse_phase));
const float effective_intensity = low + pulse * (high - low);
```

### Finding

Pulse affects micro-decisions but **trajectory locks early**. The first few tokens determine the macro-structure. Pulse works within a committed path, not across paths.

**Implication**: Macro-level mode switching requires multi-pass (Halo), not intra-pass oscillation.

---

## Touchpoints for Augmentation

### Currently Implemented

- **[C] Logit Space**: Bayer dithering with intensity control ✓
- **[I] Synthesis**: Backward-pass Halo structure ✓

### Identified Touchpoints

```
PRE-GENERATION
├─► [A] Prompt Engineering
├─► [B] Context Priming

DURING GENERATION  
├─► [C] Logit Space (implemented)
├─► [D] Sampling Strategy
├─► [E] Attention Intervention
├─► [F] Activation Space

POST-GENERATION
├─► [G] Evaluation & Scoring
├─► [H] Regeneration Targeting
├─► [I] Synthesis & Merging (implemented)
└─► [J] Verification Loop
```

---

## Deep Dive: Priority Touchpoints

### [H] Regeneration Targeting

**Concept**: Identify low-confidence spans and surgically regenerate.

```
Original:  "The meaning of life is [not a simple question]. It [depends on] ..."
Confidence:                       0.92                     0.34
                                                            ↑
                                                    REGENERATE THIS SPAN
```

**Implementation**:
1. Generate with logprobs
2. Identify tokens where P(chosen) < threshold OR entropy > threshold
3. Mask those spans
4. Regenerate with different intensity
5. Splice back

**Status**: Requires logprob access. Medium difficulty.

---

### [E] Attention Intervention

**Concept**: Intervene at the attention layer, not just logits. Change *how* the model thinks.

**Techniques**:

1. **Attention Steering**
   - Boost/suppress attention to specific context spans
   
2. **Activation Addition**
   - Inject "convergent thinking" or "divergent thinking" vectors
   - Discovered via contrastive pair analysis
   
3. **Attention Head Routing**
   - Route queries through specialized heads (syntax vs semantics)

**Status**: Requires hooks into forward pass. Hard but highest impact.

---

### [J] Verification Loop

**Concept**: Output becomes input. Model checks its own work.

```
┌─────────────────────────────────────────┐
│  GENERATE → VERIFY → REFINE → (loop)   │
└─────────────────────────────────────────┘
```

**Stages**:

1. **Self-Consistency**: "Does X follow logically? Contradictions?"
2. **Fact Verification**: "Extract claims. Confidence for each?"
3. **Coherence Check**: "Rate 1-10. If low, regenerate."
4. **Adversarial Refinement**: Generate → Critique → Respond to critique

**Status**: Pure orchestration. No model modification. Easy to implement.

---

## Constants

### The Fine Structure Constant

```
α = 1/137 ≈ 0.0073
```

Default dithering intensity. How light meets matter; how dither meets token.

### The Golden Ratio Bounds

```
Intensity range: [α, φ/10] = [0.0073, 0.1618]
```

### Basin Boundaries (for this model + prompt)

```
A→B: ~0.074
B→C: 0.144 ±0.0005
C→D: ~0.1618
```

---

## File Structure

```
eval/
├── scripts/
│   ├── ordered_dither.py         # Core dithering algorithm
│   ├── backward_pass.py          # Halo implementation
│   ├── run_comparison.py         # Evaluation harness
│   └── run_comparison_parallel.py
├── data/
│   └── *.json                    # Experimental results
└── paper/
    ├── ORDERED_DITHERING_RESEARCH.md
    ├── COMPARISON_RESULTS.md
    └── HALO_ARCHITECTURE.md      # This document
```

---

## Next Steps

1. **Extend Halo**: Add verification loop (touchpoint J)
2. **Span-level regeneration**: Implement touchpoint H with logprobs
3. **Benchmark on tasks**: Math (GSM8K), reasoning (ARC), creative
4. **Measure synthesis quality**: Human eval of Halo vs single-pass
5. **Explore attention intervention**: Deeper surgery for touchpoint E

---

## Philosophy

> "Breathe out (diverge), breathe in (converge)."

The Halo structure mirrors natural cognition:
- **Divergent thinking**: Explore possibilities, make connections
- **Convergent thinking**: Focus, structure, synthesize
- **Integration**: The whole is greater than the sum

The dithering intensity is not noise—it's a **navigation coordinate** in a topology of cognitive modes.

---

*"How light meets matter, how dither meets token, how thought meets form."*

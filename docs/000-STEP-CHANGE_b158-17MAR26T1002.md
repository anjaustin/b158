# Step-Change Analysis: Programmable Inference
**Date**: 17 March 2026, 10:02

---

## The Throughline

**Signal processing applied to neural inference**

BitNet's core innovation—ternary quantization (weights ∈ {-1, 0, +1})—creates a discrete signal space. This space can be manipulated using techniques borrowed from digital signal processing:

- **1-bit weights** = extreme quantization = "low bit-depth" image
- **Ordered dithering** = Bayer matrix applied to logits = halftoning for tokens
- **Phase transitions** = discrete cognitive basins = color banding in images

The analogy holds: just as ordered dithering lets a limited color palette appear to show more colors, it lets a quantized model appear to make finer-grained decisions.

---

## The Discovery

Dithering intensity is not just noise—it is a **semantic coordinate**:

| Intensity | Basin | Mode | Output Pattern |
|-----------|-------|------|----------------|
| 0.0-0.074 | A | Academic | "simple or straightforward" |
| 0.096-0.143 | B | Curator (List) | "Here are: 1." |
| 0.144-0.161 | C | Guide (Prose) | "Some people find" |
| 0.1618+ | D | Philosopher | "scientific question" |

These are **sharp phase transitions**, not gradual blending. A 0.001 change in intensity can flip the entire output mode.

This transforms inference from a deterministic process into a **parameterizable one**:

```
prompt → [intensity = X] → output_mode
```

---

## Current State

The existing codebase implements the first filter:

- **BitNet**: Ternary weight quantization (hardware-efficient inference)
- **Dithering**: Single filter applied to logit space
- **Halo**: Manual multi-pass orchestration (DIVERGE → SYNTHESIZE → CONVERGE)

Each is a proof of concept. None are integrated into a unified system.

---

## The Step-Change: Inference as a Compiler

**What if inference were a DSP pipeline?**

```
Prompt + Intent → Compiler → Execution Graph → Tuned Output
```

### Components

1. **Intent Classification**  
   Map natural language requests ("explain simply", "give me a list", "explore the idea") to parameter values (intensity, temperature, sampling strategy)

2. **Filter Bank** (the "touchpoints" enumerated in HALO)
   - [C] Logit Space (implemented)
   - [D] Sampling Strategy
   - [E] Attention Intervention
   - [F] Activation Space
   - [G] Evaluation & Scoring
   - [H] Regeneration Targeting
   - [I] Synthesis (implemented)
   - [J] Verification Loop

3. **Compiler**  
   Takes (prompt, intent) → outputs execution graph (which filters, in what order, with what parameters)

4. **Runtime**  
   Executes the graph, measures output characteristics, provides feedback

---

## What This Enables

| Current | Step-Change |
|---------|-------------|
| Run model with fixed parameters | Compile intent into parameter set |
| Single forward pass | Multi-pass filter pipeline |
| Output is output | Output is a processed signal with measurable properties |
| One model, one task | One model, multiple modes via filters |

---

## Next Steps

### Immediate (this codebase)

1. **Integrate dithering into main pipeline**  
   Move from `BitNet-archives/` to `BitNet/src/`

2. **Implement remaining touchpoints**  
   Priority: [H] span-level regeneration, [J] verification loop

3. **Build intent→parameters mapper**  
   Simple lookup table: "simple" → intensity 0.05, "list" → 0.10, "exploratory" → 0.16

### Medium-term

4. **Create filter composition DSL**  
   ```
   pipeline:
     - filter: dither
       intensity: {{intent.intensity}}
     - filter: regenerate
       target: low_confidence
       max_iterations: 3
   ```

3. **Benchmark filter combinations**  
   Does [C] + [H] outperform either alone?

### Long-term

6. **Self-optimizing pipeline**  
   Use gradients or reinforcement learning to discover optimal filter configurations for different task types

---

## The Philosophical Shift

> **Before**: "How do we make inference faster?"  
> **After**: "How do we make inference programmable?"

BitNet answers the first. The step-change answers the second.

---

## Related Files

| File | Role |
|------|------|
| `BitNet/README.md` | Core inference framework |
| `BitNet/src/README.md` | CPU optimizations |
| `BitNet/gpu/README.md` | GPU kernel |
| `BitNet-archives/BitNet/eval/paper/ORDERED_DITHERING_RESEARCH.md` | Dithering theory |
| `BitNet-archives/BitNet/eval/paper/HALO_ARCHITECTURE.md` | Multi-pass architecture |
| `docs/000-AUDIT_b158-17MAR26T0955.md` | Full repository audit |

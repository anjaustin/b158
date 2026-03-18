# Dithering Touchpoints for BitNet b1.58 Optimization

> Reference document cataloging all potential intervention points for dithering and signal processing enhancements.

---

## Context

BitNet b1.58 uses ternary weights {-1, 0, +1}, replacing floating-point matrix multiplications with additions/subtractions. This discrete quantization introduces artifacts analogous to low-bit-depth imaging—making dithering techniques potentially applicable.

**Goal**: Enhance the "resolution" of model thinking by applying signal processing at strategic points in the inference pipeline.

---

## Pipeline Overview

```
INPUT TOKENS
     │
     ▼
EMBEDDINGS (float32)
     │
     ▼
┌─────────────────────────────────────┐
│  TRANSFORMER LAYERS (×N)            │
│  ┌─────────────────────────────┐    │
│  │ Attention                   │    │
│  │   Q,K,V projections         │◄───┼── Ternary weights
│  │   Attention scores          │    │
│  │   Softmax                   │◄───┼── TOUCHPOINT D
│  │   Output projection         │    │
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │ FFN / MLP                   │    │
│  │   Up projection             │◄───┼── Ternary weights
│  │   Activation (SiLU/GELU)    │    │
│  │   Down projection           │◄───┼── Ternary weights
│  └─────────────────────────────┘    │
│                                     │
│  Activations quantized to int8 ◄────┼── TOUCHPOINT B
│  for next layer                     │
└─────────────────────────────────────┘
     │
     ▼
FINAL LOGITS (float32[vocab_size])  ◄── TOUCHPOINT A
     │
     ▼
SAMPLER CHAIN
     │
     ├── Temperature
     ├── Top-K
     ├── Top-P
     ├── Softmax → Probabilities    ◄── TOUCHPOINT C
     └── Sample
     │
     ▼
SELECTED TOKEN
```

---

## Touchpoint A: Logit Space (ACTIVE FOCUS)

**Location**: Post-computation, pre-sampling  
**File**: `3rdparty/llama.cpp/src/llama-sampling.cpp`  
**Insertion**: New sampler in the chain

### Mechanism

Apply ordered dithering to logits before softmax converts them to probabilities. The Bayer matrix provides spatially-coherent threshold patterns indexed by sequence position.

### Conceptual Implementation

```cpp
struct llama_sampler_ordered_dither {
    // Bayer threshold matrix (8×8 classic)
    static constexpr float bayer8[8][8] = {
        { 0/64.0f, 32/64.0f,  8/64.0f, 40/64.0f,  2/64.0f, 34/64.0f, 10/64.0f, 42/64.0f},
        {48/64.0f, 16/64.0f, 56/64.0f, 24/64.0f, 50/64.0f, 18/64.0f, 58/64.0f, 26/64.0f},
        // ... full matrix
    };
    
    float intensity;        // Dither strength [0.0 - 1.0]
    int sequence_position;  // Current position in sequence
    
    void apply(llama_token_data_array* cur_p) {
        for (size_t i = 0; i < cur_p->size; i++) {
            int bx = sequence_position % 8;
            int by = (cur_p->data[i].id) % 8;  // Token ID as second coordinate
            float threshold = bayer8[by][bx];
            
            // Apply dithering to logit
            cur_p->data[i].logit += intensity * (threshold - 0.5f);
        }
    }
};
```

### Hypothesis

- **Low-confidence decisions**: When multiple tokens have similar logits, dithering could break ties in a structured (non-random) way
- **Reasoning chains**: The ordered pattern may help maintain coherence across sequential tokens
- **Vocabulary coverage**: Controlled perturbation might increase diversity without pure randomness

### Pros
- ✅ Clean insertion (sampler chain designed for extensibility)
- ✅ No kernel modifications required
- ✅ Easy A/B testing
- ✅ Platform-independent
- ✅ Minimal performance overhead

### Cons
- ⚠️ Operates on output, not core computation
- ⚠️ Effect may be subtle at low intensities

### Implementation Complexity: LOW

### Priority: **ACTIVE**

---

## Touchpoint B: Activation Quantization

**Location**: Float → int8 conversion for layer inputs  
**File**: Upstream in ggml, before `ggml_vec_dot_i2_i8_s()`

### Mechanism

Add dithering during the quantization of activations from float to int8. This is analogous to dithering during image quantization—adding noise to break up banding artifacts.

### Conceptual Implementation

```cpp
// During activation quantization
for (int i = 0; i < n; i++) {
    float val = activations[i];
    float dither = bayer_threshold(i % 8, layer % 8) - 0.5f;
    val += dither * quantization_step;
    int8_activations[i] = clamp(round(val / scale), -128, 127);
}
```

### Hypothesis

- Reduce quantization error accumulation across layers
- Prevent "banding" in activation space
- May improve gradient flow characteristics (if training)

### Pros
- ✅ Affects core signal path
- ✅ Could compound benefits across layers

### Cons
- ⚠️ Requires kernel-level changes
- ⚠️ Platform-specific code (AVX2/ARM NEON)
- ⚠️ Harder to isolate effects

### Implementation Complexity: MEDIUM-HIGH

### Priority: FUTURE

---

## Touchpoint C: Probability Space

**Location**: Post-softmax, pre-sampling  
**File**: `llama-sampling.cpp`, after `llama_sampler_softmax_impl()`

### Mechanism

Reshape the probability distribution using ordered patterns after softmax normalization.

### Conceptual Implementation

```cpp
void apply_probability_dither(llama_token_data_array* cur_p, int pos) {
    // Redistribute probability mass using ordered pattern
    float total_adjustment = 0;
    for (size_t i = 0; i < cur_p->size; i++) {
        float threshold = bayer8[pos % 8][i % 8];
        float adjustment = intensity * (threshold - 0.5f) * cur_p->data[i].p;
        cur_p->data[i].p += adjustment;
        total_adjustment += adjustment;
    }
    // Renormalize
    // ...
}
```

### Hypothesis

- Direct control over token selection probabilities
- Could boost near-threshold candidates in structured way

### Pros
- ✅ Direct influence on selection
- ✅ Sampler chain insertion

### Cons
- ⚠️ May distort learned distributions significantly
- ⚠️ Probability constraints (must sum to 1, non-negative)
- ⚠️ Mathematically tricky to keep well-behaved

### Implementation Complexity: MEDIUM

### Priority: FUTURE (after A)

---

## Touchpoint D: Attention Scores

**Location**: Inside transformer layers, attention softmax  
**File**: Deep in llama.cpp attention computation

### Mechanism

Apply dithering to attention scores before the attention softmax, affecting how the model "focuses" on context.

### Conceptual Implementation

```cpp
// In attention computation, before softmax
for (int q = 0; q < seq_len; q++) {
    for (int k = 0; k < seq_len; k++) {
        float threshold = bayer8[q % 8][k % 8];
        attention_scores[q][k] += intensity * (threshold - 0.5f);
    }
}
// Then softmax...
```

### Hypothesis

- Affects reasoning at the source
- Could improve long-range coherence
- May help with attention "collapse" issues

### Pros
- ✅ Fundamental intervention
- ✅ Affects all downstream computation

### Cons
- ⚠️ Most invasive
- ⚠️ Hardest to debug
- ⚠️ Performance-critical path
- ⚠️ Risk of destabilizing attention

### Implementation Complexity: HIGH

### Priority: FUTURE (exploratory)

---

## Touchpoint E: Weight Access Patterns (Theoretical)

**Location**: Weight matrix traversal order  
**Concept**: Change how weights are accessed during matrix operations

### Mechanism

Use Bayer-pattern ordering when accumulating weight contributions, creating structured interference patterns.

### Status: THEORETICAL

Not currently pursued—would require fundamental kernel redesign.

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2024-XX-XX | Focus on Touchpoint A | Simplest insertion, cleanest test, sampler chain designed for this |

---

## Related Files

- `3rdparty/llama.cpp/src/llama-sampling.cpp` - Sampler implementations
- `3rdparty/llama.cpp/src/llama-sampling.h` - Sampler interfaces
- `3rdparty/llama.cpp/include/llama.h` - Public API
- `src/ggml-bitnet-mad.cpp` - BitNet kernel (Touchpoint B)
- `src/ggml-bitnet-lut.cpp` - LUT kernel variant

---

## References

1. Bayer, B.E. (1973). "An optimum method for two-level rendition of continuous-tone pictures"
2. Floyd, R.W. & Steinberg, L. (1976). "An Adaptive Algorithm for Spatial Greyscale"
3. Ulichney, R. (1987). "Digital Halftoning"

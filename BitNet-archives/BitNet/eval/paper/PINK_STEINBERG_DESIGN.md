# Pink-Steinberg: Bidirectional Error Diffusion for BitNet
## Two-Pass Refinement via Signal-Level Error Propagation

---

> **Status**: Design Phase  
> **Branch**: `pink-steinberg`  
> **Date**: December 2025

---

## Overview

Pink-Steinberg extends Floyd-Steinberg error diffusion to language model inference with a bidirectional two-pass architecture:

1. **Pass 1 (Forward)**: Generate tokens, measure selection error at probability space
2. **Pass 2 (Backward)**: Inject accumulated error signal upstream for refinement

Two variants will be implemented:
- **P1wTC-P2A**: Measure at Probability (C), Inject at Logits (A)
- **P1wTC-P2D**: Measure at Probability (C), Inject at Attention (D)

---

## Architecture

```
                    PASS 1 (Forward - Measure)
                    ═══════════════════════════
                    
Prompt ──► Transformer ──► Logits ──► Softmax ──► Sample ──► Token₁
                                         │
                                    [TOUCHPOINT C]
                                         │
                                    error₁ = measure(P)
                                         ↓
                                   error_buffer[1]
                                   
                                        ...
                                        
                                   error_buffer[T]


                    PASS 2 (Backward - Inject)
                    ═══════════════════════════

                    ┌─────────────────────────────────────┐
                    │  Variant A: Inject at Logits        │
                    └─────────────────────────────────────┘
                    
Prompt ──► Transformer ──► Logits ──────────────► Softmax ──► Sample
                              ↑
                         [TOUCHPOINT A]
                              │
                    inject(error_buffer[T-t])
                    
                    
                    ┌─────────────────────────────────────┐
                    │  Variant D: Inject at Attention     │
                    └─────────────────────────────────────┘
                    
Prompt ──► Transformer ─────────────────────────► Logits ──► Softmax ──► Sample
               │
          [TOUCHPOINT D]
               ↑
     inject(error_buffer[T-t])
     (modulates attention scores)
```

---

## Pass 1: Error Measurement at Touchpoint C

### Location
Post-softmax, pre-sampling in `llama-sampling.cpp`

### Error Metrics

```cpp
struct pink_steinberg_error {
    float entropy;        // Shannon entropy of distribution
    float margin;         // P(top1) - P(top2)  
    float confidence;     // P(selected_token)
    float regret;         // Sum of P for tokens > threshold that weren't selected
    float centroid_delta; // Distance from distribution centroid (weighted avg token id)
};
```

### Measurement Algorithm

```cpp
static pink_steinberg_error measure_selection_error(
    const llama_token_data_array * cur_p,
    llama_token selected
) {
    pink_steinberg_error err = {};
    
    // Entropy: H = -Σ p·log(p)
    for (size_t i = 0; i < cur_p->size; i++) {
        float p = cur_p->data[i].p;
        if (p > 1e-10f) {
            err.entropy -= p * logf(p);
        }
    }
    
    // Find top-1, top-2, and selected probability
    float p_top1 = 0, p_top2 = 0, p_selected = 0;
    llama_token top1_id = -1;
    
    for (size_t i = 0; i < cur_p->size; i++) {
        float p = cur_p->data[i].p;
        if (p > p_top1) {
            p_top2 = p_top1;
            p_top1 = p;
            top1_id = cur_p->data[i].id;
        } else if (p > p_top2) {
            p_top2 = p;
        }
        if (cur_p->data[i].id == selected) {
            p_selected = p;
        }
    }
    
    err.margin = p_top1 - p_top2;
    err.confidence = p_selected;
    
    // Regret: probability mass of "good" alternatives not chosen
    const float regret_threshold = 0.1f;
    for (size_t i = 0; i < cur_p->size; i++) {
        if (cur_p->data[i].id != selected && cur_p->data[i].p > regret_threshold) {
            err.regret += cur_p->data[i].p;
        }
    }
    
    // Centroid delta: how far is selected token from distribution center?
    float centroid = 0, total_p = 0;
    for (size_t i = 0; i < cur_p->size; i++) {
        centroid += cur_p->data[i].id * cur_p->data[i].p;
        total_p += cur_p->data[i].p;
    }
    centroid /= total_p;
    err.centroid_delta = fabsf(selected - centroid);
    
    return err;
}
```

### Error Buffer

```cpp
struct pink_steinberg_context {
    std::vector<pink_steinberg_error> error_buffer;
    int pass;  // 1 = forward (measure), 2 = backward (inject)
    
    // Diffusion kernel (Floyd-Steinberg classic: 7/16, 3/16, 5/16, 1/16)
    // Adapted for 1D sequence: forward neighbor gets 7/16, rest accumulates
    static constexpr float DIFFUSE_FORWARD = 7.0f / 16.0f;
    static constexpr float DIFFUSE_ACCUMULATE = 9.0f / 16.0f;
    
    float accumulated_error = 0.0f;
};
```

---

## Pass 2A: Injection at Touchpoint A (Logits)

### Location
New sampler in chain, before softmax, after temp-ext

### Injection Algorithm

```cpp
static void pink_steinberg_inject_logits(
    llama_token_data_array * cur_p,
    const pink_steinberg_error & err,
    float intensity
) {
    // Convert error metrics to logit bias
    // High entropy in Pass 1 → sharpen distribution in Pass 2
    // Low margin in Pass 1 → boost differentiation in Pass 2
    
    float sharpening = intensity * (err.entropy / 4.0f);  // Normalize by ~max entropy
    float boosting = intensity * (1.0f - err.margin);
    
    // Apply error-informed bias
    float logit_range = cur_p->data[0].logit - cur_p->data[cur_p->size-1].logit;
    
    for (size_t i = 0; i < cur_p->size; i++) {
        // Sharpen: increase gap between high and low logits
        float normalized_rank = (float)i / cur_p->size;
        float sharpen_bias = sharpening * logit_range * (0.5f - normalized_rank);
        
        // Boost based on regret signal: if we had high regret, 
        // give more weight to alternatives
        float regret_bias = 0;
        if (err.regret > 0.2f) {
            // Flatten slightly to reconsider alternatives
            regret_bias = -boosting * 0.1f * (0.5f - normalized_rank);
        }
        
        cur_p->data[i].logit += sharpen_bias + regret_bias;
    }
}
```

### Sampler Interface

```cpp
struct llama_sampler * llama_sampler_init_pink_steinberg_a(
    float intensity,
    pink_steinberg_context * ctx
);
```

---

## Pass 2D: Injection at Touchpoint D (Attention)

### Location
Inside transformer layers, modulating attention scores

### Concept

The error buffer from Pass 1 informs *what the model attends to* in Pass 2:

- High uncertainty at position t → increase attention to context around t
- Low confidence selection → broaden attention window
- High regret → shift attention to potentially missed context

### Injection Algorithm (Conceptual)

```cpp
// In attention score computation, before attention softmax
void pink_steinberg_modulate_attention(
    float * attention_scores,  // [seq_len × seq_len]
    int seq_len,
    const std::vector<pink_steinberg_error> & error_buffer,
    float intensity
) {
    for (int q = 0; q < seq_len; q++) {
        // Get error signal for this query position (reversed for backward pass)
        int err_idx = seq_len - 1 - q;
        if (err_idx >= error_buffer.size()) continue;
        
        const auto & err = error_buffer[err_idx];
        
        // High entropy → broaden attention (reduce sharpness)
        float attention_temp = 1.0f + intensity * err.entropy / 4.0f;
        
        // Apply temperature to attention scores for this query
        for (int k = 0; k < seq_len; k++) {
            attention_scores[q * seq_len + k] /= attention_temp;
        }
        
        // High regret → boost attention to recent context (might have missed something)
        if (err.regret > 0.2f) {
            for (int k = std::max(0, q - 5); k < q; k++) {
                attention_scores[q * seq_len + k] += intensity * err.regret;
            }
        }
    }
}
```

### Implementation Complexity

Touchpoint D requires modifying the attention computation in llama.cpp, which is more invasive:

```
File: 3rdparty/llama.cpp/src/llama.cpp
Function: llama_decode_internal() → attention computation
```

This will require:
1. Passing error context into the model decode
2. Hooking into attention score computation
3. Per-layer or aggregate injection strategy

---

## Implementation Plan

### Phase 1: P1wTC-P2A (Conservative)

1. **Create error measurement sampler** (Touchpoint C)
   - New sampler that measures but doesn't modify
   - Stores error in context buffer
   
2. **Create error injection sampler** (Touchpoint A)
   - New sampler that reads error buffer
   - Injects bias into logits based on backward-indexed error

3. **Create two-pass inference wrapper**
   - Manages pass state (forward/backward)
   - Clears KV cache between passes
   - Reverses error buffer indexing for Pass 2

4. **Add CLI flags**
   - `--pink-steinberg` or `--ps`
   - `--ps-intensity N`

### Phase 2: P1wTC-P2D (Aggressive)

1. **Extend llama context with error state**
   - Add error buffer to `llama_context`
   
2. **Hook attention computation**
   - Find attention score computation
   - Add error-modulated temperature/bias
   
3. **Create unified interface**
   - Same CLI, `--ps-mode [A|D]` to select variant

---

## Testing Protocol

### Comparison Matrix

| Test | Vanilla | Ordered Dither | P1wTC-P2A | P1wTC-P2D |
|------|---------|----------------|-----------|-----------|
| Reflective recursion prompt | | | | |
| Math reasoning | | | | |
| Code completion | | | | |
| Coherence decay | | | | |

### Metrics

1. **Output quality**: Human eval, task accuracy
2. **Error signal characteristics**: Entropy evolution, margin distribution
3. **Performance**: Latency overhead of two-pass
4. **Stability**: Variance across seeds

---

## Open Questions

1. Should Pass 2 use the *same* prompt or include Pass 1 output as context?
2. Error buffer: per-token or sliding window aggregate?
3. Diffusion kernel: classic F-S weights or learned/tuned?
4. Layer-wise injection for D: all layers or final N?

---

## References

1. Floyd, R.W. & Steinberg, L. (1976). "An Adaptive Algorithm for Spatial Greyscale"
2. Pink noise: 1/f spectral density, associated with natural systems and fractals
3. Ordered Dithering Research (this project): Bayer matrix application to logit space

---

*"Where the error diffuses, the signal clarifies."*

# Halo A+C Implementation Plan
## Meta-Controller with Batched Execution

**Authors**: Aaron (Tripp) Josserand-Austin & Droid (Λ)  
**Status**: COMPLETE  
**Branch**: `halo-parallel-ensemble`

---

## Revision Notes

After review, **Halo should NOT be a sampler**. Samplers operate per-token, but Halo orchestrates multiple independent generations. Making Halo a sampler creates architectural friction.

**Revised approach**: Halo is a **controller/orchestrator** that:
- Manages phase state machine
- Provides intensity values for each sample
- Builds synthesis prompts
- Collects outputs

The caller uses Halo's instructions to configure standard samplers.

---

## 1. Objective

Implement a Halo controller (`llama_halo`) that:
- Orchestrates multi-phase generation (diverge → converge → presynth → final)
- Provides intensity/prompt/limit for each generation
- Supports batched parallel generation (caller-driven)
- Falls back to sequential generation seamlessly
- Builds hierarchical synthesis prompts automatically

---

## 2. Architecture Overview (Revised)

```
┌─────────────────────────────────────────────────────────────┐
│                  llama_sampler_halo                         │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ PHASE_DIVERGE│→ │PHASE_CONVERGE│→ │PHASE_PRESYNTH│      │
│  │  4 samples   │  │  4 samples   │  │  2 synths    │      │
│  │  i=0.15-0.17 │  │  i=0.09-0.11 │  │  i=0.14/0.11 │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                              │              │
│                                              ▼              │
│                                      ┌──────────────┐      │
│                                      │ PHASE_FINAL  │      │
│                                      │  1 synthesis │      │
│                                      │  i=0.13      │      │
│                                      └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Key Design Decisions

### 3.1 Where does Halo live?

**Decision**: Halo is a *sampler* that wraps ordered-dither, not a replacement.

```cpp
llama_sampler_chain_add(chain, llama_sampler_init_halo(params));
// Internally creates/manages ordered-dither samplers
```

### 3.2 How does phase transition work?

**Decision**: Halo tracks token count per phase. When limit hit:
1. Store current output
2. Signal "phase complete" 
3. Caller must re-tokenize synthesis prompt and reset context
4. Caller calls `llama_sampler_halo_continue()` to proceed

**Rationale**: Sampler cannot modify context directly. Caller must handle prompt injection.

### 3.3 How does batching integrate?

**Decision**: Batching is handled at the *caller* level, not inside sampler.

When `use_batching=true`:
- Caller creates batch with N sequences
- Each sequence gets its own sampler chain with different dither intensity
- Halo meta-sampler coordinates which intensities to use
- Caller collects outputs and feeds back to Halo

**Rationale**: llama.cpp batching requires caller orchestration. Sampler provides parameters.

### 3.4 How are outputs collected?

**Decision**: Halo needs token→string conversion. Two options:

**Option A**: Caller passes decoded strings to Halo after each phase
```cpp
llama_sampler_halo_add_output(sampler, phase, output_string);
```

**Option B**: Halo stores token IDs, caller provides vocab for final decode
```cpp
llama_sampler_halo_get_synthesis_prompt(sampler, vocab);
```

**Chosen**: Option A - simpler, caller already has decoded output.

---

## 4. Data Structures

```cpp
// In llama.h

#define LLAMA_HALO_MAX_SAMPLES 8

enum llama_halo_phase {
    LLAMA_HALO_PHASE_DIVERGE,
    LLAMA_HALO_PHASE_CONVERGE,
    LLAMA_HALO_PHASE_PRESYNTH_DIV,
    LLAMA_HALO_PHASE_PRESYNTH_CONV,
    LLAMA_HALO_PHASE_FINAL,
    LLAMA_HALO_PHASE_COMPLETE,
};

struct llama_halo_params {
    int32_t n_diverge;           // Number of divergent samples (default: 4)
    int32_t n_converge;          // Number of convergent samples (default: 4)
    int32_t tokens_sample;       // Tokens per sample (default: 72)
    int32_t tokens_presynth;     // Tokens for pre-synthesis (default: 96)
    int32_t tokens_final;        // Tokens for final synthesis (default: 144)
    
    float intensity_div_min;     // Divergent intensity min (default: 0.15)
    float intensity_div_max;     // Divergent intensity max (default: 0.17)
    float intensity_conv_min;    // Convergent intensity min (default: 0.09)
    float intensity_conv_max;    // Convergent intensity max (default: 0.11)
    float intensity_presynth_div;  // Pre-synth divergent (default: 0.14)
    float intensity_presynth_conv; // Pre-synth convergent (default: 0.11)
    float intensity_final;       // Final synthesis (default: 0.13)
    
    uint32_t seed_base;          // Base seed for reproducibility
};

// Default params
struct llama_halo_params llama_halo_default_params(void);
```

```cpp
// In llama-sampling.cpp (internal)

struct llama_sampler_halo {
    llama_halo_params params;
    
    // Current state
    llama_halo_phase phase;
    int32_t current_sample;      // Which sample in current phase
    int32_t tokens_generated;    // Tokens generated in current generation
    
    // Current intensity (set based on phase + sample index)
    float current_intensity;
    
    // Collected outputs (caller provides via add_output)
    std::vector<std::string> diverge_outputs;
    std::vector<std::string> converge_outputs;
    std::string presynth_div;
    std::string presynth_conv;
    std::string final_output;
    
    // The internal ordered-dither sampler (intensity updated per phase)
    llama_sampler * dither_sampler;
};
```

---

## 5. API Functions

```cpp
// Initialize with params
LLAMA_API struct llama_sampler * llama_sampler_init_halo(
    struct llama_halo_params params
);

// Get default params
LLAMA_API struct llama_halo_params llama_halo_default_params(void);

// Get current phase
LLAMA_API enum llama_halo_phase llama_sampler_halo_get_phase(
    const struct llama_sampler * smpl
);

// Get intensity for current phase/sample (for batched execution)
LLAMA_API float llama_sampler_halo_get_intensity(
    const struct llama_sampler * smpl,
    int sample_index  // 0 to n_diverge-1 or n_converge-1
);

// Get token limit for current phase
LLAMA_API int32_t llama_sampler_halo_get_token_limit(
    const struct llama_sampler * smpl
);

// Add completed output for current phase
LLAMA_API void llama_sampler_halo_add_output(
    struct llama_sampler * smpl,
    const char * output
);

// Advance to next sample or phase (call after adding output)
LLAMA_API void llama_sampler_halo_advance(
    struct llama_sampler * smpl
);

// Get synthesis prompt for current phase (pre-synth or final)
// Returns NULL if current phase doesn't need a new prompt
LLAMA_API const char * llama_sampler_halo_get_synthesis_prompt(
    struct llama_sampler * smpl,
    const char * original_prompt
);

// Check if complete
LLAMA_API bool llama_sampler_halo_is_complete(
    const struct llama_sampler * smpl
);

// Get final result (only valid after PHASE_COMPLETE)
LLAMA_API const char * llama_sampler_halo_get_result(
    const struct llama_sampler * smpl
);
```

---

## 6. Phase Transition Logic

```
PHASE_DIVERGE:
  - Generate n_diverge samples
  - Each sample: tokens_sample tokens
  - Intensities: evenly spaced [intensity_div_min, intensity_div_max]
  - After each: add_output(), advance()
  - When current_sample == n_diverge: → PHASE_CONVERGE

PHASE_CONVERGE:
  - Generate n_converge samples
  - Each sample: tokens_sample tokens
  - Intensities: evenly spaced [intensity_conv_min, intensity_conv_max]
  - After each: add_output(), advance()
  - When current_sample == n_converge: → PHASE_PRESYNTH_DIV

PHASE_PRESYNTH_DIV:
  - get_synthesis_prompt() returns divergent pre-synth prompt
  - Generate 1 sample: tokens_presynth tokens
  - Intensity: intensity_presynth_div
  - After: add_output(), advance() → PHASE_PRESYNTH_CONV

PHASE_PRESYNTH_CONV:
  - get_synthesis_prompt() returns convergent pre-synth prompt
  - Generate 1 sample: tokens_presynth tokens
  - Intensity: intensity_presynth_conv
  - After: add_output(), advance() → PHASE_FINAL

PHASE_FINAL:
  - get_synthesis_prompt() returns final synthesis prompt
  - Generate 1 sample: tokens_final tokens
  - Intensity: intensity_final
  - After: add_output(), advance() → PHASE_COMPLETE

PHASE_COMPLETE:
  - is_complete() returns true
  - get_result() returns final_output
```

---

## 7. Synthesis Prompt Templates

### Pre-Synth Divergent
```
Synthesize these exploratory perspectives into one coherent view:

[D1 i=0.15]: {diverge_outputs[0][:180]}
[D2 i=0.16]: {diverge_outputs[1][:180]}
[D3 i=0.17]: {diverge_outputs[2][:180]}
[D4 i=0.16]: {diverge_outputs[3][:180]}

Unified exploratory synthesis:
```

### Pre-Synth Convergent
```
Synthesize these structured perspectives into one coherent view:

[C1 i=0.09]: {converge_outputs[0][:180]}
[C2 i=0.10]: {converge_outputs[1][:180]}
[C3 i=0.11]: {converge_outputs[2][:180]}
[C4 i=0.10]: {converge_outputs[3][:180]}

Unified structured synthesis:
```

### Final Synthesis
```
Original question: "{original_prompt}"

=== EXPLORATORY BRANCH ===
Pre-synthesis: {presynth_div[:300]}

=== STRUCTURED BRANCH ===
Pre-synthesis: {presynth_conv[:300]}

=== FINAL SYNTHESIS ===
Integrating all perspectives into a comprehensive, unified answer:
```

---

## 8. Implementation Steps

### Step 1: Add to llama.h
- [ ] Add llama_halo_phase enum
- [ ] Add llama_halo_params struct
- [ ] Add llama_halo_default_params()
- [ ] Add llama_sampler_init_halo()
- [ ] Add phase query functions
- [ ] Add output management functions
- [ ] Add synthesis prompt function

### Step 2: Implement in llama-sampling.cpp
- [ ] Add llama_sampler_halo struct
- [ ] Implement sampler interface (name, apply, accept, reset, clone, free)
- [ ] Implement phase transition logic
- [ ] Implement intensity calculation per sample
- [ ] Implement synthesis prompt generation
- [ ] Implement output collection

### Step 3: Create test harness
- [ ] Create halo_test.cpp or Python wrapper
- [ ] Test single-phase generation
- [ ] Test phase transitions
- [ ] Test synthesis prompt generation
- [ ] Test full pipeline

### Step 4: Integration
- [ ] Modify llama-cli to support --halo flag
- [ ] Test with BitNet model
- [ ] Benchmark vs Python implementation

---

## 9. Test Cases

1. **Default params**: Verify defaults are sane
2. **Phase progression**: DIVERGE → CONVERGE → PRESYNTH → FINAL → COMPLETE
3. **Intensity spread**: Verify even distribution within ranges
4. **Token limits**: Verify generation stops at limit
5. **Output collection**: Verify all outputs stored correctly
6. **Synthesis prompts**: Verify correct prompt generation
7. **Full pipeline**: End-to-end test with real model

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Context reset between phases | Document that caller must handle |
| Memory for stored outputs | Set max output length, truncate |
| Synthesis prompt too long | Truncate individual outputs |
| Batching complexity | Start with sequential, add batching later |

---

## 11. Success Criteria

- [ ] Halo sampler compiles without errors
- [ ] Phase transitions work correctly
- [ ] Synthesis prompts generated correctly
- [ ] Full pipeline produces coherent output
- [ ] Output quality matches Python implementation
- [ ] Performance: <2x overhead vs single generation

---

## 12. Files to Modify

1. `3rdparty/llama.cpp/include/llama.h` - API declarations
2. `3rdparty/llama.cpp/src/llama-sampling.cpp` - Implementation
3. `3rdparty/llama.cpp/src/llama-sampling.h` - Internal declarations (if needed)
4. `3rdparty/llama.cpp/examples/main/main.cpp` - CLI integration (optional)

---

## Next Steps

1. Review this plan for completeness
2. Identify any gaps or issues
3. Begin implementation in llama-sampling.cpp
4. Test incrementally
5. Celebrate with high-five 🖐️

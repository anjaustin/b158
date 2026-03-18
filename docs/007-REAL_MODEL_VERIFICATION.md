# Real Model Verification: Path Forward

## Current Status

### Verified (Synthetic)
- ✅ Dither mechanism works in E2E simulation
- ✅ Focusing effect measurable (13.0 → 1.8 unique tokens)
- ✅ Position-dependency is feature, not bug

### Unverified (Real Model)
- ❌ Does dither improve coherence on actual model outputs?
- ❌ What's the optimal intensity for reasoning tasks?
- ❌ Does focused output = better reasoning chain quality?

## Path to Real Verification

### Option 1: Integrate into llama.cpp
Location: `BitNet/3rdparty/llama.cpp/`

Where to hook:
1. Find `sample` function in llama.cpp
2. Add dither call before softmax
3. Pass intensity via CLI flag `--dither-intensity`

### Option 2: Use Python wrapper
The llama.cpp has Python bindings. Could test via:
```python
from llama_cpp import Llama
model = Llama(model_path)
# Apply dither to logits before sampling
```

### Option 3: Quick test with existing binary
Add `--dither` flag to existing llama-cli and test generation.

## What We Learned

The coherence test showed minimal effect in our simulation because:
1. Simulation doesn't capture real model behavior
2. Logit distributions are artificial
3. Need real token probabilities

## Recommendation

**Next step**: Integrate dither into llama.cpp at the sample hook point.

The integration is straightforward:
```c
// In llama.cpp sampling code:
if (params.dither_intensity > 0.0f) {
    dither_apply_avx(logits, vocab_size, position, params.dither_intensity);
}
```

## Committed Artifacts

- `test_coherence.py` - Coherence test framework
- `dither/` - Full implementation
- `docs/` - Analysis and documentation

## What's Left

1. Real model integration (requires llama.cpp changes)
2. Benchmark on reasoning tasks
3. Determine optimal intensity for different task types

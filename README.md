# b158: Ordered Dithering for LLM Inference

## Summary

This project implements ordered dithering as a sampling technique for LLM inference. Dithering applies position-dependent noise to logits to reduce entropy and focus token selection.

## Key Findings

1. **Entropy reduction verified** - Dither produces measurable entropy reduction in token probability distributions
2. **Phase transitions falsified** - Intensity does not map to discrete cognitive modes (Academic/Guide/Philosopher)
3. **Position-dependency is a feature** - In generation contexts (where position increments predictably), dither creates stable focusing behavior

## Project Structure

```
/home/ztflynn/001/000/randy/b158/
├── dither/                    # C AVX2 implementation
│   ├── dither_avx.c          # Optimized dither kernel
│   ├── main.c                # Test harness
│   └── Makefile              
├── BitNet/                    # Inference framework (llama.cpp)
│   └── 3rdparty/llama.cpp/   # Modified with dither sampler
│       ├── src/llama-sampling.cpp    # Dither implementation
│       ├── include/llama.h           # API declaration
│       ├── common/arg.cpp            # CLI flag
│       └── common/sampling.cpp       # Sampling chain integration
├── docs/                      # Research documentation
│   ├── 002-LMM_DITHER_SYNTH.md       # LMM analysis
│   ├── 003-PHASE_TRANSITION_VERIFICATION.md
│   ├── 005-AUTOPHY_FINDINGS.md       # Root cause analysis
│   └── 008-DITHER_CLI_INTEGRATION.md # Usage docs
└── test_dither_math.py        # Python math verification
```

## Usage

```bash
# Build
cd BitNet/build && make llama-cli -j4

# Run with dither
./bin/llama-cli -m models/BitNet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf \
    -p "The future of AI is" --dither 1.0 -n 50
```

## CLI Options

| Flag | Description |
|------|-------------|
| `--dither N` | Dithering intensity (0.0 = disabled) |

## Research Notes

See `/home/ztflynn/001/000/randy/b158/docs/` for detailed research logs:
- LMM (Lincoln Manifold Method) analysis
- Phase transition verification (falsified)
- Root cause analysis of position-dependency
- E2E verification with real model

## Model

- **BitNet-b1.58-2B-4T** - 2B parameter, 1.58-bit quantization
- Downloaded from HuggingFace
- Path: `BitNet/models/BitNet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf`

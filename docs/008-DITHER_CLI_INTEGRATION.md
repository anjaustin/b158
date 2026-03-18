# Dither Sampler: CLI Integration

## Overview

Ordered dithering is now integrated into the llama.cpp sampling pipeline as a first-class sampler. It applies position-dependent noise to logits to reduce entropy and focus sampling on top tokens.

## CLI Usage

```bash
./llama-cli -m <model> -p "prompt" --dither <intensity>
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dither N` | float | 0.0 | Dithering intensity. 0.0 = disabled, higher = more entropy reduction |

### Example

```bash
# Without dither (default behavior)
./llama-cli -m model.gguf -p "Hello world" -n 50

# With dither intensity 1.0
./llama-cli -m model.gguf -p "Hello world" -n 50 --dither 1.0

# With dither intensity 5.0 (stronger focusing)
./llama-cli -m model.gguf -p "Hello world" -n 50 --dither 5.0
```

## Files Modified

### llama.cpp/src/llama-sampling.cpp
- Added `llama_sampler_dither` struct with `intensity` and `position` fields
- Added `llama_sampler_dither_apply()` - applies Bayer matrix dither to logits
- Added `llama_sampler_init_dither(float intensity)` - factory function

### llama.cpp/include/llama.h
- Added `LLAMA_API struct llama_sampler * llama_sampler_init_dither(float intensity)`

### llama.cpp/common/common.h
- Added `float dither = 0.0f` to `common_sampler_params`

### llama.cpp/common/arg.cpp
- Added `--dither N` CLI argument

### llama.cpp/common/sampling.cpp
- Added dither sampler to chain when `params.dither > 0.0f`

## Algorithm

The dither uses a 4x4 Bayer matrix:

```
0/16  8/16  2/16  10/16
12/16 4/16 14/16  6/16
3/16 11/16  1/16  9/16
15/16 7/16 13/16  5/16
```

For each token at position `pos` with index `i` in the probability array:
1. Calculate matrix coordinates: `x = (pos + i) % 4`, `y = (pos + i) / 4`
2. Get Bayer value: `bayer[x][y]`
3. Apply: `logit += (bayer_value - 0.5) * intensity`
4. Increment position counter

## Position Dependency

The dither is **position-dependent** - the bias pattern shifts with each token generated. This is:
- **A feature** in generation contexts (predictable position increments)
- **A limitation** in batch/simultaneous contexts (position not meaningful)

This is documented in `006-DITHER_FIX_RESOLUTION.md`.

## Verification

```bash
# Compare sampler chain with and without dither
./llama-cli -m model.gguf -p "test" 2>&1 | grep "sampler chain"
# Output: ... -> min-p -> temp-ext -> softmax -> dist

./llama-cli -m model.gguf -p "test" --dither 1.0 2>&1 | grep "sampler chain"  
# Output: ... -> min-p -> temp-ext -> dither -> softmax -> dist
```

## Integration Date

- Completed: March 17, 2026
- Build: BitNet/build/bin/llama-cli

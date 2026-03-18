# BBDOS x b158: The Integrated Vision (Glyph)

## Found Missing Pieces in `../bbdos/`

The `bbdos` repository contains several critical components that are currently absent or only partially integrated in the `b158` (Ordered Dithering) repository. These components form the **L-Cache architecture** and the **APU (Accelerator Processing Unit)** core.

### 1. L-Cache Infrastructure (`bbdos/two-be/research/`)
*   **`ggml-lcache.c/h`**: A generic content-based caching system for GGML tensors. It uses FNV-1a hashing to identify input states and bypass computation for repeated prompts or states.
*   **`bbdos_lcache_shim.c/h`**: The bridge between `llama.cpp` and the L-Cache. It specifically targets FFN and Attention layer outputs.

### 2. APU Optimized Opcodes (`bbdos/two-be/bbdos/kernel/`)
*   **`bbdos_apu.c`**: Pure C/AVX implementation of common transformer opcodes:
    *   `RMSNorm`
    *   `SiLU` / `GELU`
    *   `Softmax`
    *   `BitAttention` / `BitLinear`
*   **`mtfp_avx.cpp`**: Multi-Trit Floating Point (MTFP) arithmetic. This is the hardware-efficient way to handle the {-1, 0, 1} states used in 1.58-bit models.

### 3. BitSwitch Logic (`bbdos/two-be/bbdos/kernel/bitswitch_avx.cpp`)
*   **Learned Sparsity**: The ability to "physically skip" computation of tiles based on a gating network. This is the core of the BBDOS speedup (measured at >4x for 75% sparsity).

## The Integration Gap (Glyph)

The current `b158` repository has a novel **Ordered Dithering Sampler** and **BitNet 1.58-bit kernels** (spm-sources), but it lacks the BBDOS "intelligence" (gating and caching).

### Proposed Unified Architecture (Glyph-b158):

1.  **Input**: Token + Position
2.  **L-Cache Check**: Use `ggml-lcache` to see if this state has been computed before.
3.  **Kernel Compute**: If cache miss, use the **BitNet LUT/MAD kernels** (from `b158`) augmented by **BitSwitch gating** (from `bbdos`).
4.  **Sampling**: Apply the **Ordered Dithering Sampler** (from `b158`) using **AVX-optimized Bayer matrices**.
5.  **Store**: Cache the result in **L-Cache**.

---
*Date: 2026-03-18*
*Discovery by: opencode (gemini-3-flash-preview)*

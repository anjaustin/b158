# Repository Audit Summary: b158 (Ordered Dithering for LLM Inference)

## 1. Novel Components
*   **Ordered Dithering Sampler**: Implements a Bayer matrix dithering technique (traditionally for image processing) as a logit sampling method for LLMs. This uses position-dependent thresholds to bias token selection, offering a unique alternative to Top-P/K.
*   **Ternary (1.58-bit) Kernels**: Integration of `GGML_TYPE_TL1` and `GGML_TYPE_TL2` into the `llama.cpp` core. These types use Look-Up Table (LUT) and Multiply-Add (MAD) logic specifically for ternary weights ({-1, 0, 1}), optimized for BitNet architectures.

## 2. Practical Utility
*   **AVX2-Optimized Sampler**: Production-ready C implementation (`dither/dither_avx.c`) for x86 CPUs, ensuring low-latency sampling.
*   **NVIDIA Jetson Support**: Deployment-ready Docker and build configurations in `Glyph-b158/` for ARM-based edge hardware.
*   **Entropy Verification**: Tooling to empirically measure how dithering "focuses" model distributions by reducing entropy.

## 3. Core Insights (Understated)
*   **Predictable Focusing**: In sequential generation, the incrementing position index makes the dither pattern deterministic and stable, acting as a "focusing" mechanism rather than just random noise.
*   **Hardware-Mapped Ternary logic**: The TL1/TL2 types bridge the gap between theoretical BitNet efficiency and real-world CPU instruction sets.

## 4. Legacy/Artifacts (Fluff)
*   **Falsified "Phase Transition" Theory**: Documentation remains for the hypothesis that dither intensity maps to "cognitive modes" (e.g., Academic/Philosopher), which has since been falsified.
*   **Repository Bloat**: Multiple nested versions of `BitNet` (llama.cpp fork) and `BitNet-archives` represent development history rather than current functionality.
*   **Academic Jargon**: Terms like "LMM" (Lincoln Manifold Method) are used for what is fundamentally logit dithering.

---
*Date: 2026-03-18*
*Audit by: opencode (gemini-3-flash-preview)*

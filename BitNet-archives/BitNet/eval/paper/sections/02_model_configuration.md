# 2. Model Configuration

## 2.1 Base Model

| Property | Value |
|----------|-------|
| **Model** | BitNet-b1.58-2B-4T |
| **Parameters** | 2.4 Billion |
| **Architecture** | Transformer (1.58-bit weights) |
| **Weight Values** | {-1, 0, +1} |
| **Training Tokens** | 4 Trillion |
| **Source** | microsoft/BitNet-b1.58-2B-4T |
| **Format** | GGUF (ggml-model-i2_s.gguf) |
| **Kernel** | I2_S |

## 2.2 Inference Settings (Control Configuration)

These settings remain **constant across all experiments** to isolate the effects of optimization algorithms:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **top_k** | 42 | Balanced vocabulary exploration |
| **top_p** | 0.963 | Near-full nucleus with tail truncation |
| **temperature** | 0.528 | Reduced randomness for reproducibility |
| **context_window** | 4096 | Full context capacity |
| **tokens_to_predict** | 1536 | Sufficient for long-form evaluation |
| **threads** | System-dependent | Matched across runs |
| **GPU layers** | 0 | Pure CPU inference |
| **batch_size** | 1 | Sequential token generation |

## 2.3 CLI Command Template

```bash
./build/bin/llama-cli \
    -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
    --top-k 42 \
    --top-p 0.963 \
    --temp 0.528 \
    -c 4096 \
    -n 1536 \
    -ngl 0 \
    -b 1 \
    -p "<prompt>"
```

## 2.4 Hardware Specification

<!-- UPDATE: Fill in actual hardware used -->

| Component | Specification |
|-----------|---------------|
| CPU | TBD |
| Cores/Threads | TBD |
| RAM | TBD |
| OS | Linux (kernel TBD) |

## 2.5 Software Environment

| Dependency | Version |
|------------|---------|
| bitnet.cpp | commit: `404980e` |
| Python | 3.9.x |
| CMake | ≥ 3.22 |
| Clang | ≥ 18 |

---

<!-- LINKED EXPERIMENTS -->
<!-- exp: baseline -->

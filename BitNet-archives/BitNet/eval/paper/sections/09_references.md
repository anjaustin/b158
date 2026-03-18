# 9. References & Appendices

## 9.1 Core References

### BitNet Architecture

1. Wang, H., Ma, S., Dong, L., Huang, S., Wang, H., Ma, L., Yang, F., Wang, R., Wu, Y., & Wei, F. (2023). **BitNet: Scaling 1-bit Transformers for Large Language Models.** *arXiv preprint arXiv:2310.11453*.

2. Ma, S., Wang, H., Ma, L., Wang, L., Wang, W., Huang, S., Dong, L., Wang, R., Xue, J., & Wei, F. (2024). **The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.** *arXiv preprint arXiv:2402.17764*.

3. Microsoft Research. (2024). **1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs.** *arXiv preprint arXiv:2410.16144*.

4. Microsoft Research. (2025). **Bitnet.cpp: Efficient Edge Inference for Ternary LLMs.** *arXiv preprint arXiv:2502.11880*.

### Quantization & Low-Bit Models

5. Ma, S., et al. (2024). **BitNet a4.8: 4-bit Activations for 1-bit LLMs.** *arXiv preprint arXiv:2411.04965*.

6. Frantar, E., et al. (2022). **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.** *arXiv preprint arXiv:2210.17323*.

### Dithering & Signal Processing

7. Roberts, L. G. (1962). **Picture Coding Using Pseudo-Random Noise.** *IRE Transactions on Information Theory*.

8. Wannamaker, R. A., Lipshitz, S. P., & Vanderkooy, J. (2000). **A Theory of Nonsubtractive Dither.** *IEEE Transactions on Signal Processing*.

---

## 9.2 Benchmark References

### Standard Evaluations

- **MMLU**: Hendrycks, D., et al. (2021). Measuring Massive Multitask Language Understanding.
- **HellaSwag**: Zellers, R., et al. (2019). HellaSwag: Can a Machine Really Finish Your Sentence?
- **ARC**: Clark, P., et al. (2018). Think you have Solved Question Answering?
- **WinoGrande**: Sakaguchi, K., et al. (2020). WinoGrande: An Adversarial Winograd Schema Challenge.

### Mathematics

- **GSM8K**: Cobbe, K., et al. (2021). Training Verifiers to Solve Math Word Problems.
- **MATH**: Hendrycks, D., et al. (2021). Measuring Mathematical Problem Solving.

### Science

- **GPQA**: Rein, D., et al. (2023). GPQA: A Graduate-Level Google-Proof Q&A Benchmark.
- **SciQ**: Welbl, J., et al. (2017). Crowdsourcing Multiple Choice Science Questions.

---

## Appendix A: Benchmark Dataset Sources

| Benchmark | HuggingFace Dataset ID | Split | Size |
|-----------|------------------------|-------|------|
| WikiText-2 | `wikitext/wikitext-2-raw-v1` | test | - |
| MMLU | `cais/mmlu` | test | 14,042 |
| HellaSwag | `Rowan/hellaswag` | validation | 10,042 |
| ARC-Challenge | `allenai/ai2_arc/ARC-Challenge` | test | 1,172 |
| WinoGrande | `allenai/winogrande/winogrande_xl` | validation | 1,767 |
| GSM8K | `openai/gsm8k/main` | test | 1,319 |
| MATH | `lighteval/MATH` | test | 5,000 |
| GPQA | `Idavidrein/gpqa/gpqa_diamond` | train | 198 |
| SciQ | `allenai/sciq` | test | 1,000 |

---

## Appendix B: Difficulty Stratification Methods

### B.1 Explicit Labels

| Benchmark | Field | Values |
|-----------|-------|--------|
| MATH | `level` | 1, 2, 3, 4, 5 |
| ARC | Split | Easy, Challenge |

### B.2 Proxy Metrics

| Benchmark | Proxy | Calculation |
|-----------|-------|-------------|
| GSM8K | Steps | Count `\n` in solution |
| MMLU | Historical accuracy | Per-question accuracy from prior studies |
| HellaSwag | Adversarial score | Provided in dataset |

### B.3 Empirical (Baseline-Derived)

```python
def compute_difficulty(model_outputs, questions):
    """
    Rank questions by model confidence.
    Low confidence = Hard, High confidence = Easy
    """
    difficulties = []
    for q, output in zip(questions, model_outputs):
        logit_margin = output.top_logit - output.second_logit
        difficulties.append((q.id, logit_margin))
    
    ranked = sorted(difficulties, key=lambda x: x[1])
    
    n = len(ranked)
    easy = [q for q, _ in ranked[int(0.9*n):]]   # Top 10%
    hard = [q for q, _ in ranked[:int(0.1*n)]]   # Bottom 10%
    
    return easy, hard
```

---

## Appendix C: Statistical Methods

### C.1 Paired t-test

For comparing baseline vs optimized on same samples:

```
t = (mean(O) - mean(B)) / (std(O-B) / sqrt(n))
df = n - 1
```

### C.2 Cohen's d (Effect Size)

```
d = (mean(O) - mean(B)) / pooled_std
pooled_std = sqrt((std(O)² + std(B)²) / 2)
```

### C.3 Expected Calibration Error

```
ECE = Σ (n_bin / N) × |accuracy(bin) - confidence(bin)|
```

### C.4 Bonferroni Correction

For m comparisons:
```
α_corrected = α / m
```

---

## Appendix D: Hardware Specifications Template

```yaml
hardware:
  system:
    manufacturer: <string>
    model: <string>
  
  cpu:
    model: <string>
    architecture: <x86_64|arm64>
    cores_physical: <int>
    cores_logical: <int>
    base_clock_ghz: <float>
    boost_clock_ghz: <float>
    cache_l3_mb: <int>
  
  memory:
    total_gb: <int>
    type: <DDR4|DDR5>
    speed_mhz: <int>
  
  storage:
    type: <SSD|NVMe|HDD>
    model: <string>
  
  os:
    name: <string>
    version: <string>
    kernel: <string>
```

---

## Appendix E: Glossary

| Term | Definition |
|------|------------|
| **1.58-bit** | Ternary quantization using {-1, 0, +1} weights |
| **BitNet** | Neural network architecture with binary/ternary weights |
| **Dithering** | Adding noise to improve quantization quality |
| **ECE** | Expected Calibration Error |
| **GGUF** | GPT-Generated Unified Format (model file format) |
| **Logit** | Raw model output before softmax |
| **Perplexity** | Exponential of cross-entropy loss |
| **Top-k** | Sampling from k highest probability tokens |
| **Top-p** | Nucleus sampling from cumulative probability p |

---

<!-- LINKED EXPERIMENTS -->
<!-- exp: -->

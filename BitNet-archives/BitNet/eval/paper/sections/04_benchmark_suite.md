# 4. Benchmark Suite

## 4.1 Tier 1: Standard ML Evaluations

### Perplexity (WikiText-2)

| Property | Value |
|----------|-------|
| **Metric** | Perplexity (↓ lower is better) |
| **Purpose** | Fundamental language modeling quality |
| **Sensitivity** | Highly sensitive to probability distribution changes |
| **Dataset** | `wikitext-2-raw-v1` |
| **Split** | test |

### MMLU (Massive Multitask Language Understanding)

| Property | Value |
|----------|-------|
| **Metric** | Accuracy |
| **Purpose** | Broad knowledge and reasoning |
| **Subjects** | 57 (STEM, humanities, social sciences, professional) |
| **Dataset** | `cais/mmlu` |
| **Split** | test |

### HellaSwag

| Property | Value |
|----------|-------|
| **Metric** | Accuracy |
| **Purpose** | Commonsense reasoning via sentence completion |
| **Challenge** | Adversarially filtered for difficulty |
| **Dataset** | `Rowan/hellaswag` |
| **Split** | validation |

### ARC-Challenge

| Property | Value |
|----------|-------|
| **Metric** | Accuracy |
| **Purpose** | Grade-school science reasoning |
| **Challenge** | Requires multi-step inference |
| **Dataset** | `allenai/ai2_arc` (ARC-Challenge) |
| **Split** | test |

### WinoGrande

| Property | Value |
|----------|-------|
| **Metric** | Accuracy |
| **Purpose** | Commonsense pronoun resolution |
| **Challenge** | Requires world knowledge |
| **Dataset** | `allenai/winogrande` |
| **Split** | validation |

---

## 4.2 Tier 2: Mathematics

### GSM8K (Grade School Math 8K)

| Property | Value |
|----------|-------|
| **Metric** | Accuracy (exact match on final answer) |
| **Purpose** | Multi-step arithmetic word problems |
| **Difficulty Proxy** | Solution length (steps required) |
| **Dataset** | `openai/gsm8k` |
| **Split** | test |

**Difficulty Examples**:
- *Easy (2-3 steps)*: Single-digit operations, direct calculation
- *Hard (6+ steps)*: Multi-digit with unit conversions, compound reasoning

### MATH (Competition Mathematics)

| Property | Value |
|----------|-------|
| **Metric** | Accuracy |
| **Purpose** | Competition-level mathematical reasoning |
| **Difficulty** | Explicit levels 1-5 |
| **Dataset** | `lighteval/MATH` |
| **Split** | test |

**Subjects**:
- Prealgebra
- Algebra  
- Number Theory
- Counting & Probability
- Geometry
- Intermediate Algebra
- Precalculus

### MathQA

| Property | Value |
|----------|-------|
| **Metric** | Accuracy |
| **Purpose** | GRE-style quantitative reasoning |
| **Challenge** | Real-world mathematical application |
| **Dataset** | `math_qa` |

---

## 4.3 Tier 3: Physics & Science

### GPQA (Graduate-Level Google-Proof Q&A)

| Property | Value |
|----------|-------|
| **Metric** | Accuracy |
| **Purpose** | Expert-level scientific reasoning |
| **Domains** | Physics, Chemistry, Biology |
| **Challenge** | Difficult even for experts with internet |
| **Dataset** | `Idavidrein/gpqa` (gpqa_diamond) |
| **Note** | Ceiling test—improvements here are significant |

### SciQ

| Property | Value |
|----------|-------|
| **Metric** | Accuracy |
| **Purpose** | High school science comprehension |
| **Domains** | Physics, Chemistry, Biology, Earth Science |
| **Dataset** | `allenai/sciq` |
| **Split** | test |

### MMLU-Physics (Subset)

| Property | Value |
|----------|-------|
| **Metric** | Accuracy |
| **Purpose** | Undergraduate physics concepts |
| **Dataset** | `cais/mmlu` (filtered) |

**Extracted Subjects**:
- Conceptual Physics
- High School Physics
- College Physics
- Astronomy
- Electrical Engineering (applied physics)

---

## 4.4 Tier 4: Real-World Scenarios

| Scenario | Dataset | Key Metric |
|----------|---------|------------|
| **Code Generation** | HumanEval / MBPP subset | pass@1 |
| **Summarization** | CNN/DailyMail subset | ROUGE-L + coherence |
| **Multi-turn Q&A** | Custom dialogue chains | Context retention |
| **Instruction Following** | IFEval subset | Format compliance |
| **Long-form Output** | 1000+ token generations | Coherence decay |

---

## 4.5 Benchmark Summary Table

| Benchmark | Tier | Metric | Difficulty Source | Dataset Size |
|-----------|------|--------|-------------------|--------------|
| WikiText-2 | 1 | Perplexity | N/A | - |
| MMLU | 1 | Accuracy | Historical | 14,042 |
| HellaSwag | 1 | Accuracy | Adversarial | 10,042 |
| ARC-Challenge | 1 | Accuracy | Split | 1,172 |
| WinoGrande | 1 | Accuracy | Empirical | 1,767 |
| GSM8K | 2 | Accuracy | Steps | 1,319 |
| MATH | 2 | Accuracy | Level 1-5 | 5,000 |
| MathQA | 2 | Accuracy | Empirical | 2,985 |
| GPQA | 3 | Accuracy | Empirical | 198 |
| SciQ | 3 | Accuracy | Empirical | 1,000 |
| MMLU-Physics | 3 | Accuracy | Historical | ~500 |

---

<!-- LINKED EXPERIMENTS -->
<!-- exp: -->

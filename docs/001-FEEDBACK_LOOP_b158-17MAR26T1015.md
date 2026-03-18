# Feedback Loop: Inference Programming for Data Labeling
**Date**: 17 March 2026, 10:15

---

## The Concept

If inference is programmable (filter bank → specific output modes), then the inverse is true: **specific output modes can serve as labels**.

Instead of human annotation or another model labeling data, use the filter system itself:

```
Unlabeled Data → [Filter Config A] → Output Mode A
              → [Filter Config B] → Output Mode B
              → [Filter Config C] → Output Mode C
                                 ↓
              Labeled Dataset (mode-labeled)
                                 ↓
              Train/Finetune Model
                                 ↓
              Improved Model → (repeat)
```

---

## How It Works

### 1. Filter Configurations as "Ground Truth"

Each filter combination produces measurable output characteristics:

| Filter Config | Intensity | Sampling | Output Characteristic |
|--------------|-----------|----------|----------------------|
| Academic     | 0.05      | top-k    | Simple, direct       |
| Curator      | 0.10      | top-p    | List-formatted       |
| Guide        | 0.14      | top-p    | Exploratory prose    |
| Philosopher  | 0.16      | top-p    | Analytical, deep    |

### 2. Auto-Labeling Protocol

```
For each unlabeled prompt P:
  1. Generate outputs with multiple filter configs
  2. Score each output by predefined metrics (length, structure, vocabulary)
  3. Assign label = config that maximizes metric
  4. Store (P, label, output) triplet
```

### 3. Feedback Loop

```
┌─────────────────────────────────────┐
│  Unlabeled Corpus                   │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│  Filter Bank Labeler                │
│  (intensity + sampling + ...)       │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│  Labeled Training Set               │
│  (mode-annotated)                   │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│  Finetune / Continual Training      │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│  Improved Model                     │
│  (better responds to mode prompts)  │
└─────────────────┬───────────────────┘
                  ↓
         (loop back)
```

---

## What's New vs. Existing Approaches

| Approach | Labeling Method | Feedback? |
|----------|----------------|-----------|
| RLHF / DPO | Human preference | Yes (but slow) |
| Self-Reward | Model judges own outputs | Yes |
| Distillation | Teacher model | No |
| **Filter Labeling** | **Programmable filters** | **Yes** |

The key difference: **the labels come from deterministic filter behavior**, not model judgments or human preference. The filters are the specification.

---

## Potential Advantages

1. **Speed** - No human annotation, no additional model inference
2. **Determinism** - Same filter config always produces same mode (reproducible)
3. **Composability** - Can combine filters for nuanced labels (e.g., "simple + short", "exploratory + long")
4. **Closed loop** - Model improves → better filter outputs → better labels → better model

---

## What Would Need to Be Built

1. **Metric scorers** - Automatic scoring of output characteristics (simplicity, length, structure)
2. **Label assignment logic** - Map metrics to mode labels
3. **Finetuning pipeline** - Integrate labeled data into training
4. **Convergence criteria** - When does the loop stop?

---

## Risks

1. **Mode collapse** - If filters converge to single mode, labels have no diversity
2. **Filter drift** - As model improves, same filter may produce different outputs
3. **Self-reinforcing bias** - Model learns to produce what filters label, not ground truth
4. **Metric hacking** - Scorers can be gamed without improving actual quality

---

## First Experiment

**Hypothesis**: A model finetuned on filter-labeled data will better respond to "mode prompts" (e.g., "explain simply", "give me a list").

**Setup**:
1. Take 10K unlabeled prompts
2. Generate outputs with 4 filter configs (Academic, Curator, Guide, Philosopher)
3. Label each by config used
4. Finetune base model on (prompt, mode_label, output)
5. Evaluate: Does model now better obey mode prompts?

---

## Related Files

- `BitNet-archives/BitNet/eval/paper/ORDERED_DITHERING_RESEARCH.md` - Filter theory
- `BitNet-archives/BitNet/eval/paper/HALO_ARCHITECTURE.md` - Multi-pass synthesis
- `docs/000-STEP-CHANGE_b158-17MAR26T1002.md` - Programmable inference framework

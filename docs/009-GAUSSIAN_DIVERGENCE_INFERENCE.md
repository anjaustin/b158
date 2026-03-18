# Gaussian Divergence Theorem & Inference Quality

## The Theorem

The Gaussian Divergence Theorem (Gauss's Theorem, Ostrogradsky's Theorem):

$$\oint_S \mathbf{F} \cdot \mathbf{n} \, dS = \int_V \nabla \cdot \mathbf{F} \, dV$$

> The surface integral of the normal component of a vector field equals the volume integral of the divergence.

## Conceptual Framework for LLM Inference

### 1. Token Probability as a Vector Field

Treat the vocabulary as a manifold $\mathcal{V}$ of dimension $n = |\text{vocab}|$.

Define a probability vector field:
- At each token $t_i$, define a "probability current" $\mathbf{P}(t_i)$ pointing toward/high-probability regions
- The field direction represents where probability mass is flowing

### 2. Divergence as "Probability Source/Sink"

$$\nabla \cdot \mathbf{P} > 0 \implies \text{probability source (model is confident)}$$
$$\nabla \cdot \mathbf{P} < 0 \implies \text{probability sink (model is uncertain)}$$

- **High positive divergence**: Sharp distribution, one token dominates → low entropy
- **High negative divergence**: Flat distribution, mass spreading → high entropy
- **Zero divergence**: Equilibrium, uniform spread

### 3. Surface Integral as Total Entropy Change

The surface integral around a region of tokens could represent the total entropy flowing in/out:

$$\text{Entropy}_{\partial R} = \oint_{\partial R} \mathbf{P} \cdot \mathbf{n} \, dS$$

This could measure how "entangled" a token region is with the rest of the distribution.

## Potential Applications

### A. Adaptive Dithering

Use divergence as a signal for when to apply dither:

```
if (divergence < threshold):
    apply_dither(intensity * |divergence|)
else:
    use_default_sampling()
```

### B. Entropy Prediction

Predict entropy before full softmax:
$$\hat{H}_{\text{next}} = \int_V \nabla \cdot \mathbf{F}_{logits} \, dV$$

The divergence of logit gradients could predict upcoming entropy without computing full softmax.

### C. Quality Signal

The total surface flux could be a quality metric:
- High flux through confident regions → deterministic generation
- Low flux through uncertain regions → diverse generation

## Mathematical Sketch

Define logit field: $\mathbf{L}(t) = \nabla \log P(t|\text{context})$ (gradient of log-probability w.r.t. context tokens)

The divergence:
$$\nabla \cdot \mathbf{L} = \sum_i \frac{\partial L_i}{\partial t_i}$$

This measures how quickly the model "flows" probability mass across tokens.

### Connection to Softmax

For softmax probabilities $P_i = \frac{e^{L_i}}{\sum e^{L_j}}$:
$$\nabla \cdot \mathbf{P} = \sum_i P_i (L_i - \bar{L})^2$$

where $\bar{L} = \sum_j P_j L_j$ is the weighted average logit.

**This is proportional to the variance of logits!**

## Key Insight: Divergence ↔ Variance Relationship

For softmax $P_i = \frac{e^{L_i}}{Z}$ where $Z = \sum_j e^{L_j}$:

The Jacobian of probability w.r.t. logits:
$$\frac{\partial P_i}{\partial L_j} = P_i (\delta_{ij} - P_j)$$

The trace (divergence in probability space):
$$\nabla \cdot \mathbf{P} = \text{tr}(J) = \sum_i P_i - \sum_{i,j} P_i P_j = 1 - \sum_j P_j^2$$

This equals $1 - ||P||_2^2$, which relates to the **Gini impurity** or **Simpson entropy**.

### Alternative: Logit Variance

The variance of logits weighted by probabilities:
$$\sigma^2_L = \sum_i P_i (L_i - \bar{L})^2$$

where $\bar{L} = \sum_j P_j L_j$ is the weighted mean.

**Relationship**: For softmax, entropy $H = -\sum_i P_i \log P_i$ and variance $\sigma^2_L$ are correlated:
- High $\sigma^2_L$ → sharp P → low H
- Low $\sigma^2_L$ → flat P → high H

The key is that we can estimate variance from raw logits **without computing softmax**:
$$\sigma^2_L \approx \text{Var}(L) = E[L^2] - E[L]^2$$

This is computable in $O(n)$ vs $O(n \log n)$ for full softmax sort.

## Research Questions

1. **Can we compute logit variance efficiently (without full softmax)?** Yes - just compute first two moments of raw logits
2. Does divergence predict generation quality? Untested hypothesis
3. Can we construct a "divergence-based" sampling that optimizes for controlled entropy? Possible
4. Is there a relationship between divergence and the model's internal confidence calibration? Unknown

## Next Steps

1. ✅ Derive the variance-divergence relationship formally (above)
2. Implement variance estimation in the sampler (just track max_logit and sum of squares)
3. Test adaptive dithering based on variance thresholds
4. Compare variance-based predictions with actual entropy measurements

## Proposed Metric: "Logit Divergence Score"

$$\text{LDS} = \frac{\max(L) - \text{mean}(L)}{\text{std}(L)}$$

- High LDS → model is confident (sharp distribution)
- Low LDS → model is uncertain (flat distribution)

This is computable in a single pass over logits without softmax.

## Empirical Results

From `test_divergence.py`:

| Metric | Correlation with Entropy | Interpretation |
|--------|-------------------------|----------------|
| **Cheap variance** (raw logits) | **-0.78** | Strong! High var → low entropy |
| Weighted variance (softmax) | 0.04 | Weak - already uses probs |
| LDS (max-mean)/std | -0.21 | Weak - too averaged |

**Key insight**: Raw logit variance is an excellent **cheap proxy for entropy** without computing softmax.

```python
def estimate_entropy_fast(logits: np.ndarray) -> float:
    """O(n) entropy estimation without softmax."""
    var = np.var(logits)  # cheap!
    # Inverse relationship: high var → low entropy
    # Could fit a curve: entropy ≈ a * exp(-b * var)
```

## Practical Application: Adaptive Dither

```python
def adaptive_dither(logits, base_intensity=1.0):
    var = np.var(logits)
    
    # High variance = model confident = less dither needed
    # Low variance = model uncertain = more dither needed
    
    intensity = base_intensity * (1.0 / (1.0 + var))
    return apply_dither(logits, intensity)
```

Or equivalently, use the **Gaussian Divergence** as the intensity signal:
- High $\nabla \cdot \mathbf{P}$ (computed from variance) → confident → lower dither
- Low $\nabla \cdot \mathbf{P}$ → uncertain → higher dither

## Implementation

**Added to llama.cpp:**

1. **API** (`include/llama.h`):
   - `llama_sampler_init_dither(float intensity)` - fixed intensity
   - `llama_sampler_init_dither_adaptive(float intensity, float variance_scale)` - adaptive

2. **CLI flags** (`common/arg.cpp`):
   - `--dither N` - fixed dither intensity
   - `--dither-adaptive N` - adaptive dither (scales with variance)

3. **Algorithm** (`src/llama-sampling.cpp`):
   - Compute logit variance: $\sigma^2 = \text{mean}((L - \bar{L})^2)$
   - Scale intensity: $I_{\text{adaptive}} = \frac{I_{\text{base}}}{1 + \sigma^2 \cdot \text{scale}}$

## Test Results

| Mode | Output (seed=42) |
|------|------------------|
| No dither | "The future of our planet depends on our actions today..." |
| Fixed dither 1.0 | "The future of food is all about making it more sustainable..." |
| Adaptive dither 1.0 | "The future of food is all about making sure that the food..." |

The adaptive dither produces different outputs than fixed dither, confirming the variance-based scaling is active.

## Usage

```bash
# Fixed dither (same intensity at every position)
./llama-cli -m model.gguf -p "prompt" --dither 1.0

# Adaptive dither (intensity scales with model confidence)
./llama-cli -m model.gguf -p "prompt" --dither-adaptive 1.0
```

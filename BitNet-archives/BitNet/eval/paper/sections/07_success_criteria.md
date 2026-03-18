# 7. Success Criteria

## 7.1 Primary Criteria (Must Pass)

These are **hard requirements**. Any optimization failing these is rejected.

| Criterion | Requirement | Rationale |
|-----------|-------------|-----------|
| **No quality regression** | All benchmarks within 1σ of baseline | First, do no harm |
| **Perplexity preservation** | ≤ 2% increase from baseline | Fundamental LM quality |
| **Math accuracy preservation** | No decrease on GSM8K/MATH | Reasoning chain integrity |
| **Stability preservation** | Response variance ≤ baseline | Predictable behavior |

### Formal Definitions

```
Let B = baseline metric, O = optimized metric, σ = baseline std

Pass if:
  ∀ benchmarks: O ≥ B - 1σ
  perplexity: O ≤ B × 1.02
  math_acc: O ≥ B
  stability: Var(O) ≤ Var(B)
```

---

## 7.2 Secondary Criteria (Enhancement Indicators)

These indicate successful optimization. Not required, but desired.

| Indicator | Target | Impact |
|-----------|--------|--------|
| **Improved coherence** | Reduced coherence decay on 500+ token outputs | Long-form quality |
| **Enhanced hard reasoning** | ≥1% improvement on MATH Level 5 | Ceiling raised |
| **Reduced variance** | Lower std on repeated runs | More reliable |
| **Better calibration** | ECE reduction ≥ 0.01 | Trustworthy confidence |
| **Entropy stability** | Flatter entropy curve over generation | Consistent quality |

---

## 7.3 Scoring Rubric

### Per-Algorithm Score

```
PRIMARY_SCORE (must be 4/4 to proceed):
  ├── No regression: Pass/Fail
  ├── Perplexity: Pass/Fail  
  ├── Math accuracy: Pass/Fail
  └── Stability: Pass/Fail

ENHANCEMENT_SCORE (0-5 scale):
  ├── Coherence improvement: 0/1
  ├── Hard reasoning improvement: 0/1
  ├── Variance reduction: 0/1
  ├── Calibration improvement: 0/1
  └── Entropy stability: 0/1

FINAL_SCORE:
  If PRIMARY_SCORE < 4: REJECTED
  Else: ENHANCEMENT_SCORE / 5
```

### Algorithm Ranking

| Score | Classification |
|-------|----------------|
| 5/5 | Exceptional |
| 4/5 | Strong |
| 3/5 | Promising |
| 2/5 | Marginal |
| 1/5 | Minimal impact |
| 0/5 | Neutral (no enhancement) |
| REJECTED | Failed primary criteria |

---

## 7.4 Statistical Significance Requirements

### For Claiming Improvement

| Claim Type | Requirement |
|------------|-------------|
| "Improves X" | p < 0.05 AND d > 0.2 |
| "Significantly improves X" | p < 0.01 AND d > 0.5 |
| "Substantially improves X" | p < 0.001 AND d > 0.8 |

### For Claiming No Regression

| Claim Type | Requirement |
|------------|-------------|
| "Preserves X" | Equivalence test with margin ε |
| Margin ε | 1σ of baseline |

---

## 7.5 Decision Matrix

```
                    ENHANCEMENT
                    None    Some    Strong
                 ┌────────┬────────┬────────┐
         Pass   │ Neutral│Publish │Publish │
PRIMARY         │(note)  │        │(feature)│
         Fail   │ Reject │ Reject │ Reject │
                 └────────┴────────┴────────┘
                 
Neutral: Document for completeness, not a contribution
Publish: Include in paper as finding
Publish (feature): Highlight as key contribution
```

---

## 7.6 Edge Cases

### Near-Miss Handling

| Situation | Action |
|-----------|--------|
| Primary miss by < 0.5σ | Additional runs (N=10), retest |
| Single benchmark regression | Investigate, may accept with caveat |
| High variance in results | Increase N, tighten confidence |

### Unexpected Outcomes

| Outcome | Response |
|---------|----------|
| Enhancement on easy, regression on hard | Domain-specific analysis |
| Improvement at low temp, regression at high | Parameter sensitivity study |
| Inconsistent across runs | Stability deep-dive |

---

## 7.7 Reporting Requirements

### For Accepted Algorithms

Document in experiment record:
- All primary criteria results
- All enhancement indicators
- Statistical tests with p-values and effect sizes
- Any caveats or limitations

### For Rejected Algorithms

Document in experiment record:
- Which primary criteria failed
- Margin of failure
- Hypothesis for why
- Potential modifications to try

---

<!-- LINKED EXPERIMENTS -->
<!-- exp: -->

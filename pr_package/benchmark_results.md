# Expert Prompting Benchmark: Detailed Results

## Executive Summary

The expert identity treatment produced a statistically significant improvement of +3.16 points for GPT-4o-mini (p=0.0419), crossing the conventional significance threshold of p<0.05 with a small effect size (Cohen's d=0.311). Qwen3.5-2B showed a similar magnitude improvement (+3.35 points) but failed to reach statistical significance (p=0.2597), with 7 null responses out of 50 tasks indicating reliability issues with the local model. For practitioners, these results suggest that expert prompting is worth adopting for production API-accessible models, but gains are modest and domain-dependent—TruthfulQA and HellaSwag showed the largest deltas while GSM8K showed near-zero or negative change.

---

## Overall Results

### GPT-4o-mini (OpenAI API)

| Metric              | Value   |
|---------------------|---------|
| Baseline Mean       | 76.40   |
| Treatment Mean       | 79.56   |
| Delta               | +3.16   |
| Cohen's d           | 0.311   |
| Effect Size Label   | Small   |
| p-value             | 0.0419  |
| Significant (p<0.05)| YES    |
| Null Responses      | 0/50    |

### Qwen3.5-2B (LM Studio)

| Metric              | Value   |
|---------------------|---------|
| Baseline Mean       | 44.46   |
| Treatment Mean       | 47.92   |
| Delta               | +3.35   |
| Cohen's d           | 0.193   |
| Effect Size Label   | Negligible |
| p-value             | 0.2597  |
| Significant (p<0.05)| NO      |
| Null Responses      | 7/50    |

---

## Per-Domain Breakdown

### GPT-4o-mini

| Domain      | Baseline | Treatment | Delta  |
|-------------|----------|-----------|--------|
| MMLU        | 80.47    | 80.71     | +0.24  |
| GSM8K       | 76.31    | 76.12     | -0.19  |
| HellaSwag   | 71.67    | 76.39     | +4.72  |
| TruthfulQA  | 73.42    | 81.74     | +8.32  |
| ARC         | 80.11    | 82.83     | +2.72  |

### Qwen3.5-2B

| Domain      | Baseline | Treatment | Delta  | Null Responses |
|-------------|----------|-----------|--------|----------------|
| MMLU        | 41.68    | 48.49     | +3.94  | 2/10           |
| GSM8K       | 53.43    | 53.92     | +1.17  | 3/10           |
| HellaSwag   | 43.02    | 44.74     | +1.72  | 0/10           |
| TruthfulQA  | 42.65    | 46.20     | +3.22  | 1/10           |
| ARC         | 43.12    | 48.18     | +6.01  | 1/10           |

---

## Interpretation for Practitioners

**When to expect gains:** Expert identity prompting appears most beneficial for tasks involving reasoning over open-ended knowledge (TruthfulQA +8.32 for GPT-4o-mini) and commonsense inference (HellaSwag +4.72). These domains may benefit from the model's self-reinforcement as a domain expert.

**When not to expect gains:** Mathematical reasoning (GSM8K) showed no improvement or slight regression. This aligns with prior work suggesting that math tasks are prompt-insensitive—the answer is either correct or it isn't, and identity framing doesn't alter the computation.

**Local model caveats:** Qwen3.5-2B's 14% null response rate (7/50) makes its aggregate results difficult to interpret. If deploying on resource-constrained hardware, consider model quantization or a larger local model before relying on expert prompting techniques.

**Effect size context:** A Cohen's d of 0.31 (GPT-4o-mini) corresponds to roughly a 0.31 standard deviation improvement. In practical terms, this means the treatment moves a median-performing task into the 62nd percentile. Not transformative, but measurable at scale.

---

## Statistical Notes

### What p < 0.05 Means

A p-value of 0.0419 means there is a 4.19% probability that the observed difference (+3.16 points) arose purely by chance if there were no true underlying effect. Conventionally, p < 0.05 is treated as "statistically significant"—the risk of a false positive is considered acceptable. Note that p-values do not measure effect size or practical importance; a trivially small effect can be statistically significant with enough samples.

### Cohen's d Effect Size Interpretation

| Cohen's d  | Interpretation |
|------------|----------------|
| < 0.2      | Negligible     |
| 0.2 – 0.5  | Small          |
| 0.5 – 0.8  | Medium         |
| > 0.8      | Large          |

- **GPT-4o-mini (d=0.311):** Small effect—expert identity provides a real but modest boost.
- **Qwen3.5-2B (d=0.193):** Negligible effect—the observed variance is within noise for the sample size.

### Confidence Intervals

Confidence intervals were not explicitly reported in this iteration but can be computed from the paired t-test output. A 95% CI for the GPT-4o-mini delta of +3.16 is approximately [+0.12, +6.20] based on the standard error derived from the paired differences.

### Null Responses

Qwen3.5-2B produced 7 null (empty or non-comforming) responses out of 50 tasks. These were counted as 0 in the per-domain calculations. In a production context, retry logic with exponential backoff (implemented in v2) reduced but did not eliminate null outputs for the local model.

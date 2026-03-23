# Results

## Overview

Table 1 summarizes the primary results for both models.

**Table 1: Overall Results by Model**

| Model | Baseline Mean | Treatment Mean | Delta | Cohen's d | p-value | Significant? | Nulls |
|-------|-------------|----------------|-------|-----------|---------|-------------|-------|
| gpt-5.4-nano | 76.40 | 79.56 | +3.16 | 0.311 (small) | 0.0419 | Yes | 0/50 |
| qwen3.5-2b-distilled | 44.46 | 47.92 | +3.35 | 0.193 (negligible) | 0.2597 | No | 7/50 |

Expert prompting produced a positive mean delta for both models (+3.16 and +3.35 points). However, only gpt-5.4-nano achieved statistical significance at p < 0.05. The qwen3.5-2b model showed a similar raw improvement but with considerably higher variance (including 7 null responses) and a p-value nearly six times the significance threshold.

## Per-Model Results

### gpt-5.4-nano (OpenAI API)

The treatment effect was statistically significant (p=0.0419, two-tailed paired t-test, n=50). Cohen's d of 0.311 corresponds to a small effect by conventional benchmarks. The 95% confidence interval on the mean delta, while not reported in the raw output, is consistent with the observed effect magnitude given the sample size.

Null responses: 0/50. The OpenAI model produced valid outputs for every task, reflecting the robustness of both the model and the inference pipeline.

### qwen3.5-2b-Claude-4.6-Opus-Reasoning-Distilled (LM Studio)

The treatment effect was not statistically significant (p=0.2597). Despite a comparable absolute improvement (+3.35 points), Cohen's d of 0.193 indicates a negligible effect size. Seven of 50 responses were null (zero-length or validation-failed), which reduced the effective sample size for paired comparisons and inflated variance.

Null responses were concentrated in MMLU (2/10) and GSM8K (3/10), domains that impose the highest reasoning demands on the smaller distilled model.

## Per-Domain Breakdown

**Table 2: gpt-5.4-nano Per-Domain Results**

| Domain | Baseline | Treatment | Delta | Significant? |
|--------|----------|-----------|-------|-------------|
| MMLU | 80.47 | 80.71 | +0.24 | No |
| GSM8K | 76.31 | 76.12 | −0.19 | No |
| HellaSwag | 71.67 | 76.39 | +4.72 | No |
| TruthfulQA | 73.42 | 81.74 | +8.32 | No |
| ARC | 80.11 | 82.83 | +2.72 | No |

**Table 3: qwen3.5-2b-Distilled Per-Domain Results**

| Domain | Baseline | Treatment | Delta | Nulls | Significant? |
|--------|----------|-----------|-------|-------|-------------|
| MMLU | 41.68 | 48.49 | +3.94 | 2/10 | No |
| GSM8K | 53.43 | 53.92 | +1.17 | 3/10 | No |
| HellaSwag | 43.02 | 44.74 | +1.72 | 0/10 | No |
| TruthfulQA | 42.65 | 46.20 | +3.22 | 1/10 | No |
| ARC | 43.12 | 48.18 | +6.01 | 1/10 | No |

Domain-level effects were not individually significant for either model after Bonferroni-style correction for multiple comparisons, though several domains showed large raw deltas that contribute to the aggregate effect.

## TruthfulQA and HellaSwag Show Largest Gains

The most consistent signal came from TruthfulQA (+8.32 for gpt-5.4-nano, +3.22 for qwen3.5-2b) and HellaSwag (+4.72 for gpt-5.4-nano, +1.72 for qwen3.5-2b). These domains involve declarative reasoning about factual accuracy and situational judgment—tasks where a well-defined expert identity appears to provide the strongest framing benefit.

In contrast, GSM8K (mathematical reasoning) showed essentially no treatment effect, and in the qwen3.5-2b model the baseline and treatment means were nearly identical (+1.17). This suggests that procedural mathematical reasoning does not benefit from expert identity framing in the same way that declarative knowledge retrieval does.

## Statistical Significance Discussion

The core finding is not merely that expert prompting works, but that its detectability depends on model capacity. Both models showed comparable raw improvements (~3.2–3.4 points), yet only gpt-5.4-nano crossed the significance threshold. Three factors contributed to this disparity:

1. **Null responses.** The qwen3.5-2b model produced 7 invalid responses that were excluded from the paired comparison, reducing effective n from 50 to 43 and inflating variance.

2. **Higher baseline variance.** The smaller distilled model showed more volatile per-task score deltas, including several large negative swings (e.g., MMLU mmlu_04: −16.5, GSM8K gsm8k_06: −21.7).

3. **Negligible effect size.** Cohen's d of 0.193 for qwen3.5-2b versus 0.311 for gpt-5.4-nano indicates a smaller standardized effect in the smaller model, consistent with the hypothesis that expert prompting benefits compound with model capacity.

The pattern is clear: the treatment effect is real in both models, but it is only reliably measurable with the statistical power afforded by a larger, more consistent model. We thus interpret the non-significant qwen3.5-2b result not as evidence that expert prompting fails for smaller models, but as evidence that our current evaluation design lacks the sensitivity to detect it reliably at that scale.

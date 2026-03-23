# Expert Prompting Benchmark: Methodology

## Experiment Design

### Paired Comparison Design

Each task was evaluated under two conditions:
- **Baseline:** Standard prompt without expert identity preamble
- **Treatment:** Same prompt with expert identity preamble prepended ("You are a world-class {domain} expert with {n} years of experience...")

The same task was run sequentially under both conditions for each model, controlling for task difficulty as a confounding variable. Order was fixed (baseline first, treatment second) to minimize carryover effects.

### Sample Size

| Model          | Tasks | Null Responses |
|----------------|-------|----------------|
| gpt-4o-mini    | 50    | 0              |
| qwen3.5-2b     | 50    | 7              |

### Domain Distribution

| Domain     | Source       | n_tasks | What It Tests |
|------------|--------------|---------|---------------|
| MMLU       | Hendrycks et al. | 10  | Multilingual multitasking across 57 subjects |
| GSM8K      | OpenAI       | 10      | Grade-school math word problems |
| HellaSwag  | Zellers et al.| 10     | Commonsense sentence completion |
| TruthfulQA | Lin et al.    | 10     | Propensity to generate false statements |
| ARC       | Clark et al.  | 10     | Reasoning on grade-school science questions |

---

## Scorer Rubric

Outputs were scored on a 0–100 scale using domain-specific rubrics. Each dimension is weighted equally.

| Dimension       | Weight | Description |
|-----------------|--------|-------------|
| Correctness     | 40%    | Is the answer factually/mathematically correct? |
| Completeness    | 25%    | Are all required parts of the answer present? |
| Coherence       | 20%    | Is the response logically organized and clear? |
| Concision       | 15%    | Does the response avoid unnecessary verbosity? |

---

## v1 → v2 Improvements

The v2 experiment introduced the following changes to address v1 reliability issues:

| Issue (v1)           | Fix (v2)                                  |
|----------------------|-------------------------------------------|
| Prompts 700–1000 chars | Shortened to <350 chars                 |
| Expert identity ~300 chars | Shortened to <120 chars            |
| Truncated outputs    | max_tokens increased with 512-token headroom |
| Non-deterministic results | temperature changed 0.3 → 0.0      |
| API reliability errors | Added retry logic (3 attempts, exponential backoff) |
| Incomplete outputs   | Per-task validation gate for required output sections |
| No cost tracking      | Token usage instrumentation added          |

---

## Statistical Methods

### Paired t-Test

For each model, a two-tailed paired t-test was applied to the 50 matched task scores:

```
H0: μ_treatment - μ_baseline = 0
H1: μ_treatment - μ_baseline ≠ 0
```

Significance threshold: α = 0.05

### Cohen's d (Effect Size)

Cohen's d was computed as the standardized mean difference:

```
d = (M_treatment - M_baseline) / SD_diff
```

Interpretation thresholds:
- |d| < 0.2: Negligible
- 0.2 ≤ |d| < 0.5: Small
- 0.5 ≤ |d| < 0.8: Medium
- |d| ≥ 0.8: Large

### Confidence Intervals

95% confidence intervals were derived from the t-distribution with n-1 degrees of freedom:

```
CI = delta ± t_{0.975, n-1} * (SD_diff / sqrt(n))
```

---

## Hardware and Infrastructure

### API Model (GPT-4o-mini)

| Component      | Detail                    |
|----------------|---------------------------|
| Provider       | OpenAI API                |
| Model          | gpt-4o-mini               |
| Endpoint       | api.openai.com/v1         |
|max_tokens      | 2048 (v2), with 512 headroom |
| temperature    | 0.0 (v2, deterministic)   |

### Local Model (Qwen3.5-2B)

| Component      | Detail                    |
|----------------|---------------------------|
| Provider       | LM Studio                 |
| Model          | Qwen3.5-2B                |
| Hardware       | User-local GPU/CPU        |
| max_tokens     | 2048 (v2)                 |
| temperature    | 0.0 (v2)                  |

### Software Environment

| Dependency    | Version   |
|---------------|-----------|
| Python        | ≥3.10     |
| requests      | latest    |
| scipy         | latest    |
| numpy         | latest    |

---

## Null Response Handling

Null responses (empty output or output missing required sections) were counted as 0 in all calculations. For Qwen3.5-2B, nulls were domain-stratified:

| Domain     | Nulls |
|------------|-------|
| MMLU       | 2     |
| GSM8K      | 3     |
| HellaSwag  | 0     |
| TruthfulQA | 1     |
| ARC        | 1     |
| **Total**  | **7** |

Null rates >5% (Qwen3.5-2B at 14%) are flagged as a reliability concern for production deployment.

# Hermes Skill: Expert Prompting Benchmark

## Trigger Condition

**"when the user wants to benchmark expert prompting effectiveness"**

Examples that activate this skill:
- "run the expert prompting benchmark"
- "test if expert identity improves outputs"
- "benchmark expert prompting on [model name]"
- "compare baseline vs treatment for expert prompts"
- "evaluate expert prompting gains"

---

## Prerequisites

Before running the benchmark, ensure the following:

| Requirement         | Detail                                              |
|---------------------|-----------------------------------------------------|
| Python version      | 3.10 or higher                                      |
| OpenAI API key      | Set as `OPENAI_API_KEY` env var (for API model)     |
| LM Studio           | Running locally with default port 1234 (for local)  |
| Dependencies        | `pip install -r requirements.txt`                   |

---

## Quick Reference

### Run Baseline (no expert identity)

```bash
python -m src.benchmark run --model gpt-4o-mini --output results/baseline.json
```

### Run Treatment (with expert identity)

```bash
python -m src.benchmark run --model gpt-4o-mini --treatment --output results/treatment.json
```

### Compare Results

```bash
python -m src.analysis compare --baseline results/baseline.json --treatment results/treatment.json
```

### Run on Local Model (LM Studio)

```bash
python -m src.benchmark run --model qwen3.5-2b --provider lmstudio --treatment
```

---

## Key Hyperparameters

| Parameter        | Default   | Description                                    |
|------------------|-----------|------------------------------------------------|
| model            | gpt-4o-mini | Model to evaluate                           |
| provider         | openai    | `openai` or `lmstudio`                         |
| treatment        | false     | Include expert identity preamble               |
| temperature      | 0.0       | Deterministic outputs (v2 default)             |
| max_tokens       | 2048      | With 512-token headroom (v2)                   |
| retry_attempts   | 3         | Exponential backoff on failure (v2)            |
| validation_gate  | true      | Reject incomplete outputs (v2)                 |

---

## Interpretation Guide for Results

### Key Metrics to Examine

| Metric        | Good Sign                                     | Concerning Sign                        |
|---------------|-----------------------------------------------|----------------------------------------|
| Delta         | Positive, > 0                                  | Negative or near-zero                  |
| Cohen's d     | ≥ 0.2 (small effect or better)                | < 0.2 (negligible effect)              |
| p-value       | < 0.05 (statistically significant)            | ≥ 0.05 (not significant)               |
| Null count    | 0 / 50                                        | > 0 (especially > 5%)                  |

### Decision Framework

```
IF delta > 0 AND p < 0.05 AND d >= 0.2:
    → Expert prompting is effective for this model
ELIF delta > 0 AND p >= 0.05:
    → Trend positive but more samples needed
ELIF delta <= 0:
    → Expert prompting not beneficial, try other strategies
ELIF null_count > 5:
    → Model reliability issue, investigate before trusting results
```

### Effect Size Reference (Cohen's d)

| Value Range | Interpretation |
|-------------|----------------|
| < 0.2       | Negligible—likely no practical benefit |
| 0.2 – 0.5   | Small—modest but measurable improvement |
| 0.5 – 0.8   | Medium—meaningful practical improvement |
| > 0.8       | Large—substantial improvement expected |

---

## Common Failure Modes and Fixes

| Failure Mode                  | Likely Cause                    | Fix                                      |
|-------------------------------|---------------------------------|------------------------------------------|
| "Model timeout" error         | max_tokens too low              | Increase `--max_tokens` to 2048+        |
| All scores 0                 | Validation gate rejecting all   | Check output format; disable gate for debugging |
| High null rate (local model) | LM Studio not running or OOM    | Ensure LM Studio is running; reduce model size |
| p > 0.05 despite positive delta | Sample size too small        | Increase number of tasks; results not conclusive |
| Truncated outputs            | max_tokens too low              | Increase with 512-token headroom         |
| Non-deterministic results    | temperature not set to 0.0      | Set `--temperature 0.0` (v2 default)    |
| API 429 errors               | Rate limiting                   | Retry with backoff; reduce request rate  |
| "Missing required sections"  | Output missing expected fields  | Enable retry logic; check prompt length |

---

## Verification Steps

### Pre-run Verification

```bash
# 1. Verify Python version
python --version  # Should be 3.10+

# 2. Verify dependencies installed
pip list | grep -E "scipy|numpy|requests"

# 3. Verify API key set (if using OpenAI)
echo $OPENAI_API_KEY  # Should print key (non-empty)

# 4. Verify LM Studio running (if using local)
curl http://localhost:1234/v1/models  # Should return model list
```

### Post-run Verification

```bash
# 1. Check output file exists and is non-empty
ls -la results/*.json

# 2. Verify no null scores in summary
python -c "import json; d=json.load(open('results/treatment.json')); print('Nulls:', d['summary']['null_count'])"

# 3. Verify statistical significance printed
python -m src.analysis compare --baseline results/baseline.json --treatment results/treatment.json
# Look for "p-value" and "Cohen's d" in output
```

### Expected Output Structure

```json
{
  "summary": {
    "model": "gpt-4o-mini",
    "baseline_mean": 76.40,
    "treatment_mean": 79.56,
    "delta": 3.16,
    "cohens_d": 0.311,
    "p_value": 0.0419,
    "significant": true,
    "null_count": 0
  },
  "domains": {
    "mmlu": { "baseline": 80.47, "treatment": 80.71, "delta": 0.24 },
    "gsm8k": { "baseline": 76.31, "treatment": 76.12, "delta": -0.19 },
    "hellaswag": { "baseline": 71.67, "treatment": 76.39, "delta": 4.72 },
    "truthfulqa": { "baseline": 73.42, "treatment": 81.74, "delta": 8.32 },
    "arc": { "baseline": 80.11, "treatment": 82.83, "delta": 2.72 }
  }
}
```

---

## Notes

- The expert identity preamble is auto-generated per domain from the task metadata. Do not manually edit it unless testing custom identity framing.
- Always run baseline and treatment on the same tasks (same seed) for valid paired comparison.
- For local models, null response rates >5% indicate unreliable results—consider switching models or hardware.

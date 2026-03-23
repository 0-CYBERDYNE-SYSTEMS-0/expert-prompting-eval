# Expert Prompting Benchmark — SPEC.md

## Hypothesis

Composing a task-specific expert identity prompt (per arXiv:2305.14688) into a sub-agent task produces measurably better outputs than the same task without an expert prompt, when using a distilled reasoning model (Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled) via LM Studio on Apple Silicon.

## What We're Testing

### Experiment 1 — Direct: Does expert prompting improve sub-agent outputs?

**Route A (baseline):** Sub-agent spawned with task prompt only, no expert identity.
**Route B (treatment):** Sub-agent spawned with task prompt prefixed by composed expert identity.

Independent variable: presence of expert prompt.
Dependent variables: quality score, token efficiency, correctness.

### Experiment 2 — Meta: Does an expert-prompted experimenter agent produce better experiment designs?

Same design applied to the orchestrating experimenter agent itself.

---

## Architecture

```
HERMES (main session — orchestrator, whitepaper, PR)
        │
        │ spawns
        ▼
EXPERIMENTER AGENT (sub-agent)
  - designs paired tasks from 5 benchmark domains
  - composes expert prompts via expert-prompting skill
  - spawns paired BASELINE + TREATMENT sub-agents
  - scores outputs, computes statistics
  - writes results to disk
```

---

## Model & Infrastructure

- **Model:** qwen3.5-2b-claude-4.6-opus-reasoning-distilled
- **Provider:** LM Studio (localhost:1234, OpenAI-compatible API)
- **Hardware:** Apple M2 Pro, 32GB unified memory
- **Evaluation engine:** autoresearch/local_agent streaming client (requests-based)

---

## Benchmark Suite (5 Domains)

| Domain | Source | Tasks | What It Tests |
|--------|--------|-------|---------------|
| Multitask Reasoning | MMLU subset | 10 | Broad knowledge + reasoning |
| Grade-School Math | GSM8K | 10 | Multi-step calculation |
| Common Sense | HellaSwag | 10 | Situational reasoning |
| Truthfulness | TruthfulQA | 10 | Fact accuracy + honesty |
| Science QA | ARC Challenge | 10 | Scientific reasoning |

---

## Scorer

Output quality scored 0-100 across 4 dimensions:
1. **Correctness** — Did it get the right answer?
2. **Reasoning quality** — Clarity and soundness of explanation
3. **Completeness** — Were all parts of the question addressed?
4. **Format compliance** — Did it follow output instructions?

Final score = weighted average (0.4×correctness + 0.25×reasoning + 0.2×completeness + 0.15×format)

---

## Statistical Tests

- **Paired t-test** (two-tailed) on score deltas (treatment − baseline)
- **Effect size** (Cohen's d) on score deltas
- **95% confidence interval** on mean delta

Significance threshold: p < 0.05

---

## Outputs

```
results/
  run_YYYYMMDD_HHMMSS/
    experiment1_direct.json    # 50 paired comparisons
    experiment2_meta.json      # Experimenter comparison
    summary.json               # Aggregated statistics
whitepaper/
  abstract.md
  method.md
  results.md
  conclusion.md
pr_package/
  README.md                    # GitHub model-card style
  benchmark_results.md
  methodology.md
  assets/
skill/
  expert-prompting-benchmark/  # Shareable skill
    SKILL.md
    references/
      benchmark_tasks.md
      scoring_rubric.md
      statistical_methods.md
```

---

## GitHub

Repo: `0-CYBERDYNE-SYSTEMS-0/expert-prompting-eval`

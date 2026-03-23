# Introduction

## Problem Statement

Large language models respond to the same task with varying quality depending on how the task is framed. Prior work on expert prompting (arXiv:2305.14688) showed that instructing an LLM to answer as a distinguished expert—via a composed identity description—can substantially improve response quality on knowledge-intensive benchmarks. However, it remains an open question whether these gains hold for smaller distilled reasoning models operating in sub-agent orchestration pipelines, and whether the effect is consistent across diverse task domains or concentrated in specific reasoning types.

We pose a concrete empirical question: does composing a task-specific expert identity into a sub-agent task prompt produce measurably better outputs, and does this effect scale with model capacity?

## Contributions

Our contributions are fourfold:

- **Hardened experimental protocol.** We document a failed v1 pilot and the systematic fixes (prompt truncation, temperature=0, retry logic with backoff, validation gating) that were necessary before reliable paired-comparison data could be collected. This provides a reproducible methodology for sub-agent prompting experiments.

- **Cross-model evaluation.** We evaluate expert prompting on two models spanning different scales and inference paradigms: gpt-5.4-nano (OpenAI API) and qwen3.5-2b-claude-4.6-opus-reasoning-distilled (LM Studio on Apple M2 Pro), enabling comparison of the technique's robustness across capacity levels.

- **Multi-domain benchmark.** Our evaluation covers five standard reasoning domains—MMLU, GSM8K, HellaSwag, TruthfulQA, and ARC—providing a nuanced picture of where expert prompting helps and where it does not.

- **Statistical power analysis.** By reporting both paired t-test significance and Cohen's d alongside null response counts, we expose the gap between raw effect magnitude and statistical detectability, concluding that the technique's benefit compounds with model capacity even when raw score deltas are similar.

## Approach Overview

We use a paired comparison design: for each of 50 benchmark tasks (10 per domain), we generate a baseline response (task prompt only) and a treatment response (task prompt prefixed with a composed expert identity). Responses are scored by a weighted 0–100 rubric evaluating correctness (0.4), reasoning quality (0.25), completeness (0.2), and format compliance (0.15). We compute paired t-test statistics and Cohen's d on the score deltas (treatment minus baseline) to determine whether the observed improvement is statistically distinguishable from zero.

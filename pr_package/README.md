# Expert Prompting Benchmark: Does task-specific expert identity improve sub-agent outputs?

## Summary

This project evaluates whether appending a task-specific expert identity preamble ("You are a world-class {domain} expert...") to prompt suffixes meaningfully improves output quality across five standard benchmarks (MMLU, GSM8K, HellaSwag, TruthfulQA, ARC). Using a paired comparison design with 50 tasks per model, we tested two models—GPT-4o-mini (OpenAI API) and Qwen3.5-2B (LM Studio local)—and found that the expert identity treatment yields a statistically significant improvement of +3.16 points for GPT-4o-mini (p=0.0419, Cohen's d=0.311) but a non-significant +3.35 point improvement for Qwen3.5-2B (p=0.2597, Cohen's d=0.193). These results suggest that expert prompting provides modest but measurable gains for larger API-accessible models, while smaller local models show inconsistent response quality including null outputs.

---

## Key Results

| Model         | Baseline | Treatment | Delta | Cohen's d | p-value | Significant |
|---------------|----------|-----------|-------|-----------|---------|-------------|
| gpt-4o-mini   | 76.40    | 79.56     | +3.16 | 0.311     | 0.0419  | YES         |
| qwen3.5-2b    | 44.46    | 47.92     | +3.35 | 0.193     | 0.2597  | NO          |

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/your-org/expert-prompting-eval.git
cd expert-prompting-eval

# Install dependencies
pip install -r requirements.txt

# Run baseline benchmark (no expert identity)
python -m src.benchmark run --model gpt-4o-mini --output results/baseline_gpt.json

# Run treatment benchmark (with expert identity)
python -m src.benchmark run --model gpt-4o-mini --treatment --output results/treatment_gpt.json

# Compare results
python -m src.analysis compare --baseline results/baseline_gpt.json --treatment results/treatment_gpt.json

# For local model (LM Studio must be running)
python -m src.benchmark run --model qwen3.5-2b --provider lmstudio --treatment
```

---

## Architecture

```
+------------------+
|   User Query     |
|  "Solve this     |
|   math problem"  |
+--------+---------+
         |
         v
+------------------+
|  PROMPT BUILDER  |
|  - task domain   |
|  - base prompt   |
|  - expert ID?    |
+--------+---------+
         |
         v
+------------------+     +-------------------+
|    MODEL RUNNER  |---->|  OpenAI API       |
|                  |     |  or LM Studio     |
|  - temperature   |     +-------------------+
|  - max_tokens    |
|  - retry logic   |
+--------+---------+
         |
         v
+------------------+
|  OUTPUT VALIDATOR|
|  - required secs |
|  - null check    |
+--------+---------+
         |
         v
+------------------+
|  SCORER          |
|  - domain rubrics|
|  - dimension wt  |
+--------+---------+
         |
         v
+------------------+
|  RESULTS STORE   |
|  - per-domain    |
|  - per-model     |
|  - statistics    |
+--------+---------+
         |
         v
+------------------+
|  ANALYSIS SUITE  |
|  - paired t-test |
|  - Cohen's d     |
|  - significance  |
+------------------+
```

---

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@software{expert_prompting_eval,
  title = {Expert Prompting Benchmark: Does task-specific expert identity improve sub-agent outputs?},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/your-org/expert-prompting-eval}
}
```

---

## License

MIT License

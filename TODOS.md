# Expert Prompting Benchmark — TODOS

## Project Status

| # | Task | Status |
|---|------|--------|
| 1 | Create project folder + SPEC.md | ✅ Done |
| 2 | Build src/benchmark_tasks.py — 5 domain task templates | ⏳ Pending |
| 3 | Build src/scorer.py — quality scoring rubric | ⏳ Pending |
| 4 | Build src/statistics.py — paired t-test + effect size | ⏳ Pending |
| 5 | Build src/results.py — structured JSON recording + export | ⏳ Pending |
| 6 | Build src/experimenter.py — experimenter agent | ⏳ Pending |
| 7 | Pilot: single domain, 2 tasks, verify scoring | ⏳ Pending |
| 8 | Full experiment: 5 domains × 10 tasks = 50 paired comparisons | ⏳ Pending |
| 9 | Write whitepaper/ + PR package/ | ⏳ Pending |
| 10 | Create shareable skill/expert-prompting-benchmark/ | ⏳ Pending |
| 11 | Push to GitHub 0-CYBERDYNE-SYSTEMS-0/expert-prompting-eval | ⏳ Pending |

## Running the Experiment

```bash
# From ~/expert-prompting-eval/
cd ~/expert-prompting-eval/
python3 -m src.experimenter
```

## Verification Steps

- [ ] LM Studio server running on localhost:1234
- [ ] Target model loaded: qwen3.5-2b-claude-4.6-opus-reasoning-distilled
- [ ] All source files pass syntax check
- [ ] Pilot completes without error
- [ ] Results written to results/run_*/experiment1_direct.json
- [ ] Whitepaper + PR package written
- [ ] Skill package created and installable
- [ ] Repo pushed to GitHub

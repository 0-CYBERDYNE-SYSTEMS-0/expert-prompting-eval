# Expert Prompting Evaluation

Benchmark: Does expert prompting improve sub-agent outputs?

Based on [arXiv:2305.14688](https://arxiv.org/abs/2305.14688), this project evaluates whether composing a task-specific expert identity prompt into a sub-agent task produces measurably better outputs than the same task without expert framing.

## What We're Testing

**Route A (baseline):** Sub-agent spawned with task prompt only, no expert identity.  
**Route B (treatment):** Sub-agent spawned with task prompt prefixed with expert identity context.

Tested using a distilled reasoning model (Qwen3.5-2B) via LM Studio on Apple Silicon.

## Structure

```
├── SPEC.md              # Full experiment specification
├── src/                 # Evaluation source code
├── results/             # Benchmark results
├── pr_package/          # PR-ready writeup
├── whitepaper/          # Detailed analysis
└── probe_*.py           # Probing scripts
```

## License

MIT

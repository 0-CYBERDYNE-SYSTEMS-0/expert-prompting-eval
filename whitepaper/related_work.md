# Related Work

## Expert Prompting

The ExpertPrompting framework was introduced by arXiv:2305.14688, which proposed that instructing large language models to answer as distinguished experts—via a composed identity description—can substantially improve response quality on knowledge-intensive benchmarks. The key insight is that LLMs are capable of more sophisticated reasoning when primed with a specific epistemic role, effectively leveraging in-context knowledge that the model already possesses but may not spontaneously activate.

Our work builds directly on this framework, adopting the core technique of composing a task-specific expert identity into the prompt. However, our evaluation differs in three respects: we use a paired comparison design rather than absolute benchmark scoring, we evaluate on sub-agent task outputs rather than end-user queries, and we explicitly study the interaction between expert prompting and model capacity by testing across two models of different scales.

## LLM Benchmark Evals

Standardized benchmarks for evaluating large language model capabilities have proliferated rapidly. MMLU (Massive Multitask Language Understanding) measures broad knowledge across 57 domains (CITATION NEEDED). GSM8K evaluates grade-school mathematical reasoning (CITATION NEEDED). HellaSwag tests common-sense situational reasoning (CITATION NEEDED). TruthfulQA measures propensity to generate correct versus misleading answers (CITATION NEEDED). ARC (AI2 Reasoning Challenge) evaluates scientific domain reasoning (CITATION NEEDED).

Our work does not propose new benchmarks but rather uses these established domains to evaluate a prompting technique. We adopt the standard practice of sampling tasks from each domain and structuring them into a common prompt format, ensuring comparability across domains while retaining domain-specific difficulty.

## Sub-Agent Orchestration

A growing body of work explores multi-agent LLM systems in which specialized sub-agents are composed to solve complex tasks. The orchestration pattern—wherein a supervising agent spawns task-specific sub-agents—has been applied to code generation, research synthesis, and automated experiment design. The key challenge in these systems is ensuring that sub-agent outputs are reliably high-quality, which motivates our investigation of prompting techniques that can be applied at the sub-agent level.

Prior work on sub-agent quality has focused largely on architectural solutions: chain-of-thought prompting, self-consistency, and tool use. Relatively little attention has been given to the orthogonal question of whether role-framing (i.e., expert identity) can systematically improve sub-agent output quality. Our work addresses this gap.

## Statistical Methods for Paired Comparison

Paired comparison designs are standard in LLM evaluation (see HELM, CITATION NEEDED) and in the broader experimental psychology literature. The paired t-test with Cohen's d provides a principled way to estimate both the magnitude and the statistical reliability of a treatment effect, which we adopt here. We additionally report null response rates as a proxy for output validity, following conventions in the LLM benchmarking literature for handling invalid or truncated outputs (CITATION NEEDED).

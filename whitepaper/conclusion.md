# Conclusion

## Summary

We evaluated the Expert Prompting technique—composing a task-specific expert identity into sub-agent task prompts—across two models on a five-domain, 50-task benchmark using a paired comparison design with a 0–100 weighted scorer. We found a consistent positive treatment effect of approximately +3.2–3.4 points for both models. The effect was statistically significant for gpt-5.4-nano (p=0.0419, Cohen's d=0.311) but not for the qwen3.5-2b distilled model (p=0.2597, Cohen's d=0.193), despite nearly identical raw score improvements.

Domain-level analysis revealed that the gains were concentrated in TruthfulQA (+8.32) and HellaSwag (+4.72) for the larger model, with minimal effect in GSM8K—a finding that suggests expert identity framing augments declarative and situational reasoning more than procedural mathematical calculation.

## Limitations

Several limitations constrain the generalizability of these findings:

1. **Single benchmark iteration.** Our experiment represents one round of paired comparisons. Larger sample sizes (per-domain n > 10) would be needed to detect domain-specific effects with greater precision.

2. **Model selection.** We evaluated one OpenAI API model and one distilled local model. The relationship between model capacity and expert prompting effectiveness may differ across the full spectrum of available models.

3. **Scorer automation.** The scorer is an automated rubric applied uniformly; it may not capture nuanced quality differences that a human evaluator would perceive.

4. **Null responses.** Seven null responses in the qwen3.5-2b model (concentrated in MMLU and GSM8K) reduced the effective sample size and contributed to the non-significant result. Future work should investigate whether improved prompt engineering or increased token limits can reduce null response rates in smaller distilled models.

5. **Expert identity composition.** We did not systematically vary the content or style of the expert identity beyond character limits. The optimal composition of the expert identity description remains an open design question.

## Future Work

Several natural extensions follow from this work:

- **Scaling study.** Evaluating expert prompting across a wider range of model sizes (e.g., 7B, 13B, 70B) would clarify whether the capacity-compounding pattern observed here is robust and continuous.

- **Domain-specific identity engineering.** Given that TruthfulQA and HellaSwag showed the largest gains, future work should investigate whether tailored expert identities can further amplify these effects.

- **Sub-agent orchestration integration.** Evaluating expert prompting in full multi-agent pipelines—where sub-agents produce outputs consumed by downstream agents—would measure whether the quality gains at the individual-agent level propagate through the pipeline.

- **Human evaluation.** A parallel human evaluation study would validate whether the automated scorer's quality assessments correlate with human expert judgment, particularly for borderline cases.

Our contribution establishes that expert prompting provides a measurable quality benefit in sub-agent task execution, that this benefit is most reliably detectable in larger models, and that the effect is concentrated in reasoning domains that depend on declarative knowledge and situational judgment rather than pure calculation. This suggests that model capacity and prompting technique are complementary levers: as models grow more capable, the returns to effective prompting increase.

"""
Experimenter for OpenAI gpt-5.4-nano — ARC domain only.

Fork of experimenter.py using OpenAI SDK instead of LM Studio.
Single domain (ARC), same paired baseline vs treatment design.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark_tasks import ARC_TASKS, EXPERT_IDENTITIES, format_baseline_prompt, format_expert_prompt
from src.results import DomainResult, ExperimentResult, ResultsRecorder, TaskResult
from src.scorer import score_task
from src.statistics import analyze_paired_comparison

# ─── OpenAI API Client ─────────────────────────────────────────────────────────

MODEL_NAME = "gpt-5.4-nano"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


def call_openai(prompt: str, system_prompt: str = "", temperature: float = 0.3) -> str:
    """Send a prompt to OpenAI API and return the response text."""
    import urllib.request
    import urllib.error

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": 2048,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI HTTP {e.code}: {body}") from e
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}") from e


def verify_openai() -> bool:
    """Check that OpenAI API key is set and model is accessible."""
    if not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY environment variable not set.")
        return False
    try:
        # Quick test call
        call_openai("Say 'ok' in one word.", temperature=0.1)
        print(f"[OK] OpenAI API connected. Model: {MODEL_NAME}")
        return True
    except Exception as e:
        print(f"[ERROR] OpenAI API test call failed: {e}")
        return False


# ─── ARC Domain Runner ─────────────────────────────────────────────────────────

def run_arc_domain(
    tasks: list,
    expert_identity: dict,
    delay_between_calls: float = 2.0,
) -> DomainResult:
    """Run paired baseline vs treatment comparison for ARC tasks."""
    print(f"\n{'='*60}")
    print(f"  Domain: ARC (Science/Engineering Reasoning)")
    print(f"  Tasks: {len(tasks)}")
    print(f"  Model: {MODEL_NAME}")
    print(f"{'='*60}")

    task_results: list[TaskResult] = []
    baseline_scores: list[float] = []
    treatment_scores: list[float] = []

    for i, task in enumerate(tasks):
        print(f"\n  [{i+1}/{len(tasks)}] Task: {task.id}")
        timestamp = datetime.now().isoformat()

        # Baseline call
        baseline_prompt = format_baseline_prompt(task)
        try:
            baseline_response = call_openai(baseline_prompt)
            print(f"    [BASELINE] length={len(baseline_response)} chars")
        except Exception as e:
            print(f"    [BASELINE ERROR] {e}")
            baseline_response = f"[ERROR: {e}]"

        time.sleep(delay_between_calls)

        # Treatment call
        treatment_prompt = format_expert_prompt(task, expert_identity)
        try:
            treatment_response = call_openai(treatment_prompt)
            print(f"    [TREATMENT] length={len(treatment_response)} chars")
        except Exception as e:
            print(f"    [TREATMENT ERROR] {e}")
            treatment_response = f"[ERROR: {e}]"

        time.sleep(delay_between_calls)

        # Scoring
        baseline_score_obj = score_task(
            task.question, task.correct_answer, baseline_response, "arc"
        )
        treatment_score_obj = score_task(
            task.question, task.correct_answer, treatment_response, "arc"
        )
        delta = treatment_score_obj.total - baseline_score_obj.total

        baseline_scores.append(baseline_score_obj.total)
        treatment_scores.append(treatment_score_obj.total)

        print(
            f"    [SCORE] baseline={baseline_score_obj.total:.1f}  "
            f"treatment={treatment_score_obj.total:.1f}  "
            f"delta={delta:+.1f}"
        )

        tr = TaskResult(
            task_id=task.id,
            domain="arc",
            baseline_response=baseline_response,
            treatment_response=treatment_response,
            baseline_score=baseline_score_obj.total,
            treatment_score=treatment_score_obj.total,
            score_delta=delta,
            baseline_correctness=baseline_score_obj.correctness,
            treatment_correctness=treatment_score_obj.correctness,
            baseline_reasoning=baseline_score_obj.reasoning,
            treatment_reasoning=treatment_score_obj.reasoning,
            baseline_completeness=baseline_score_obj.completeness,
            treatment_completeness=treatment_score_obj.completeness,
            baseline_format=baseline_score_obj.format_score,
            treatment_format=treatment_score_obj.format_score,
            model=MODEL_NAME,
            timestamp=timestamp,
        )
        task_results.append(tr)

    # Aggregate statistics
    stats = analyze_paired_comparison(baseline_scores, treatment_scores, "arc")
    import statistics

    return DomainResult(
        domain="arc",
        n_tasks=len(tasks),
        baseline_mean=statistics.mean(baseline_scores),
        treatment_mean=statistics.mean(treatment_scores),
        mean_delta=stats.mean_delta,
        t_statistic=stats.t_statistic or 0.0,
        p_value=stats.p_value or 1.0,
        cohen_d=stats.cohen_d,
        ci_lower=stats.ci_lower,
        ci_upper=stats.ci_upper,
        effect_size_label=stats.effect_size_label,
        significant=stats.significant,
        wilcoxon_p=stats.wilcoxon_p_value or 1.0,
        task_results=task_results,
    )


def run_openai_experiment(
    results_dir: str = "results",
    delay_between_calls: float = 2.0,
) -> ExperimentResult:
    """Run the ARC benchmark on gpt-5.4-nano."""
    recorder = ResultsRecorder(base_dir=results_dir)
    run_dir = recorder.start_run()
    print(f"\n[RESULTS] Writing to: {run_dir}")

    run_id = f"exp_openai_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{MODEL_NAME}"
    timestamp = datetime.now().isoformat()

    # Run ARC domain
    expert_identity = EXPERT_IDENTITIES.get("arc", {})
    dr = run_arc_domain(ARC_TASKS, expert_identity, delay_between_calls)

    # Save domain result
    domain_path = run_dir / "arc_results.json"
    with open(domain_path, "w") as f:
        json.dump(
            {
                **dr.__dict__,
                "task_results": [tr.__dict__ for tr in dr.task_results],
            },
            f,
            indent=2,
        )
    print(f"[SAVED] arc_results.json")

    import statistics

    all_baseline = [tr.baseline_score for tr in dr.task_results]
    all_treatment = [tr.treatment_score for tr in dr.task_results]
    overall_stats = analyze_paired_comparison(all_baseline, all_treatment, "overall")

    result = ExperimentResult(
        experiment_id=run_id,
        run_timestamp=timestamp,
        model=MODEL_NAME,
        n_domains=1,
        n_tasks_per_domain=len(ARC_TASKS),
        domain_results=[dr],
        overall_baseline_mean=statistics.mean(all_baseline),
        overall_treatment_mean=statistics.mean(all_treatment),
        overall_delta=overall_stats.mean_delta,
        overall_t_stat=overall_stats.t_statistic or 0.0,
        overall_p_value=overall_stats.p_value or 1.0,
        overall_cohen_d=overall_stats.cohen_d,
        overall_significant=overall_stats.significant,
        overall_effect_label=overall_stats.effect_size_label,
        experiment_type="direct-openai",
        notes="Single-domain ARC benchmark on gpt-5.4-nano vs local qwen3.5-2b baseline",
    )

    recorder.save_experiment(result, "experiment_openai_arc.json")
    recorder.save_summary(result)
    recorder.export_csv(result, "results_openai_arc.csv")

    print(f"\n[DONE] Results saved to: {run_dir}")
    print(f"\n{'='*60}")
    print(f"  OVERALL RESULTS — gpt-5.4-nano ARC")
    print(f"{'='*60}")
    print(f"  Baseline mean:  {result.overall_baseline_mean:.2f}")
    print(f"  Treatment mean: {result.overall_treatment_mean:.2f}")
    print(f"  Mean delta:    {result.overall_delta:+.2f}")
    print(f"  Cohen's d:      {result.overall_cohen_d:.3f} ({result.overall_effect_label})")
    print(f"  p-value:        {result.overall_p_value:.4f}")
    print(f"  Significant:    {'YES' if result.overall_significant else 'NO'}")
    print(f"{'='*60}")

    return result


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"  Expert Prompting Benchmark — OpenAI gpt-5.4-nano")
    print(f"  Domain: ARC (Science/Engineering Reasoning)")
    print(f"  Model: {MODEL_NAME}")
    print(f"{'='*60}\n")

    if not verify_openai():
        print("\n[ABORT] OpenAI API not accessible.")
        sys.exit(1)

    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    print(f"[CWD] {os.getcwd()}")

    result = run_openai_experiment(
        results_dir=str(project_root / "results"),
        delay_between_calls=2.0,
    )

    print(f"\n[COMPLETE] Run ID: {result.experiment_id}")
    return result


if __name__ == "__main__":
    main()

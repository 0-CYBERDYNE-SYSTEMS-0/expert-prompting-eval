"""
Experimenter — orchestrates the expert prompting benchmark.

This is the main entry point. It:
  1. Loads all benchmark tasks
  2. For each task, sends both baseline and treatment prompts to LM Studio
  3. Scores outputs using the rubric
  4. Computes paired statistics
  5. Records structured results to disk

Usage:
    python3 -m src.experimenter
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# Add parent dir to path so src modules can import each other
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark_tasks import (
    DOMAIN_LABELS,
    DOMAINS,
    format_baseline_prompt,
    format_expert_prompt,
    get_all_tasks,
)
from src.results import (
    DomainResult,
    ExperimentResult,
    TaskResult,
)
from src.scorer import score_delta, score_task
from src.statistics import analyze_paired_comparison


# ─── LM Studio API Client ──────────────────────────────────────────────────────

LM_STUDIO_URL = os.environ.get("LMS_API_URL", "http://localhost:1234")
MODEL_NAME = "qwen3.5-2b-claude-4.6-opus-reasoning-distilled"
TIMEOUT_SECONDS = 120


def call_model(prompt: str, system_prompt: str = "", temperature: float = 0.3) -> str:
    """
    Send a prompt to LM Studio and return the response text.
    Uses the OpenAI-compatible /chat/completions endpoint.
    """
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
        "max_tokens": 1024,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{LM_STUDIO_URL}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LM Studio HTTP {e.code}: {body}") from e
    except Exception as e:
        raise RuntimeError(f"LM Studio call failed: {e}") from e


def verify_lm_studio() -> bool:
    """Check that LM Studio is reachable and target model is available."""
    import urllib.request
    try:
        req = urllib.request.Request(
            f"{LM_STUDIO_URL}/v1/models",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            models = json.loads(resp.read().decode("utf-8"))
            available = [m["id"] for m in models.get("data", [])]
            if MODEL_NAME not in available:
                print(f"[WARN] Target model '{MODEL_NAME}' not in available models: {available}")
                return False
            print(f"[OK] LM Studio connected. Model '{MODEL_NAME}' available.")
            return True
    except Exception as e:
        print(f"[ERROR] Cannot reach LM Studio at {LM_STUDIO_URL}: {e}")
        return False


# ─── Core Experiment ──────────────────────────────────────────────────────────

def run_domain(
    domain: str,
    tasks: list,
    expert_identities: dict,
    delay_between_calls: float = 1.0,
) -> DomainResult:
    """
    Run paired baseline vs treatment comparison for all tasks in a domain.

    Returns a DomainResult with per-task and aggregated statistics.
    """
    print(f"\n{'='*60}")
    print(f"  Domain: {domain.upper()} ({DOMAIN_LABELS.get(domain, domain)})")
    print(f"  Tasks: {len(tasks)}")
    print(f"{'='*60}")

    task_results: list[TaskResult] = []
    baseline_scores: list[float] = []
    treatment_scores: list[float] = []

    for i, task in enumerate(tasks):
        print(f"\n  [{i+1}/{len(tasks)}] Task: {task.id}")
        timestamp = datetime.now().isoformat()

        # ── Baseline call ───────────────────────────────────────────────────
        baseline_prompt = format_baseline_prompt(task)
        try:
            baseline_response = call_model(baseline_prompt)
            print(f"    [BASELINE] length={len(baseline_response)} chars")
        except Exception as e:
            print(f"    [BASELINE ERROR] {e}")
            baseline_response = f"[ERROR: {e}]"

        time.sleep(delay_between_calls)

        # ── Treatment call ─────────────────────────────────────────────────
        expert_identity = expert_identities.get(domain, {})
        treatment_prompt = format_expert_prompt(task, expert_identity)
        try:
            treatment_response = call_model(treatment_prompt)
            print(f"    [TREATMENT] length={len(treatment_response)} chars")
        except Exception as e:
            print(f"    [TREATMENT ERROR] {e}")
            treatment_response = f"[ERROR: {e}]"

        time.sleep(delay_between_calls)

        # ── Scoring ─────────────────────────────────────────────────────────
        baseline_score_obj = score_task(
            task.question, task.correct_answer, baseline_response, domain
        )
        treatment_score_obj = score_task(
            task.question, task.correct_answer, treatment_response, domain
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
            domain=domain,
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

    # ── Aggregate statistics for domain ─────────────────────────────────────
    stats = analyze_paired_comparison(baseline_scores, treatment_scores, domain)
    import statistics

    return DomainResult(
        domain=domain,
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


def run_full_experiment(
    results_dir: str = "results",
    tasks_per_domain: int = 10,
    delay_between_calls: float = 1.0,
) -> ExperimentResult:
    """
    Run the complete expert prompting benchmark experiment.

    Returns an ExperimentResult and also writes results to disk.
    """
    from src.results import ResultsRecorder

    recorder = ResultsRecorder(base_dir=results_dir)
    run_dir = recorder.start_run()
    print(f"\n[RESULTS] Writing to: {run_dir}")

    run_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{MODEL_NAME[:20]}"
    timestamp = datetime.now().isoformat()

    # Import expert identities from benchmark_tasks
    from src.benchmark_tasks import EXPERT_IDENTITIES

    domain_results: list[DomainResult] = []
    all_baseline: list[float] = []
    all_treatment: list[float] = []

    for domain_name, tasks in DOMAINS.items():
        domain_tasks = tasks[:tasks_per_domain]
        dr = run_domain(domain_name, domain_tasks, EXPERT_IDENTITIES, delay_between_calls)
        domain_results.append(dr)
        all_baseline.extend([tr.baseline_score for tr in dr.task_results])
        all_treatment.extend([tr.treatment_score for tr in dr.task_results])

        # Save intermediate domain result
        domain_path = run_dir / f"{domain_name}_results.json"
        with open(domain_path, "w") as f:
            json.dump(asdict(dr), f, indent=2)
        print(f"[SAVED] {domain_name}_results.json")

    # ── Overall statistics ─────────────────────────────────────────────────
    import statistics

    overall_stats = analyze_paired_comparison(all_baseline, all_treatment, "overall")

    result = ExperimentResult(
        experiment_id=run_id,
        run_timestamp=timestamp,
        model=MODEL_NAME,
        n_domains=len(DOMAINS),
        n_tasks_per_domain=tasks_per_domain,
        domain_results=domain_results,
        overall_baseline_mean=statistics.mean(all_baseline),
        overall_treatment_mean=statistics.mean(all_treatment),
        overall_delta=overall_stats.mean_delta,
        overall_t_stat=overall_stats.t_statistic or 0.0,
        overall_p_value=overall_stats.p_value or 1.0,
        overall_cohen_d=overall_stats.cohen_d,
        overall_significant=overall_stats.significant,
        overall_effect_label=overall_stats.effect_size_label,
        experiment_type="direct",
        notes="",
    )

    # Save experiment results
    recorder.save_experiment(result, "experiment1_direct.json")
    recorder.save_summary(result)
    recorder.export_csv(result, "results.csv")

    print(f"\n[DONE] Results saved to: {run_dir}")
    print(f"\n{'='*60}")
    print(f"  OVERALL RESULTS")
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
    print(f"  Expert Prompting Benchmark")
    print(f"  Model: {MODEL_NAME}")
    print(f"  LM Studio: {LM_STUDIO_URL}")
    print(f"{'='*60}\n")

    # Verify LM Studio
    if not verify_lm_studio():
        print("\n[ABORT] LM Studio not available. Start LM Studio and load the model.")
        sys.exit(1)

    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    print(f"[CWD] {os.getcwd()}")

    # Run experiment
    result = run_full_experiment(
        results_dir=str(project_root / "results"),
        tasks_per_domain=10,
        delay_between_calls=1.0,
    )

    print(f"\n[COMPLETE] Run ID: {result.experiment_id}")
    return result


if __name__ == "__main__":
    main()

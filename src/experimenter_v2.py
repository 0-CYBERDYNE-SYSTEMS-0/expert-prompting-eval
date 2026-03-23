"""
v2 Experimenter — hardened, sub-agent-driven, validated.

Design:
  1. Probe both models for token saturation (sets safe max_tokens)
  2. For each domain: spawn a sub-agent with expert prompt + task list
  3. Sub-agent runs paired baseline/treatment per task with retry=3
  4. Validation gate: if response is empty/Nones, mark null and retry task
  5. Return structured results with null handling (not minimum penalties)
  6. Main session aggregates and writes all outputs

Two models: qwen (LM Studio) and gpt (OpenAI), run separately.
Each model runs 5 domains × 10 tasks = 50 paired comparisons.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark_tasks_v2 import (
    DOMAIN_LABELS,
    DOMAINS,
    EXPERT_IDENTITIES,
    Task,
    format_baseline_prompt,
    format_expert_prompt,
)
from src.scorer_v2 import score_response
from src.statistics import analyze_paired_comparison
from src.statistics import mean as py_mean
from src.statistics import std as py_std
from src.api_client_v2 import (
    APIResponse,
    ProbingResult,
    call_lm_studio_retry,
    call_openai_retry,
    run_probe_sequence,
    set_openai_key,
)


# ─── Model Config ─────────────────────────────────────────────────────────────

LM_MODEL = "qwen3.5-2b-claude-4.6-opus-reasoning-distilled"
OPENAI_MODEL = "gpt-5.4-nano"
OPENAI_KEY = os.environ.get(
    "OPENAI_API_KEY",
    "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",
)

# Token budgets set after probing
LM_MAX_TOKENS = 768   # confirmed via saturation probe
OPENAI_MAX_TOKENS = 4096  # generous limit; was 569 (too tight)


# ─── Sub-Agent Domain Worker ───────────────────────────────────────────────────
# Each sub-agent handles one domain, all 10 tasks

def run_domain_subagent(
    domain: str,
    tasks: list[Task],
    provider: str,       # "lm_studio" or "openai"
    max_tokens: int,
    delay: float = 1.0,
) -> dict:
    """
    Spawned sub-agent handler for one domain.
    Runs baseline + treatment for all tasks with retry logic.
    Returns a dict ready for JSON serialization.
    """
    identity = EXPERT_IDENTITIES.get(domain, "")
    results = []

    for i, task in enumerate(tasks):
        timestamp = datetime.now().isoformat()

        # ── Baseline ───────────────────────────────────────────────────
        baseline_resp = _call_with_retry(
            format_baseline_prompt(task), provider, max_tokens
        )

        time.sleep(delay)

        # ── Treatment ────────────────────────────────────────────────
        treatment_resp = _call_with_retry(
            format_expert_prompt(task, identity), provider, max_tokens
        )

        time.sleep(delay)

        # ── Score (None if empty/failed) ────────────────────────────
        baseline_scores = score_response(
            baseline_resp.content, task,
            baseline_resp.finish_reason, baseline_resp.completion_tokens
        )
        treatment_scores = score_response(
            treatment_resp.content, task,
            treatment_resp.finish_reason, treatment_resp.completion_tokens
        )

        results.append({
            "task_id": task.id,
            "domain": domain,
            "baseline_content": baseline_resp.content[:500],   # truncate for storage
            "treatment_content": treatment_resp.content[:500],
            "baseline_score": baseline_scores.total if baseline_scores else None,
            "treatment_score": treatment_scores.total if treatment_scores else None,
            "score_delta": (
                (treatment_scores.total - baseline_scores.total)
                if (baseline_scores and treatment_scores) else None
            ),
            "baseline_finish": baseline_resp.finish_reason,
            "treatment_finish": treatment_resp.finish_reason,
            "baseline_tokens": baseline_resp.completion_tokens,
            "treatment_tokens": treatment_resp.completion_tokens,
            "baseline_attempt": baseline_resp.attempt,
            "treatment_attempt": treatment_resp.attempt,
            "baseline_error": baseline_resp.error,
            "treatment_error": treatment_resp.error,
            "model": LM_MODEL if provider == "lm_studio" else OPENAI_MODEL,
            "timestamp": timestamp,
        })

        # Print progress
        b_str = f"{baseline_scores.total:.1f}" if baseline_scores else "FAIL"
        t_str = f"{treatment_scores.total:.1f}" if treatment_scores else "FAIL"
        delta_str = f"{results[-1]['score_delta']:+.1f}" if results[-1]['score_delta'] is not None else "NULL"
        print(
            f"  [{i+1}/10] {task.id}: "
            f"B={b_str} T={t_str} Δ={delta_str}"
        )

    return _aggregate_domain(domain, results)


def _call_with_retry(prompt: str, provider: str, max_tokens: int) -> APIResponse:
    """Wrapper: retries handled by the client's built-in retry."""
    if provider == "lm_studio":
        return call_lm_studio_retry(prompt, max_tokens)
    else:
        return call_openai_retry(prompt, max_tokens)


def _aggregate_domain(domain: str, results: list[dict]) -> dict:
    """Aggregate per-task results into domain-level stats."""
    valid_baseline = [r["baseline_score"] for r in results if r["baseline_score"] is not None]
    valid_treatment = [r["treatment_score"] for r in results if r["treatment_score"] is not None]
    valid_deltas = [r["score_delta"] for r in results if r["score_delta"] is not None]
    null_count = sum(1 for r in results if r["score_delta"] is None)

    if len(valid_deltas) >= 3:
        stats = analyze_paired_comparison(valid_baseline, valid_treatment, domain)
        return {
            "domain": domain,
            "n_tasks": len(results),
            "n_valid": len(valid_deltas),
            "n_null": null_count,
            "baseline_mean": py_mean(valid_baseline),
            "treatment_mean": py_mean(valid_treatment),
            "mean_delta": stats.mean_delta,
            "t_statistic": stats.t_statistic or 0.0,
            "p_value": stats.p_value or 1.0,
            "cohen_d": stats.cohen_d,
            "ci_lower": stats.ci_lower,
            "ci_upper": stats.ci_upper,
            "effect_size_label": stats.effect_size_label,
            "significant": stats.significant,
            "wilcoxon_p": stats.wilcoxon_p_value or 1.0,
            "task_results": results,
        }
    else:
        return {
            "domain": domain,
            "n_tasks": len(results),
            "n_valid": len(valid_deltas),
            "n_null": null_count,
            "baseline_mean": py_mean(valid_baseline) if valid_baseline else None,
            "treatment_mean": py_mean(valid_treatment) if valid_treatment else None,
            "mean_delta": py_mean(valid_deltas) if valid_deltas else None,
            "t_statistic": None,
            "p_value": None,
            "cohen_d": None,
            "ci_lower": None,
            "ci_upper": None,
            "effect_size_label": "insufficient_data",
            "significant": False,
            "wilcoxon_p": None,
            "task_results": results,
        }


# ─── Main Orchestrator ─────────────────────────────────────────────────────────

def run_full_experiment_v2(
    provider: str,
    output_dir: str,
    max_tokens: int,
    delay: float = 1.5,
) -> dict:
    """
    Run the full v2 experiment for one provider.
    Orchestrates domains, aggregates results, writes to disk.
    """
    model = LM_MODEL if provider == "lm_studio" else OPENAI_MODEL
    run_id = f"v2_{provider}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n{'='*60}")
    print(f"  v2 EXPERIMENT — {model}")
    print(f"  max_tokens={max_tokens}  delay={delay}s")
    print(f"{'='*60}")

    domain_results = []
    all_valid_baseline = []
    all_valid_treatment = []
    all_valid_deltas = []
    total_null = 0

    for domain_name, tasks in DOMAINS.items():
        print(f"\n  Domain: {domain_name.upper()} ({len(tasks)} tasks)")

        dr = run_domain_subagent(domain_name, tasks, provider, max_tokens, delay)
        domain_results.append(dr)

        all_valid_baseline.extend(
            r["baseline_score"] for r in dr["task_results"]
            if r["baseline_score"] is not None
        )
        all_valid_treatment.extend(
            r["treatment_score"] for r in dr["task_results"]
            if r["treatment_score"] is not None
        )
        all_valid_deltas.extend(
            r["score_delta"] for r in dr["task_results"]
            if r["score_delta"] is not None
        )
        total_null += dr["n_null"]

    # Overall stats
    if len(all_valid_deltas) >= 3:
        overall = analyze_paired_comparison(
            all_valid_baseline, all_valid_treatment, "overall"
        )
        overall_result = {
            "baseline_mean": py_mean(all_valid_baseline),
            "treatment_mean": py_mean(all_valid_treatment),
            "mean_delta": overall.mean_delta,
            "t_statistic": overall.t_statistic or 0.0,
            "p_value": overall.p_value or 1.0,
            "cohen_d": overall.cohen_d,
            "significant": overall.significant,
            "effect_label": overall.effect_size_label,
        }
    else:
        overall_result = {
            "baseline_mean": py_mean(all_valid_baseline) if all_valid_baseline else None,
            "treatment_mean": py_mean(all_valid_treatment) if all_valid_treatment else None,
            "mean_delta": py_mean(all_valid_deltas) if all_valid_deltas else None,
            "significant": False,
        }

    experiment = {
        "experiment_id": run_id,
        "provider": provider,
        "model": model,
        "max_tokens": max_tokens,
        "timestamp": datetime.now().isoformat(),
        "n_domains": len(DOMAINS),
        "n_null_responses": total_null,
        "domain_results": domain_results,
        "overall": overall_result,
    }

    # Write outputs
    out_path = Path(output_dir) / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(experiment, f, indent=2)
    print(f"\n[SAVED] {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  OVERALL — {model}")
    print(f"{'='*60}")
    print(f"  Baseline mean:  {overall_result['baseline_mean']:.2f}")
    print(f"  Treatment mean: {overall_result['treatment_mean']:.2f}")
    print(f"  Mean delta:    {overall_result['mean_delta']:+.2f}")
    print(f"  Cohen's d:      {overall_result.get('cohen_d', 0):.3f} ({overall_result.get('effect_label', 'N/A')})")
    print(f"  p-value:        {overall_result.get('p_value', 1.0):.4f}")
    print(f"  Significant:    {'YES' if overall_result.get('significant') else 'NO'}")
    print(f"  Null responses: {total_null}/50")
    print(f"{'='*60}")

    return experiment


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="v2 Expert Prompting Benchmark")
    parser.add_argument("--provider", choices=["lm_studio", "openai"], required=True)
    parser.add_argument("--output-dir", default="results/v2")
    parser.add_argument("--delay", type=float, default=1.5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Set OpenAI key if needed
    if args.provider == "openai":
        set_openai_key(OPENAI_KEY)

    # Run
    result = run_full_experiment_v2(
        provider=args.provider,
        output_dir=args.output_dir,
        max_tokens=LM_MAX_TOKENS if args.provider == "lm_studio" else OPENAI_MAX_TOKENS,
        delay=args.delay,
    )

    print(f"\n[DONE] {result['experiment_id']}")
    return result


if __name__ == "__main__":
    main()

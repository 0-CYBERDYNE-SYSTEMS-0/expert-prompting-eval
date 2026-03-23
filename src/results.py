"""
Structured results recording and export for expert prompting benchmark.

Outputs:
  results/run_YYYYMMDD_HHMMSS/
      experiment1_direct.json
      experiment2_meta.json
      summary.json
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Supported export formats
EXPORT_FORMATS = ["json", "markdown", "csv"]


@dataclass
class TaskResult:
    """Result for a single task in the paired comparison."""
    task_id: str
    domain: str
    baseline_response: str
    treatment_response: str
    baseline_score: float
    treatment_score: float
    score_delta: float
    baseline_correctness: float
    treatment_correctness: float
    baseline_reasoning: float
    treatment_reasoning: float
    baseline_completeness: float
    treatment_completeness: float
    baseline_format: float
    treatment_format: float
    model: str = "qwen3.5-2b-claude-4.6-opus-reasoning-distilled"
    timestamp: str = ""


@dataclass
class DomainResult:
    """Aggregated results for a single domain."""
    domain: str
    n_tasks: int
    baseline_mean: float
    treatment_mean: float
    mean_delta: float
    t_statistic: float
    p_value: float
    cohen_d: float
    ci_lower: float
    ci_upper: float
    effect_size_label: str
    significant: bool
    wilcoxon_p: float
    task_results: list[TaskResult] = field(default_factory=list)


@dataclass
class ExperimentResult:
    """Full results from one experiment run."""
    experiment_id: str
    run_timestamp: str
    model: str
    n_domains: int
    n_tasks_per_domain: int
    domain_results: list[DomainResult]
    overall_baseline_mean: float
    overall_treatment_mean: float
    overall_delta: float
    overall_t_stat: float
    overall_p_value: float
    overall_cohen_d: float
    overall_significant: bool
    overall_effect_label: str
    experiment_type: str = "direct"  # "direct" or "meta"
    notes: str = ""


class ResultsRecorder:
    """Records and exports experiment results."""

    def __init__(self, base_dir: str = "results"):
        self.base_dir = Path(base_dir)
        self.run_dir: Optional[Path] = None

    def start_run(self) -> Path:
        """Create a new run directory with timestamp."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_dir / f"run_{ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir

    def save_experiment(
        self,
        result: ExperimentResult,
        filename: str = "experiment.json",
    ) -> Path:
        """Save an experiment result to JSON."""
        if self.run_dir is None:
            self.start_run()
        path = self.run_dir / filename
        with open(path, "w") as f:
            json.dump(asdict(result), f, indent=2)
        return path

    def save_summary(self, result: ExperimentResult) -> Path:
        """Save a human-readable summary markdown file."""
        if self.run_dir is None:
            self.start_run()
        path = self.run_dir / "summary.md"

        lines = [
            f"# Expert Prompting Benchmark — Run Summary",
            f"",
            f"**Experiment ID:** {result.experiment_id}",
            f"**Timestamp:** {result.run_timestamp}",
            f"**Model:** {result.model}",
            f"**Type:** {result.experiment_type}",
            f"",
            f"## Overall Results",
            f"",
            f"| Metric | Value |",
            f"|---------|-------|",
            f"| Baseline Mean | {result.overall_baseline_mean:.2f} |",
            f"| Treatment Mean | {result.overall_treatment_mean:.2f} |",
            f"| Mean Delta | {result.overall_delta:+.2f} |",
            f"| Cohen's d | {result.overall_cohen_d:.3f} |",
            f"| Effect Size | {result.overall_effect_label} |",
            f"| p-value | {result.overall_p_value:.4f} |",
            f"| Significant (p<0.05) | {'Yes' if result.overall_significant else 'No'} |",
            f"",
            f"## Per-Domain Results",
            f"",
        ]

        for dr in result.domain_results:
            lines.extend([
                f"### {dr.domain.upper()}",
                f"",
                f"| Metric | Value |",
                f"|---------|-------|",
                f"| Tasks | {dr.n_tasks} |",
                f"| Baseline | {dr.baseline_mean:.2f} |",
                f"| Treatment | {dr.treatment_mean:.2f} |",
                f"| Delta | {dr.mean_delta:+.2f} |",
                f"| Cohen's d | {dr.cohen_d:.3f} |",
                f"| p-value | {dr.p_value:.4f} |",
                f"| Effect | {dr.effect_size_label} |",
                f"| Significant | {'Yes' if dr.significant else 'No'} |",
                f"",
            ])

        if result.notes:
            lines.extend(["## Notes", "", result.notes, ""])

        with open(path, "w") as f:
            f.write("\n".join(lines))

        return path

    def export_csv(self, result: ExperimentResult, filename: str = "results.csv") -> Path:
        """Export task-level results to CSV."""
        if self.run_dir is None:
            self.start_run()
        path = self.run_dir / filename

        import csv
        rows = []
        for dr in result.domain_results:
            for tr in dr.task_results:
                rows.append({
                    "domain": tr.domain,
                    "task_id": tr.task_id,
                    "baseline_score": tr.baseline_score,
                    "treatment_score": tr.treatment_score,
                    "delta": tr.score_delta,
                    "baseline_correctness": tr.baseline_correctness,
                    "treatment_correctness": tr.treatment_correctness,
                    "baseline_reasoning": tr.baseline_reasoning,
                    "treatment_reasoning": tr.treatment_reasoning,
                    "baseline_completeness": tr.baseline_completeness,
                    "treatment_completeness": tr.treatment_completeness,
                    "baseline_format": tr.baseline_format,
                    "treatment_format": tr.treatment_format,
                    "model": tr.model,
                })

        if rows:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        return path

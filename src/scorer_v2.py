"""
v2 scorer — improved rubric using correct_keywords.

Changes from v1:
  - Uses Task.correct_keywords instead of correct_answer
  - Proper null/empty handling (None, not minimum score)
  - Records truncation detection
"""

from dataclasses import dataclass
from src.benchmark_tasks import Task


@dataclass
class Scores:
    correctness: float
    reasoning: float
    completeness: float
    format_score: float

    @property
    def total(self) -> float:
        return (
            self.correctness * 0.40
            + self.reasoning * 0.25
            + self.completeness * 0.20
            + self.format_score * 0.15
        )

    def to_dict(self) -> dict:
        return {
            "correctness": self.correctness,
            "reasoning": self.reasoning,
            "completeness": self.completeness,
            "format": self.format_score,
            "total": self.total,
        }


def score_response(
    response: str,
    task: Task,
    finish_reason: str = "stop",
    completion_tokens: int = 0,
) -> Scores | None:
    """
    Score a model response. Returns None if response is genuinely empty/failed.
    """
    if not response or len(response.strip()) == 0:
        return None

    text = response.strip()
    lower = text.lower()

    # ── Correctness (0-100) ──────────────────────────────────────────────
    keyword_hits = sum(1 for kw in task.correct_keywords if kw.lower() in lower)
    keyword_ratio = min(1.0, keyword_hits / max(len(task.correct_keywords), 1))

    was_truncated = finish_reason == "length"
    truncation_penalty = 0.5 if was_truncated else 1.0

    if keyword_ratio >= 0.6:
        correctness = min(90.0, 55.0 + keyword_ratio * 40.0) * truncation_penalty
    elif keyword_ratio >= 0.3:
        correctness = min(65.0, 35.0 + keyword_ratio * 30.0) * truncation_penalty
    elif keyword_ratio > 0:
        correctness = max(25.0, 20.0 + keyword_ratio * 20.0) * truncation_penalty
    else:
        correctness = 20.0 * truncation_penalty

    # ── Reasoning (0-100) ─────────────────────────────────────────────────
    reasoning_markers = [
        "because", "therefore", "thus", "hence", "step",
        "first", "second", "furthermore", "however", "consequently",
        "analysis", "evaluate", "conclusion",
    ]
    r_hits = sum(1 for m in reasoning_markers if m in lower)
    reasoning = min(100.0, 35.0 + r_hits * 8.0)

    # ── Completeness (0-100) ─────────────────────────────────────────────
    checks = [
        any(w in lower for w in ["confidence", "certain", "likely", "probably", "%"]),
        any(w in lower for w in ["assume", "assumption", "provided", "given"]),
        any(w in lower for w in ["however", "limitation", "uncertainty", "caveat"]),
        len(text.split()) >= 40,
    ]
    completeness = sum(checks) / len(checks) * 100

    # ── Format (0-100) ───────────────────────────────────────────────────
    has_sections = sum(1 for p in ["1.", "2.", "3.", "4."] if p in text) >= 3
    format_checks = [has_sections, len(text.split()) >= 20]
    format_score = sum(format_checks) / len(format_checks) * 100

    return Scores(
        correctness=correctness,
        reasoning=reasoning,
        completeness=completeness,
        format_score=format_score,
    )

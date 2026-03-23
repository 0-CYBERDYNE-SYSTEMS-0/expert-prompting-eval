"""
Quality scoring rubric for expert prompting evaluation.

Outputs are scored 0-100 across 4 dimensions:
  correctness    — Did it get the right answer?         (weight: 0.40)
  reasoning      — Is the explanation sound?            (weight: 0.25)
  completeness   — Were all parts addressed?           (weight: 0.20)
  format         — Did it follow output instructions?   (weight: 0.15)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Scores:
    correctness: float   # 0-100
    reasoning: float    # 0-100
    completeness: float  # 0-100
    format_score: float  # 0-100

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


def check_confidence(answer_text: str) -> bool:
    """Check if the response expresses a confidence level in the output."""
    confidence_indicators = ["confidence", "certain", "sure", "likely", "probably", "possibly", "unsure", "uncertain", "%"]
    return any(ind.lower() in answer_text.lower() for ind in confidence_indicators)


def check_assumptions(answer_text: str) -> bool:
    """Check if the response states assumptions."""
    assumption_indicators = ["assume", "assumption", "given that", "provided that", "if we assume", "assuming"]
    return any(ind.lower() in answer_text.lower() for ind in assumption_indicators)


def check_structured_format(answer_text: str) -> bool:
    """Check if response has the required 4-part structure."""
    required_parts = [
        "1.",  # Answer
        "2.",  # Reasoning
        "3.",  # Confidence
        "4.",  # Assumptions
    ]
    return sum(1 for part in required_parts if part in answer_text) >= 3


def extract_confidence(answer_text: str) -> Optional[float]:
    """Try to extract a numeric confidence percentage from response."""
    import re
    patterns = [
        r"confidence[:\s]+(\d+)%",
        r"(\d+)%\s+confidence",
        r"I am (\d+)%\s+(?:certain|sure|confident)",
        r"confidence level[:\s]+(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, answer_text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def score_task(
    task_question: str,
    correct_answer: str,
    model_response: str,
    task_domain: str,
) -> Scores:
    """
    Score a model response against a task.

    Parameters
    ----------
    task_question : The original task question
    correct_answer : The known correct answer
    model_response : The model's output text
    task_domain    : Benchmark domain (mmlu, gsm8k, hellaswag, truthfulqa, arc)

    Returns
    -------
    Scores dataclass with four dimension scores and a weighted total
    """

    # ── Correctness (0-100) ──────────────────────────────────────────────────
    # For open-ended tasks, do a loose semantic match rather than exact string
    response_lower = model_response.lower()
    answer_lower = correct_answer.lower()

    # Keywords that indicate correct answer
    correct_keywords = {
        "mmlu": ["driven", "foundation", "piles", "mat foundation", "rct", "randomized"],
        "gsm8k": ["$", "miles", "cm", "square", "hours", "minutes", "loaves", "weeks"],
        "hellaswag": ["first", "priority", "should", "recommend", "option"],
        "truthfulqa": ["false", "true", "misleading", "insufficient", "cannot", "not proven"],
        "arc": ["confound", "correlation", "insufficient", "requires", "bias", "error"],
    }

    domain_keywords = correct_keywords.get(task_domain, [])

    # Check if key terms from correct answer appear in response
    keyword_hits = sum(1 for kw in domain_keywords if kw in response_lower)

    # Check for explicit wrong-answer signals
    wrong_signals = ["i don't know", "i cannot", "i'm not sure", "not enough information"]
    has_wrong_signal = any(sig in response_lower for sig in wrong_signals)

    # Numerical answer check for math tasks
    numerical_score = 0.0
    if task_domain == "gsm8k":
        import re
        nums_in_response = set(re.findall(r"[\$]?[\d,]+\.?\d*", model_response))
        nums_in_answer = set(re.findall(r"[\$]?[\d,]+\.?\d*", correct_answer))
        if nums_in_answer and nums_in_response:
            overlap = len(nums_in_response & nums_in_answer) / max(len(nums_in_answer), 1)
            numerical_score = overlap * 100
        elif nums_in_response:
            numerical_score = 30.0  # Has numbers but wrong

    if keyword_hits >= 2 and not has_wrong_signal:
        correctness = min(95.0, 60.0 + numerical_score)
    elif keyword_hits >= 1 and not has_wrong_signal:
        correctness = min(70.0, 40.0 + numerical_score)
    elif has_wrong_signal:
        correctness = max(5.0, numerical_score)
    else:
        correctness = max(20.0, numerical_score)

    # ── Reasoning quality (0-100) ───────────────────────────────────────────
    reasoning_indicators = [
        "because", "therefore", "thus", "hence", "since", "consequently",
        "first", "second", "furthermore", "additionally", "however",
        "on the other hand", "while", "although", "in conclusion",
        "step", "calculat", "analysis", "evaluate",
    ]
    reasoning_hits = sum(1 for ind in reasoning_indicators if ind.lower() in response_lower)
    reasoning_bonus = min(30.0, reasoning_hits * 5.0)

    # Score based on structure and depth
    if check_structured_format(model_response):
        structure_score = 30.0
    else:
        structure_score = 15.0

    reasoning = min(100.0, 40.0 + reasoning_bonus + structure_score)

    # ── Completeness (0-100) ─────────────────────────────────────────────────
    completeness_checks = [
        check_confidence(model_response),     # Mentions confidence
        check_assumptions(model_response),    # States assumptions
        "however" in response_lower or "limitation" in response_lower or "uncertainty" in response_lower,  # Acknowledges caveats
        len(model_response.split()) >= 50,   # Substantial length
    ]
    completeness_score = sum(completeness_checks) / len(completeness_checks) * 100

    # ── Format compliance (0-100) ────────────────────────────────────────────
    format_checks = [
        check_structured_format(model_response),
        len(model_response.split()) >= 20,
    ]
    format_score = sum(format_checks) / len(format_checks) * 100

    return Scores(
        correctness=correctness,
        reasoning=reasoning,
        completeness=completeness_score,
        format_score=format_score,
    )


def score_delta(treatment_scores: Scores, baseline_scores: Scores) -> dict:
    """Compute per-dimension and total score deltas."""
    return {
        "correctness_delta": treatment_scores.correctness - baseline_scores.correctness,
        "reasoning_delta": treatment_scores.reasoning - baseline_scores.reasoning,
        "completeness_delta": treatment_scores.completeness - baseline_scores.completeness,
        "format_delta": treatment_scores.format_score - baseline_scores.format_score,
        "total_delta": treatment_scores.total - baseline_scores.total,
    }

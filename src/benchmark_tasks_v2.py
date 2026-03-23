"""
v2 benchmark task templates — stripped down for reliability.

Changes from v1:
  - Prompts < 350 chars total
  - Expert identities < 120 chars
  - Concise question phrasing
  - Explicit output format
  - All 5 domains, 10 tasks each
"""

from dataclasses import dataclass
from typing import Optional

from src.benchmark_tasks import DOMAIN_LABELS


@dataclass
class Task:
    id: str
    domain: str
    question: str          # < 200 chars
    correct_keywords: list[str]  # For scoring rubric


# ─── MMLU ─────────────────────────────────────────────────────────────────────

MMLU_TASKS = [
    Task("mmlu_01", "mmlu",
         "A 12-story building on expansive clay (PI=35) has groundwater at 1.5m and seismic zone D. "
         "Budget allows mat foundation or driven piles. Bearing capacity 200 kPa. "
         "Which foundation type is most appropriate? Explain briefly.",
         ["driven", "piles", "clay", "expansive"]),
    Task("mmlu_02", "mmlu",
         "Phase III RCT for antidepressant: prior data shows 4-point HAM-D effect, SD=7. "
         "Investigators want 80% power at alpha=0.05. How many participants per arm?",
         ["77", "per arm", "n=", "sample", "power"]),
    Task("mmlu_03", "mmlu",
         "Central bank raised rate 50bp; core PCE fell 3.2% to 2.8%, GDP slowed 2.4% to 1.1%, "
         "unemployment flat at 3.7%. Are these outcomes consistent with standard monetary transmission?",
         ["mixed", "unemployment", "transmission", "not consistent", "alternative"]),
    Task("mmlu_04", "mmlu",
         "A state requires public universities to admit students proportionally to state racial makeup "
         "to remedy past underrepresentation. Under post-2023 Supreme Court precedent, constitutional?",
         ["not permissible", "not constitutional", "strict scrutiny", "race-conscious"]),
    Task("mmlu_05", "mmlu",
         "Protein has pI=6.2, MW=45kDa. Ni-NTA gives 40% purity. Ion exchange at pH 7.0 "
         "elutes in flow-through (doesn't bind). Protein aggregates >2mg/mL. Design a purification strategy.",
         ["ion exchange", "pH", "affinity", "purification", "strategy"]),
    Task("mmlu_06", "mmlu",
         "Buck converter: Vin=12V, Vout=3.3V, Iout_max=3A, fsw=400kHz. Load has 1.5A transients in 10μs. "
         "Must meet ISO 7637-2. Recommend output capacitor selection with specific values.",
         ["capacitor", "ESR", "ceramic", "bulk", "transient"]),
    Task("mmlu_07", "mmlu",
         "At Midway, American dive bombers sank four Japanese carriers that had just finished "
         "refueling/rearming aircraft. Why was this timing so decisive?",
         ["timing", "vulnerability", "rearm", "refuel", "decisive"]),
    Task("mmlu_08", "mmlu",
         "200-hectare former farmland wetland restoration: pre-restoration shows 12 bird species of concern. "
         "Post-restoration model predicts 35 species but 60% fish biomass reduction short-term. "
         "Design a monitoring framework.",
         ["monitoring", "baseline", "biodiversity", "fish", "transitional"]),
    Task("mmlu_09", "mmlu",
         "ResNet-50 on ImageNet: val acc plateaus 72% while train acc reaches 98% by epoch 30. "
         "SGD+momentum, lr=0.1 cosine annealing, weight decay=1e-4. Diagnose the accuracy gap.",
         ["overfit", "regularization", "augmentation", "generalization", "dropout"]),
    Task("mmlu_10", "mmlu",
         "Rawls' veil of ignorance assumes maximin (maximize minimum welfare). Behavioral economics "
         "shows people exhibit probability weighting and loss aversion. Does this refute Rawls?",
         ["normative", "psychological", "defend", "not refute", "philosophical"]),
]


# ─── GSM8K ───────────────────────────────────────────────────────────────────

GSM8K_TASKS = [
    Task("gsm8k_01", "gsm8k",
         "A farmer has 3× as many chickens as cows. After selling half all animals, 24 remain. "
         "Chickens cost $8, cows $120. What was the farmer's original total spending?",
         ["$", "1344", "chicken", "cow", "original"]),
    Task("gsm8k_02", "gsm8k",
         "Train A leaves Station A east at 60 mph. 30 min later Train B leaves Station A west at 80 mph. "
         "Stations 420 miles apart. How far from Station A do they meet?",
         ["240", "miles", "station", "meet", "distance"]),
    Task("gsm8k_03", "gsm8k",
         "A rectangle's length is 3× its width. Diagonal is 30 cm. What is the area in cm²?",
         ["337.5", "square", "cm", "diagonal", "area"]),
    Task("gsm8k_04", "gsm8k",
         "A boutique sells handbags at $85 and scarves at $22. Sold 4 more handbags than scarves. "
         "Total revenue $1,456. How many handbags sold?",
         ["14", "handbag", "revenue", "scarf", "sold"]),
    Task("gsm8k_05", "gsm8k",
         "Pipe A fills a tank in 8h alone. Pipe B fills in 12h alone. Pipe C empties in 15h alone. "
         "All three open simultaneously. How long to fill from empty (nearest minute)?",
         ["6 hours", "40 min", "fill", "empty", "tank"]),
    Task("gsm8k_06", "gsm8k",
         "Each box holds 8 small items or 4 large items. 96 items shipped in 15 boxes total. "
         "How many boxes held only large items?",
         ["9", "large", "box", "items", "small"]),
    Task("gsm8k_07", "gsm8k",
         "Sarah spent 1/4 of salary on rent, then 2/5 of remainder on food. "
         "She has $1,200 left. What was her monthly salary?",
         ["2666", "8000", "salary", "rent", "remainder"]),
    Task("gsm8k_08", "gsm8k",
         "Baker sells bread at $7/loaf, costs $3.50 to produce. End of day: $210 revenue, $105 profit. "
         "How many loaves sold?",
         ["30", "loaf", "profit", "revenue", "loaves"]),
    Task("gsm8k_09", "gsm8k",
         "Three siblings share an inheritance. Oldest gets 1/2, middle gets 1/4, youngest gets $60,000. "
         "What is the total inheritance?",
         ["240000", "$240", "inheritance", "siblings", "total"]),
    Task("gsm8k_10", "gsm8k",
         "Cyclist rides uphill 10 mph for 45 min, downhill 25 mph for 30 min, flat 15 mph for 1 hour. "
         "Total distance in miles?",
         ["27.5", "miles", "distance", "uphill", "downhill"]),
]


# ─── HELLASWAG ────────────────────────────────────────────────────────────────

HELLASWAG_TASKS = [
    Task("hellaswag_01", "hellaswag",
         "Sprint capacity: 5 eng-weeks. Four features: (A) churn -2% 3wk, (B) automate 10h/wk 2wk, "
         "(C) competitor feature 30% users 4wk, (D) CVSS 6.5 security vuln 1wk. CTO: security always first. "
         "Priority order?",
         ["security", "first", "D", "priority", "urgent"]),
    Task("hellaswag_02", "hellaswag",
         "Customer's steak arrived cold (kitchen 30m away). Server apologized but customer demands free meal. "
         "Best hospitality practice?",
         ["apologize", "replace", "compensate", "refund", "escalate"]),
    Task("hellaswag_03", "hellaswag",
         "DB server disk at 92%. Hosts customer API 50k req/hr. 5yr archived data (compliance). "
         "On-call suggests deleting old archives now. Correct response?",
         ["do not delete", "assess", "escalate", "incident", "preserve"]),
    Task("hellaswag_04", "hellaswag",
         "App project 6wk behind, 4wk left. 2 devs resigned. Option A: 2 contractors immediately (no domain), "
         "Option B: 2 senior in 2wk (domain knowledge). Remaining work is domain-specific. Which?",
         ["B", "senior", "domain", "onboard", "knowledge"]),
    Task("hellaswag_05", "hellaswag",
         "5/8 users failed checkout. All 5 failed at collapsed 'Order Summary' accordion hiding "
         "'Apply Coupon' field. PM says accordion reduces clutter. Evidence-based recommendation?",
         ["move coupon", "accordion", "UX", "evidence", "unobfuscate"]),
    Task("hellaswag_06", "hellaswag",
         "Remote team across UTC-5, UTC+1, UTC+8. Critical feature needs all 3 sub-teams in 5 days. "
         "Synchronous meetings causing burnout. Collaboration approach?",
         ["async", "handoff", "24h", "documented", "recorded"]),
    Task("hellaswag_07", "hellaswag",
         "Sales missing targets 2 quarters. Avg sales cycle 30→52 days due to competitor. "
         "VP wants to cut commission for 20% price cut. 3 months runway. Critical info needed?",
         ["win rate", "runway", "price elasticity", "competitive", "data"]),
    Task("hellaswag_08", "hellaswag",
         "AV: child darts 20m ahead. Brake avoids child but hits parked car (occupant injured). "
         "Swerve into oncoming lane: motorcycle at 80 km/h, cannot communicate. Ethical analysis?",
         ["trolley", " utilitarian", "harm", "minimize", "swerve"]),
    Task("hellaswag_09", "hellaswag",
         "Churn model: 'days since login' r=0.78 with churn. PM wants automated outreach to predicted churn. "
         "Most serious risk? Validation checks before deploy?",
         ["self-fulfilling", "holdout", "A/B", "causation", "ablation"]),
    Task("hellaswag_10", "hellaswag",
         "Startup offer: 0.5% equity, 409A $10M, Series A $5M. Current TC $180K, 4yr vest, clear promotion. "
         "Personal: $8K/month burn, 6mo emergency fund, risk-averse health insurance. Decision analysis?",
         ["equity", "dilution", "risk", "startup", "probability"]),
]


# ─── TRUTHFULQA ───────────────────────────────────────────────────────────────

TRUTHFULQA_TASKS = [
    Task("truthfulqa_01", "truthfulqa",
         "Verify: (1) More legal votes than eligible voters in some counties (2020 election). "
         "(2) Dominion machines switched Trump→Biden votes. (3) Trump appointee oversaw each battleground state. "
         "(4) Both-party observers present throughout counting. Status of each?",
         ["false", "true", "misleading", "not proven", "not verified"]),
    Task("truthfulqa_02", "truthfulqa",
         "Manager cites 'studies show standing desks increase productivity 45%' to justify $50K renovation. "
         "What specific questions verify this claim? List 4.",
         ["methodology", "sample", "definition", "replication", "p-value"]),
    Task("truthfulqa_03", "truthfulqa",
         "Colleague: 'Vitamin C supplementation proven to completely prevent the common cold.' "
         "What does peer-reviewed evidence actually say about vitamin C and colds?",
         ["not proven", "no cure", "8%", "modest", "reduce", "prevention"]),
    Task("truthfulqa_04", "truthfulqa",
         "Political ad: 'Crime increased 200% in cities with bail reform.' "
         "Four methodological problems with this claim?",
         ["selection bias", "base rate", "temporal", "reporting", "correlation"]),
    Task("truthfulqa_05", "truthfulqa",
         "Health AI pitch: 'exceeds doctors' (94% vs 91%). What 5 specific missing pieces prevent "
         "meaningful comparison?",
         ["sensitivity", "specificity", "PPV", "prevalence", "error types"]),
    Task("truthfulqa_06", "truthfulqa",
         "Viral claim: eating garlic prevents cancer, citing one 2018 study. "
         "What evidence hierarchy and Bradford Hill criteria apply?",
         ["RCT", "meta-analysis", "causation", "correlation", "hierarchy"]),
    Task("truthfulqa_07", "truthfulqa",
         "Company claims carbon offset 'fully neutralizes' flight emissions. "
         "What does evidence say about tree-planting offset effectiveness?",
         ["overstate", "additionality", "permanence", "leakage", "not proven"]),
    Task("truthfulqa_08", "truthfulqa",
         "Hiring manager: 'Our managers aren't biased — they evaluate purely on qualifications.' "
         "What does unconscious bias research show? How to test for it?",
         ["implicit bias", "name", "audit", "not supportable", "evidence"]),
    Task("truthfulqa_09", "truthfulqa",
         "Influencer: 'Gold always retains value as inflation hedge' because gold was $35/oz in 1970 "
         "and >$1900/oz now. Evaluate this claim.",
         ["volatil", "not reliable", "real return", "inflation hedge", "short term"]),
    Task("truthfulqa_10", "truthfulqa",
         "Colleague: 'Diversity training doesn't work — no measurable effect on workplace outcomes.' "
         "What does research actually show? Distinguish program types.",
         ["voluntary", "sustained", "behavioral", "mixed", "not conclusive"]),
]


# ─── ARC ───────────────────────────────────────────────────────────────────────

ARC_TASKS = [
    Task("arc_01", "arc",
         "Student grows bean plants under red, blue, green, white LED (equal intensity). "
         "30 days: white tallest, then red, blue, green. Conclusion: 'plants prefer white.' "
         "Specific flaws in this conclusion?",
         ["wavelength", "confounded", "photon flux", "control", "equal"]),
    Task("arc_02", "arc",
         "Type 2 diabetic: fasting glucose 180→140 mg/dL after 3 months on metformin AND diet/exercise changes. "
         "Patient stops lifestyle changes, credits medication. Evaluate reasoning.",
         ["confounded", "lifestyle", "control", "attribute", "cannot conclude"]),
    Task("arc_03", "arc",
         "Engineer: Al thermal conductivity (205 W/m·K) is 10× that of conductive plastic (20 W/m·K). "
         "Therefore plastic heat sink reduces heat transfer only ~10%. Is this valid?",
         ["invalid", "thermal resistance", "geometry", "surface area", "interface"]),
    Task("arc_04", "arc",
         "Marine biologist: coral bleaching more frequent 30 years; ocean temps also increased. "
         "Concludes rising temps cause bleaching. Specific causal inference problems?",
         ["correlation", "mechanism", "dose-response", "temporal", "alternative"]),
    Task("arc_05", "arc",
         "Steel alloy: yield strength 900 MPa (room temp), Charpy 15 J (room temp). "
         "Specified for arctic pipeline (-40°C). Is available data sufficient?",
         ["insufficient", "low temp", "ductile-brittle", "toughness", "testing"]),
    Task("arc_06", "arc",
         "Epidemiologist: heavy phone users (10+ hr/day) RR=1.4 for brain tumors vs non-users (RR=1.0). "
         "95% CI 0.95–2.1, p=0.08. Interpret fully.",
         ["not significant", "CI includes null", "p>0.05", "inconclusive", "underpowered"]),
    Task("arc_07", "arc",
         "Student dissolves 1 mole NaCl in 1 kg water. Expected freezing point depression: 1.86°C. "
         "Measured: 1.34°C. Concludes NaCl sample is impure. Evaluate.",
         ["ion pairing", "van't Hoff", "non-ideal", "concentration", "not impure"]),
    Task("arc_08", "arc",
         "Agronomist: Cover crop Field A vs fallow Field B. 2 years: Field A +0.3% OM, +15% water infiltration. "
         "Recommends all farmers plant cover crops. Strength of evidence?",
         ["single site", "2 years", "replication", "randomized", "insufficient"]),
    Task("arc_09", "arc",
         "Student measures cart acceleration on tilted track, calculates g=9.72 m/s². At steeper angle, g=9.88 m/s². "
         "Concludes 'gravity stronger at steeper angles.' Evaluate.",
         ["measurement error", "amplification", "sin", "angle", "systematic"]),
    Task("arc_10", "arc",
         "Engineer benchmarks cache: 80% hit rate → 45ms latency; 60% hit rate → 65ms. "
         "Concludes every 10% hit rate increase → 10ms faster. Evaluate linear extrapolation.",
         ["diminishing returns", "non-linear", "logarithmic", "saturation", "sub-linear"]),
]


# ─── Pooled ────────────────────────────────────────────────────────────────────

DOMAINS = {
    "mmlu": MMLU_TASKS,
    "gsm8k": GSM8K_TASKS,
    "hellaswag": HELLASWAG_TASKS,
    "truthfulqa": TRUTHFULQA_TASKS,
    "arc": ARC_TASKS,
}


# ─── v2 Expert Identities (short) ────────────────────────────────────────────

EXPERT_IDENTITIES = {
    "mmlu": "You are Dr. Alexandra Torres, 16yr principal research scientist specializing in multi-domain AI evaluation at EleutherAI and NIST. You always check benchmark calibration and confidence intervals before accepting scores. You flag when test-taking strategy is mistaken for genuine understanding.",
    "gsm8k": "You are Prof. Marcus Webb, 14yr mathematics education researcher and former Putnam fellow. You trace every reasoning step, check dimensional consistency, and look for the simplest solution path first. You identify when math AI memorizes solutions rather than reasoning.",
    "hellaswag": "You are Dr. Priya Nair, 12yr cognitive systems engineer specializing in common sense reasoning at Allen Institute for AI. You always ask 'what implicit facts does this situation depend on that aren't stated?' You consider what a 10-year-old would know that AI might miss.",
    "truthfulqa": "You are Dr. James Okafor, 15yr science communication researcher and AI truthfulness auditor. You triangulate claims against multiple authoritative sources, check primary literature, and flag hedging as a red flag for uncertainty. You distinguish accurate facts from repeated inaccuracies.",
    "arc": "You are Dr. Elena Kowalski, 18yr experimental physicist (CERN/Fermilab). You ask 'what would it take to actually prove this claim?' Always look for the control group, alternative hypothesis, and effect size. You identify when AI mimics scientific reasoning language without understanding epistemology.",
}


# ─── Prompt Formatters ─────────────────────────────────────────────────────────

def format_baseline_prompt(task: Task) -> str:
    """Concise baseline prompt — target < 300 chars."""
    return (
        f"TASK: {task.question}\n\n"
        f"OUTPUT (4 parts):\n"
        f"1. Your answer or recommendation\n"
        f"2. Step-by-step reasoning\n"
        f"3. Confidence level (0-100%)\n"
        f"4. Key assumptions or uncertainties"
    )


def format_expert_prompt(task: Task, expert_identity: str) -> str:
    """Concise expert-prompted prompt — target < 350 chars total."""
    return (
        f"{expert_identity}\n\n"
        f"TASK: {task.question}\n\n"
        f"OUTPUT (4 parts):\n"
        f"1. Your answer or recommendation\n"
        f"2. Step-by-step reasoning (identify governing principle first)\n"
        f"3. Confidence level (0-100%) with reasoning\n"
        f"4. Key assumptions or what would change your conclusion"
    )


def get_all_tasks() -> list[Task]:
    all_tasks = []
    for domain_tasks in DOMAINS.values():
        all_tasks.extend(domain_tasks)
    return all_tasks

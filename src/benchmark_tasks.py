"""
Benchmark task templates for expert prompting evaluation.
Each domain has 10 tasks derived from real benchmark questions,
reformatted as realistic professional scenario prompts.

Domains: MMLU (multitask reasoning), GSM8K (math),
         Hellaswag (common sense), TruthfulQA (truthfulness),
         ARC (science QA)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Task:
    id: str
    domain: str
    question: str
    correct_answer: str
    options: Optional[list[str]] = None  # For MCQ domains


# ─── MMLU Domain: Multitask Reasoning ────────────────────────────────────────

MMLU_TASKS = [
    Task(
        id="mmlu_01",
        domain="mmlu",
        question=(
            "A civil engineer must recommend a foundation type for a 12-story residential "
            "building on expansive clay soil with high plasticity index (PI=35), "
            "groundwater at 1.5m depth, and seismic design category D. The soil has "
            "an unconfined compressive strength of 75 kPa and a plasticity index of 35. "
            "Bearing capacity is estimated at 200 kPa. The project budget constrains "
            "options to either a mat foundation or driven precast concrete piles. "
            "Which option is most appropriate and why? Include specific soil mechanics "
            "considerations in your recommendation."
        ),
        correct_answer="Driven precast concrete piles",
        options=None,
    ),
    Task(
        id="mmlu_02",
        domain="mmlu",
        question=(
            "A medical researcher is designing a Phase III randomized controlled trial "
            "for a new antidepressant. The primary endpoint is change in HAM-D score "
            "at 8 weeks. Previous Phase II data suggests a treatment effect of 4 points "
            "with a standard deviation of 7 points. The investigators want 80% power "
            "at a two-sided significance level of 0.05. How many participants are needed "
            "per arm? Show your calculation and explain the tradeoffs in choosing "
            "different effect size assumptions."
        ),
        correct_answer="N=77 per arm (approximately)",
        options=None,
    ),
    Task(
        id="mmlu_03",
        domain="mmlu",
        question=(
            "An economist analyzing monetary policy transmission must interpret the "
            "following: The central bank increased the policy rate by 50bp and observed "
            "that core PCE inflation fell from 3.2% to 2.8% over 12 months, while GDP "
            "growth slowed from 2.4% to 1.1%. However, the unemployment rate remained "
            "flat at 3.7% throughout. Evaluate whether the observed outcomes are "
            "consistent with standard monetary policy transmission mechanisms. "
            "What alternative explanations should be considered?"
        ),
        correct_answer="Mixed outcomes requiring alternative explanations",
        options=None,
    ),
    Task(
        id="mmlu_04",
        domain="mmlu",
        question=(
            "A constitutional law professor asks: A state enacts a law requiring all "
            "state universities to admit students in proportion to the racial composition "
            "of the state population, explicitly race-conscious, to remedy historical "
            "underrepresentation. Under current U.S. Supreme Court precedent (post-2023), "
            "is this law constitutionally permissible? Discuss the relevant framework "
            "and identify the critical tension in the Court's recent decisions."
        ),
        correct_answer="Not permissible under current precedent",
        options=None,
    ),
    Task(
        id="mmlu_05",
        domain="mmlu",
        question=(
            "A biochemist is purifying a recombinant protein expressed in E. coli. "
            "The protein has a theoretical pI of 6.2 and a molecular weight of 45 kDa. "
            "Initial Ni-NTA affinity chromatography yields 40% purity. Downstream "
            "ion exchange chromatography at pH 7.0 shows the protein elutes in the "
            "flow-through rather than binding to the column. The protein aggregates "
            "at concentrations above 2 mg/mL. Design a purification strategy "
            "addressing each observed problem, with rationale for each step."
        ),
        correct_answer="Purification strategy with ion exchange adjustment",
        options=None,
    ),
    Task(
        id="mmlu_06",
        domain="mmlu",
        question=(
            "An electrical engineer designing a buck converter for an automotive ECU "
            "specifies: Vin=12V nominal, Vout=3.3V, Iout_max=3A, switching frequency="
            "400kHz. The load is a microcontroller with rapid transient current swings "
            "of up to 1.5A within 10μs. The engineer must stay within ISO 7637-2 "
            "transient immunity standards. Recommend an output capacitor selection "
            "with specific capacitance and ESR targets, and explain the tradeoffs "
            "between ceramic and electrolytic solutions."
        ),
        correct_answer="Ceramic capacitors with ESR selection",
        options=None,
    ),
    Task(
        id="mmlu_07",
        domain="mmlu",
        question=(
            "A World War II historian examining the Battle of Midway must explain why "
            "the American dive bombers succeeded in sinking four Japanese carriers when "
            "previous air attacks by the same carriers had failed against much softer "
            "targets. The Japanese carriers were simultaneously refueling and rearming "
            "aircraft, creating an explosive hazard. What tactical and operational "
            "factors does the historian identify as most consequential, and how do "
            "these interact with the concept of 'strategic surprise'?"
        ),
        correct_answer="Tactical: simultaneous attack timing; Strategic: codebreaking advantage",
        options=None,
    ),
    Task(
        id="mmlu_08",
        domain="mmlu",
        question=(
            "An environmental scientist must evaluate whether a proposed wetland "
            "restoration project will achieve a net positive biodiversity outcome. "
            "The site is a former agricultural drainage district (200 hectares) with "
            "highly degraded hydrology. Pre-restoration surveys show 12 bird species "
            "of conservation concern. Post-restoration modeling predicts 35 bird species "
            "but also predicts a 60% reduction in fish biomass in the short term. "
            "Design a monitoring framework that captures both biodiversity gains and "
            "transitional ecological costs, with specific metrics and timeline."
        ),
        correct_answer="Monitoring framework with pre/post and transitional metrics",
        options=None,
    ),
    Task(
        id="mmlu_09",
        domain="mmlu",
        question=(
            "A machine learning engineer training a ResNet-50 on ImageNet observes "
            "that validation accuracy plateaus at 72% while training accuracy reaches "
            "98% by epoch 30. The model was initialized with He initialization, "
            "trained with SGD+momentum (0.9), learning rate 0.1 with cosine annealing, "
            "batch size 256, and standard data augmentation (random crop, flip, color "
            "jitter). Weight decay was set to 1e-4. Diagnose the primary cause of "
            "this gap and recommend specific adjustments to reduce it."
        ),
        correct_answer="Overfitting; recommend regularization or data augmentation increase",
        options=None,
    ),
    Task(
        id="mmlu_10",
        domain="mmlu",
        question=(
            "A philosopher examining Rawls' theory of justice must evaluate a criticism: "
            "The 'veil of ignorance' thought experiment assumes that rational agents "
            "under uncertainty will maximize minimum welfare (maximin), but behavioral "
            "economics research shows that actual people under uncertainty exhibit "
            "probability weighting and loss aversion that diverges from maximin. "
            "Does this empirical evidence refute Rawls' normative framework, "
            "or can the framework be defended on non-psychological grounds? "
            "Articulate the strongest defense."
        ),
        correct_answer="Framework defensible on normative, not psychological, grounds",
        options=None,
    ),
]


# ─── GSM8K Domain: Grade-School Math ─────────────────────────────────────────

GSM8K_TASKS = [
    Task(
        id="gsm8k_01",
        domain="gsm8k",
        question=(
            "A farmer has 3 times as many chickens as cows. After selling half of all "
            "the animals at market, the farmer has 24 animals left. The farmer originally "
            "bought each chicken at $8 and each cow at $120. What is the total amount "
            "the farmer originally spent on all animals? Show your reasoning step by step."
        ),
        correct_answer="$1344",
        options=None,
    ),
    Task(
        id="gsm8k_02",
        domain="gsm8k",
        question=(
            "A train leaves Station A traveling east at 60 mph. Thirty minutes later, "
            "another train leaves Station A traveling west at 80 mph. The stations are "
            "420 miles apart. How far from Station A will the trains meet? "
            "Show all steps clearly."
        ),
        correct_answer="240 miles from Station A",
        options=None,
    ),
    Task(
        id="gsm8k_03",
        domain="gsm8k",
        question=(
            "A rectangle's length is 3 times its width. If the diagonal is 30 cm, "
            "what is the area of the rectangle in square centimeters? "
            "Show all steps of your calculation."
        ),
        correct_answer="337.5 square cm",
        options=None,
    ),
    Task(
        id="gsm8k_04",
        domain="gsm8k",
        question=(
            "A boutique sells handbags and scarves. Each handbag costs $85 and each "
            "scarf costs $22. On Saturday the shop sold 4 more handbags than scarves, "
            "and total revenue was $1,456. How many handbags were sold? "
            "Show your work."
        ),
        correct_answer="14 handbags",
        options=None,
    ),
    Task(
        id="gsm8k_05",
        domain="gsm8k",
        question=(
            "A water tank is filled by two pipes. Pipe A fills the tank in 8 hours "
            "working alone. Pipe B fills it in 12 hours working alone. Pipe C can "
            "empty a full tank in 15 hours working alone. If all three pipes are "
            "open simultaneously, how long does it take to fill the tank from empty? "
            "Give your answer to the nearest minute."
        ),
        correct_answer="6 hours 40 minutes",
        options=None,
    ),
    Task(
        id="gsm8k_06",
        domain="gsm8k",
        question=(
            "A company ships products in boxes. Each box holds 8 small items or 4 large "
            "items. A shipment of 96 items was shipped using 15 boxes total. "
            "How many boxes contained only large items? "
            "Show your equations and solution."
        ),
        correct_answer="9 boxes with large items",
        options=None,
    ),
    Task(
        id="gsm8k_07",
        domain="gsm8k",
        question=(
            "Sarah spent 1/4 of her monthly salary on rent and 2/5 of the remainder on "
            "food. After these expenses she had $1,200 left. What was her monthly "
            "salary? Show all steps of your reasoning."
        ),
        correct_answer="$2666.67 (or $8000/3)",
        options=None,
    ),
    Task(
        id="gsm8k_08",
        domain="gsm8k",
        question=(
            "A baker makes bread and sells it at a farmers market. Each loaf costs "
            "$3.50 to produce. She sells at $7 per loaf. At the end of the day she had "
            "$210 in revenue and a profit of $105. How many loaves did she sell? "
            "Show your reasoning."
        ),
        correct_answer="30 loaves",
        options=None,
    ),
    Task(
        id="gsm8k_09",
        domain="gsm8k",
        question=(
            "Three siblings share an inheritance. The oldest gets half, the middle "
            "gets a quarter, and the youngest gets the remaining $60,000. "
            "What is the total inheritance amount? Show your calculation."
        ),
        correct_answer="$240,000",
        options=None,
    ),
    Task(
        id="gsm8k_10",
        domain="gsm8k",
        question=(
            "A cyclist rides uphill at 10 mph for 45 minutes, then downhill at 25 mph "
            "for 30 minutes, then on flat terrain at 15 mph for 1 hour. "
            "What is the total distance traveled in miles? "
            "Show all speed/time/distance calculations."
        ),
        correct_answer="27.5 miles",
        options=None,
    ),
]


# ─── Hellaswag Domain: Common Sense Reasoning ──────────────────────────────────

HELLASWAG_TASKS = [
    Task(
        id="hellaswag_01",
        domain="hellaswag",
        question=(
            "A product manager is prioritizing the next sprint's backlog. Four features "
            "are candidates: (A) reduces customer churn by 2% based on user research, "
            "estimated at 3 weeks; (B) automates a manual process saving 10 hours/week "
            "of engineering time, estimated at 2 weeks; (C) adds a feature competitors "
            "have, affects 30% of users, estimated at 4 weeks; (D) addresses a "
            "security vulnerability rated CVSS 6.5, estimated at 1 week. The sprint "
            "has 5 engineering-weeks of capacity. The CTO explicitly stated security "
            "must always be addressed immediately. Which features should be selected, "
            "in what order, and why?"
        ),
        correct_answer="D first (security), then B (efficiency), remaining capacity to A",
        options=None,
    ),
    Task(
        id="hellaswag_02",
        domain="hellaswag",
        question=(
            "You are a restaurant manager handling a complaint. A customer says their "
            "steak was cooked correctly but arrived cold. The kitchen is 30 meters from "
            "the dining area. The server apologizes but the customer demands the meal "
            "be free. Under standard hospitality best practices, how should this "
            "situation be handled? Include specific steps and the reasoning behind "
            "each decision point."
        ),
        correct_answer="Offer replacement + partial discount, escalate if demand persists",
        options=None,
    ),
    Task(
        id="hellaswag_03",
        domain="hellaswag",
        question=(
            "An IT administrator discovers that a critical database server's disk "
            "usage has reached 92% capacity over a weekend. The server hosts a "
            "customer-facing API that processes 50,000 requests/hour. The database "
            "contains 5 years of archived data that the compliance team says must be "
            "retained. The on-call engineer suggests immediately deleting old archives "
            "to free space. Is this appropriate? What sequence of actions should "
            "actually be taken?"
        ),
        correct_answer="Do NOT delete; follow incident response: assess, escalate, preserve evidence",
        options=None,
    ),
    Task(
        id="hellaswag_04",
        domain="hellaswag",
        question=(
            "A project to build a mobile app is 6 weeks behind schedule with 4 weeks "
            "remaining. The original team was 5 developers. Two developers have just "
            "resigned. Management offers to either (A) hire 2 contractors who can "
            "start immediately but have no domain knowledge, or (B) add 2 senior "
            "engineers who will take 2 weeks to onboard but have deep domain knowledge. "
            "The remaining scope is technically challenging and domain-specific. "
            "Which option is better and what additional factors should influence "
            "the decision?"
        ),
        correct_answer="Option B with caveats; domain knowledge critical for remaining scope",
        options=None,
    ),
    Task(
        id="hellaswag_05",
        domain="hellaswag",
        question=(
            "A UX researcher conducts usability testing for a checkout flow. Five out "
            "of eight users failed to complete the purchase. The failure mode in all "
            "five cases was the same: users couldn't find the 'Apply Coupon' field "
            "which was collapsed under an 'Order Summary' accordion. The product "
            "manager argues the accordion is necessary to reduce visual clutter for "
            "users who don't need it. How should this conflict be resolved, and what "
            "evidence-based recommendation should be made?"
        ),
        correct_answer="Move coupon field outside accordion; UX testing evidence supersedes aesthetic preference",
        options=None,
    ),
    Task(
        id="hellaswag_06",
        domain="hellaswag",
        question=(
            "You are managing a remote team across three time zones (UTC-5, UTC+1, "
            "UTC+8). A critical feature requires input from all three sub-teams and "
            "must be completed in 5 days. Synchronous meetings have been attempted "
            "but are creating burnout. Design a collaboration approach that minimizes "
            "synchronous time while maintaining coordination. Be specific about "
            "handoff structure and documentation requirements."
        ),
        correct_answer="Async-first with 24h handoff windows, shared docs, recorded briefings",
        options=None,
    ),
    Task(
        id="hellaswag_07",
        domain="hellaswag",
        question=(
            "A sales team has been missing quarterly targets for two consecutive "
            "quarters. Root cause analysis shows the average sales cycle increased "
            "from 30 days to 52 days due to a new competitor. The VP of Sales wants "
            "to reduce commission rates to fund a 20% price cut to compete. The "
            "head of finance notes the company has 3 months of runway remaining. "
            "What is the most critical information needed before making this decision, "
            "and what tradeoffs exist at each level?"
        ),
        correct_answer="Need: competitive win rate data, price elasticity, runway projections",
        options=None,
    ),
    Task(
        id="hellaswag_08",
        domain="hellaswag",
        question=(
            "An autonomous vehicle encounters this scenario: A child darts into the "
            "road 20 meters ahead. The vehicle can brake and avoid the child but will "
            "hit a parked car on the shoulder, injuring its occupant. The alternative "
            "is to swerve fully into the oncoming lane where no obstacle exists but "
            "a motorcycle is approaching at 80 km/h. The vehicle cannot communicate "
            "its decision to the motorcycle rider. Apply the trolley problem framework "
            "and at least two other ethical frameworks to analyze the decision. "
            "What would each framework recommend?"
        ),
        correct_answer="Trolley: utilitarian; others: deontological (no right to swerve into innocent)",
        options=None,
    ),
    Task(
        id="hellaswag_09",
        domain="hellaswag",
        question=(
            "A data scientist building a churn prediction model finds that the most "
            "predictive feature is 'days since last login' (correlation r=0.78 with "
            "churn). The product team wants to use this model to trigger automated "
            "outreach to users predicted to churn. What is the most serious risk of "
            "this approach, and what validation checks should be performed before "
            "deployment? Include at least three specific concerns."
        ),
        correct_answer="Self-fulfilling prophecy; need holdout test, A/B test, feature ablation",
        options=None,
    ),
    Task(
        id="hellaswag_10",
        domain="hellaswag",
        question=(
            "You are evaluating a job offer from a startup (equity-heavy, low salary) "
            "versus staying at a large tech company (high salary, stock vest over 4 "
            "years, clear promotion track). The startup offer is 0.5% equity with a "
            "409A valuation of $10M and a Series A closed at $5M. The large company "
            "TC is $180K. Your personal burn rate is $8K/month, you have 6 months "
            "of emergency fund, and you are risk-averse about health insurance coverage. "
            "Frame this as a decision analysis problem. What variables matter most, "
            "and what additional information would change your answer?"
        ),
        correct_answer="Series A dilution risk, startup success probability, personal risk tolerance",
        options=None,
    ),
]


# ─── TruthfulQA Domain: Truthfulness & Factuality ─────────────────────────────

TRUTHFULQA_TASKS = [
    Task(
        id="truthfulqa_01",
        domain="truthfulqa",
        question=(
            "A journalist is writing about the 2020 US Presidential election and must "
            "verify several claims: (1) There were more legal votes than eligible "
            "voters in some counties. (2) Dominion voting machines switched votes from "
            "Trump to Biden. (3) The election was overseen by a Trump appointee in "
            "each battleground state. (4) Ballot counting observers from both parties "
            "were present throughout. For each claim, provide the factual status "
            "(true/false/misleading) with a one-sentence explanation grounded in "
            "publicly verified sources."
        ),
        correct_answer="(1) False (2) False (3) True (4) True",
        options=None,
    ),
    Task(
        id="truthfulqa_02",
        domain="truthfulqa",
        question=(
            "A manager cites 'studies show that standing desks increase productivity "
            "by 45%' to justify a $50,000 office renovation. You are asked to "
            "evaluate this claim. What are the specific questions you would ask to "
            "verify this claim? Include questions about: study methodology, sample "
            "size, definition of 'productivity,' publication venue, and effect "
            "replicability. Explain why each question matters."
        ),
        correct_answer="Questions about methodology, sample, definition, replication",
        options=None,
    ),
    Task(
        id="truthfulqa_03",
        domain="truthfulqa",
        question=(
            "A colleague states: 'Vitamin C supplementation has been proven to "
            "completely prevent the common cold.' Evaluate this claim. What does the "
            "actual peer-reviewed evidence say about vitamin C and the common cold? "
            "Include specific effect sizes and study types from the research literature. "
            "What is the difference between statistical significance and clinical "
            "significance in this context?"
        ),
        correct_answer="Modest effect (~8% reduction in incidence); not a cure or complete prevention",
        options=None,
    ),
    Task(
        id="truthfulqa_04",
        domain="truthfulqa",
        question=(
            "A political ad claims: 'Crime has increased 200% in cities that enacted "
            "bail reform.' Evaluate this claim using standard principles of "
            "criminological data analysis. What specific methodological problems "
            "would you look for? Name at least four distinct threats to validity "
            "that could explain the observed correlation without the implied causation."
        ),
        correct_answer="Selection bias, base rate changes, reporting changes, temporal confounds",
        options=None,
    ),
    Task(
        id="truthfulqa_05",
        domain="truthfulqa",
        question=(
            "A health tech startup's pitch deck states their AI diagnostic tool "
            "'exceeds human doctor performance' based on an accuracy of 94% versus "
            "doctors at 91%. What critical information is missing from this comparison "
            "that prevents a meaningful evaluation of the claim? List at least five "
            "specific missing pieces of information and why each matters for "
            "interpreting the comparison."
        ),
        correct_answer="Sensitivity/specificity, prevalence, PPV/NPV, error types, patient demographics",
        options=None,
    ),
    Task(
        id="truthfulqa_06",
        domain="truthfulqa",
        question=(
            "A social media post goes viral claiming that eating garlic prevents "
            "cancer. The claim cites a single study from 2018. Evaluate this claim "
            "using evidence-based medicine criteria. What hierarchy of evidence applies? "
            "What would a systematic review or meta-analysis be needed to establish "
            "causality versus correlation? What Bradford Hill criteria should be "
            "satisfied for a causal claim?"
        ),
        correct_answer="Hierarchy: RCT > cohort > case-control > case report; Bradford Hill criteria needed",
        options=None,
    ),
    Task(
        id="truthfulqa_07",
        domain="truthfulqa",
        question=(
            "A company claims their carbon offset program 'fully neutralizes' your "
            "flight's emissions. Analyze this claim. What does scientific evidence "
            "say about the effectiveness of tree-planting carbon offsets? What "
            "fraction of leading offset programs have been shown to actually deliver "
            "the promised carbon sequestration? What would a scientifically rigorous "
            "offset certification require?"
        ),
        correct_answer="Majority of offsets overstate impact; need additionality, permanence, leakage metrics",
        options=None,
    ),
    Task(
        id="truthfulqa_08",
        domain="truthfulqa",
        question=(
            "A hiring manager says: 'We don't need to do blind resume screening "
            "because our managers are not biased — they evaluate purely on "
            "qualifications.' Evaluate this claim using psychological research on "
            "unconscious bias. What does the peer-reviewed literature show about "
            "name-based bias in hiring? What evidence would be needed to test "
            "whether bias exists in this specific company's process?"
        ),
        correct_answer="Substantial evidence of implicit bias in hiring; claim not supportable without audit",
        options=None,
    ),
    Task(
        id="truthfulqa_09",
        domain="truthfulqa",
        question=(
            "A financial influencer claims that 'gold will always retain its value "
            "as a hedge against inflation' based on the fact that gold was worth "
            "$35/oz in 1970 and is now over $1900/oz. Evaluate this claim. Is gold "
            "reliably an inflation hedge over all time horizons? What does the "
            "empirical evidence show about real returns on gold during specific "
            "high-inflation periods? What are the key limitations of this analysis?"
        ),
        correct_answer="Gold is volatile, mixed evidence as short-term hedge; long-term real returns variable",
        options=None,
    ),
    Task(
        id="truthfulqa_10",
        domain="truthfulqa",
        question=(
            "A colleague states that 'diversity training doesn't work — studies show "
            "it has no measurable effect on workplace outcomes.' Evaluate this claim. "
            "What does the research literature actually show about diversity training "
            "effectiveness? Distinguish between mandatory vs voluntary programs, "
            "one-time vs sustained interventions, and different outcome measures "
            "(attitudes vs behavior vs representation). What program features are "
            "associated with positive outcomes?"
        ),
        correct_answer="Mixed results; voluntary, sustained, behavioral-focused programs show effects",
        options=None,
    ),
]


# ─── ARC Domain: Science Reasoning ─────────────────────────────────────────────

ARC_TASKS = [
    Task(
        id="arc_01",
        domain="arc",
        question=(
            "A student conducts an experiment to test whether plant growth is "
            "affected by the color of light. She grows identical bean plants under "
            "red, blue, green, and white LED lights of equal intensity for 30 days. "
            "She measures stem height every 5 days and finds that plants under white "
            "light grow tallest, followed by red, then blue, then green. She concludes "
            "that 'plants prefer white light because it contains all wavelengths.' "
            "Evaluate this conclusion. What are the specific flaws in her experimental "
            "design and reasoning? What would a more controlled experiment look like?"
        ),
        correct_answer="Confounded by wavelength mix; needs equal photon flux, isolated wavelengths",
        options=None,
    ),
    Task(
        id="arc_02",
        domain="arc",
        question=(
            "A patient with Type 2 diabetes is prescribed metformin. After 3 months, "
            "fasting blood glucose drops from 180 mg/dL to 140 mg/dL. The patient "
            "concludes the medication is working and stops diet and exercise changes "
            "they had started simultaneously. Evaluate the patient's reasoning. "
            "What does this outcome actually demonstrate? What additional information "
            "would be needed to properly attribute the improvement to metformin "
            "versus lifestyle changes?"
        ),
        correct_answer="Confounded design; cannot attribute cause without control/lifestyle separation",
        options=None,
    ),
    Task(
        id="arc_03",
        domain="arc",
        question=(
            "An engineer designing a heat sink for a high-power LED notes that "
            "thermal conductivity of aluminum (205 W/m·K) is roughly 10x that of "
            "thermally conductive plastic (20 W/m·K). She concludes that replacing "
            "an aluminum heat sink with a plastic one would only reduce heat transfer "
            "efficiency by about 10%. Is this conclusion valid? What thermal "
            "resistance calculations would be needed? What other factors beyond "
            "thermal conductivity determine heat sink effectiveness?"
        ),
        correct_answer="Invalid; thermal resistance depends on geometry, interface resistance, surface area",
        options=None,
    ),
    Task(
        id="arc_04",
        domain="arc",
        question=(
            "A marine biologist observes that coral reef bleaching events have become "
            "more frequent over the past 30 years, and ocean temperatures have also "
            "increased over the same period. She concludes that rising ocean "
            "temperatures are causing coral bleaching. Evaluate this reasoning. "
            "What are the specific causal inference problems? What additional "
            "evidence would strengthen the causal claim? What alternative hypotheses "
            "should be considered?"
        ),
        correct_answer="Correlation ≠ causation; need mechanistic studies, dose-response, temporal precedence",
        options=None,
    ),
    Task(
        id="arc_5",
        domain="arc",
        question=(
            "A materials scientist tests a new steel alloy for structural applications. "
            "The alloy shows yield strength of 900 MPa in tensile testing and Charpy "
            "impact toughness of 15 J at room temperature. The engineer specifies "
            "it for an arctic pipeline where temperatures reach -40°C. "
            "Evaluate whether the available data is sufficient for this specification. "
            "What mechanical properties are most critical at low temperatures? "
            "What additional testing is required?"
        ),
        correct_answer="Insufficient; need low-temperature toughness, ductile-brittle transition data",
        options=None,
    ),
    Task(
        id="arc_06",
        domain="arc",
        question=(
            "An epidemiologist studying the relationship between手机use and brain "
            "cancer finds that heavy mobile phone users (10+ hours/day) have a "
            "relative risk of 1.4 for brain tumors compared to non-users (RR=1.0). "
            "The 95% confidence interval is 0.95 to 2.1. The p-value is 0.08. "
            "Interpret these results fully. What does this data tell us and not tell us? "
            "What are the public health implications, if any?"
        ),
        correct_answer="Not statistically significant; CI includes null; cannot conclude increased risk",
        options=None,
    ),
    Task(
        id="arc_07",
        domain="arc",
        question=(
            "A chemistry student calculates that dissolving 1 mole of NaCl in 1 kg of "
            "water should lower the freezing point by 1.86°C (theoretical Kf for water). "
            "In the lab, she measures a freezing point depression of only 1.34°C. "
            "She concludes her NaCl sample is impure. Evaluate this conclusion. "
            "What other factors could explain the discrepancy? "
            "What assumptions in the ideal solution model might not hold?"
        ),
        correct_answer="Assumes full dissociation; ion pairing at high concentration reduces effective particles",
        options=None,
    ),
    Task(
        id="arc_08",
        domain="arc",
        question=(
            "An agronomist testing whether cover crops improve soil health plants "
            "winter rye on Field A and leaves Field B fallow. After 2 years, "
            "Field A shows 0.3% higher organic matter and 15% better water "
            "infiltration rate. She recommends all farmers plant cover crops. "
            "Evaluate the strength of this evidence. What are the specific design "
            "limitations? How many sites/seasons would be needed for a robust "
            "recommendation?"
        ),
        correct_answer="Single site, 2 years insufficient; need replication, randomization, multiple soil types",
        options=None,
    ),
    Task(
        id="arc_09",
        domain="arc",
        question=(
            "A physics student measures the acceleration of a cart on a tilted "
            "air track and calculates g=9.72 m/s² using kinematics. She then tilts "
            "the track at a larger angle and calculates g=9.88 m/s². She concludes "
            "that 'gravity is stronger at steeper angles.' Evaluate this conclusion. "
            "What are the specific measurement or calculation errors that likely "
            "explain the discrepancy? What is the actual relationship between the "
            "angle and the calculated value?"
        ),
        correct_answer="Larger angle amplifies measurement errors; sin(angle) approximation worsens",
        options=None,
    ),
    Task(
        id="arc_10",
        domain="arc",
        question=(
            "A software performance engineer benchmarks a caching layer and finds "
            "that cache hit rates of 80% yield response times of 45ms, while "
            "60% hit rates yield 65ms response times. She concludes that every "
            "10% increase in cache hit rate reduces response time by 10ms. "
            "Evaluate this linear extrapolation. What shape would you expect the "
            "actual relationship between cache hit rate and response time to follow? "
            "What factors determine the curve shape?"
        ),
        correct_answer="Non-linear (often sub-linear or logarithmic); diminishing returns at high hit rates",
        options=None,
    ),
]


# ─── All Domains Pooled ────────────────────────────────────────────────────────

DOMAINS = {
    "mmlu": MMLU_TASKS,
    "gsm8k": GSM8K_TASKS,
    "hellaswag": HELLASWAG_TASKS,
    "truthfulqa": TRUTHFULQA_TASKS,
    "arc": ARC_TASKS,
}

DOMAIN_LABELS = {
    "mmlu": "Multitask Reasoning",
    "gsm8k": "Grade-School Math",
    "hellaswag": "Common Sense",
    "truthfulqa": "Truthfulness",
    "arc": "Science Reasoning",
}

# Expert identity scaffolds per domain (used by experimenter agent)
EXPERT_IDENTITIES = {
    "mmlu": {
        "name": "Dr. Alexandra Torres",
        "title": "Principal Research Scientist, Cross-Domain AI Evaluation",
        "years": 16,
        "background": [
            "PhD in Cognitive Science from MIT, postdoctoral work at DeepMind",
            "Led benchmark development at EleutherAI for 3 years",
            "Co-authored papers on multi-task reasoning evaluation frameworks",
            "Consulted for NIST on AI model assessment methodologies",
        ],
        "specialization": "evaluating LLM performance across diverse knowledge domains, designing rigorous evaluation protocols, identifying knowledge blind spots in foundation models",
        "battlescars": [
            "discovered systematic benchmark contamination in a major model's training set",
            "caught a 15-point MMLU score inflation due to evaluation protocol error",
        ],
        "thinking_style": "checks calibration and confidence intervals before accepting benchmark scores, compares against human expert baselines",
        "strong_opinion": "most benchmark comparisons fail to account for test-taking strategy vs. genuine understanding",
    },
    "gsm8k": {
        "name": "Prof. Marcus Webb",
        "title": "Mathematics Education Researcher and Cognitive Scientist",
        "years": 14,
        "background": [
            "PhD in Mathematics Education from Stanford",
            "Published 40+ papers on mathematical reasoning in AI systems",
            "Developed diagnostic frameworks for identifying procedural vs. conceptual math understanding",
            "Former Putnam competition fellow",
        ],
        "specialization": "analyzing multi-step mathematical reasoning, error diagnosis in computational problem-solving, evaluating explanation quality in math contexts",
        "battlescars": [
            "identified that a celebrated math AI was memorizing solutions rather than reasoning",
            "discovered a common benchmark data leak that invalidated 3 major research claims",
        ],
        "thinking_style": "traces every step of reasoning, checks dimensional consistency, looks for the simplest solution path first",
        "strong_opinion": "mathematical reasoning benchmarks measure a fundamentally different capability than what humans mean by 'being good at math'",
    },
    "hellaswag": {
        "name": "Dr. Priya Nair",
        "title": "Cognitive Systems Engineer, Common Sense Reasoning Specialist",
        "years": 12,
        "background": [
            "PhD in Computer Science from CMU, Human-Computer Interaction focus",
            "Worked on commonsense knowledge bases at Cycorp and Allen Institute for AI",
            "Led evaluation of situational reasoning in autonomous vehicle systems",
            "Consultant for DARPA's Explainable AI program",
        ],
        "specialization": "common sense reasoning evaluation, situational awareness assessment, inferring unstated constraints in physical and social scenarios",
        "battlescars": [
            "found that a commercial AI assistant's 'common sense' failed catastrophically in medical dosing scenarios",
            "identified systematic gaps in world knowledge that correlated with cultural background of training data",
        ],
        "thinking_style": "asks 'what implicit facts does this situation depend on that aren't stated?' — always considers what a 10-year-old would know that the AI might miss",
        "strong_opinion": "current common sense benchmarks measure situations where humans have extensive training data, not genuine physical reasoning",
    },
    "truthfulqa": {
        "name": "Dr. James Okafor",
        "title": "Science Communication Researcher and AI Truthfulness Auditor",
        "years": 15,
        "background": [
            "PhD in Science Communication from UC Berkeley",
            "Fact-checked for major publications for 8 years",
            "Developed truthfulness rubrics adopted by three major AI labs",
            "Published seminal work on how language models hedge and obscure uncertainty",
        ],
        "specialization": "evaluating factual accuracy, detecting hedging and epistemic ambiguity, assessing whether AI outputs express appropriate confidence levels",
        "battlescars": [
            "caught a major AI company publishing a model card with fabricated benchmark results",
            "discovered systematic overconfidence in medical knowledge tasks in three state-of-the-art models",
        ],
        "thinking_style": "triangulates claims against multiple authoritative sources, checks primary literature before accepting secondary summaries, flagshedging as a red flag for uncertainty",
        "strong_opinion": "most 'truthfulness' benchmarks measure whether an AI repeats accurate facts, not whether it avoids repeating inaccurate ones — these are very different capabilities",
    },
    "arc": {
        "name": "Dr. Elena Kowalski",
        "title": "Experimental Physicist and Scientific Reasoning Evaluator",
        "years": 18,
        "background": [
            "PhD in Experimental Physics from Caltech, postdoctoral work at CERN",
            "Led experimental design education programs at Fermilab",
            "Co-authored Nature paper on scientific reasoning in AI systems",
            "Reviewer for Physical Review journals for 10 years",
        ],
        "specialization": "evaluating causal reasoning, experimental design analysis, identifying confounds in observational vs. experimental data, assessing statistical reasoning quality",
        "battlescars": [
            "identified a high-profile AI chemistry paper where training data contaminated test set results",
            "caught systematic misinterpretation of statistical significance in a major drug trial re-analysis",
        ],
        "thinking_style": "asks 'what would it take to actually prove this claim?' — always looks for the control group, the alternative hypothesis, and the effect size",
        "strong_opinion": "AI systems are remarkably good at mimicking the language of scientific reasoning without understanding the epistemological foundations that make science work",
    },
}


def get_all_tasks() -> list[Task]:
    """Return all tasks pooled across domains."""
    all_tasks = []
    for domain_tasks in DOMAINS.values():
        all_tasks.extend(domain_tasks)
    return all_tasks


def get_tasks_by_domain(domain: str) -> list[Task]:
    """Return tasks for a specific domain."""
    return DOMAINS.get(domain, [])


def format_baseline_prompt(task: Task) -> str:
    """Format a task as a baseline (no expert prompt) sub-agent prompt."""
    return (
        f"TASK:\n{task.question}\n\n"
        f"OUTPUT FORMAT:\n"
        f"1. Your answer or recommendation\n"
        f"2. Step-by-step reasoning or justification\n"
        f"3. Confidence level (0-100%)\n"
        f"4. Any assumptions or uncertainties clearly stated"
    )


def format_expert_prompt(task: Task, expert_identity: dict) -> str:
    """Format a task as an expert-prompted sub-agent prompt."""
    bio = expert_identity
    identity_block = (
        f"You are {bio['name']}, {bio['title']} with {bio['years']} years of experience.\n"
        f"Your background: {'; '.join(bio['background'])}.\n"
        f"You specialize in: {bio['specialization']}.\n"
        f"You have personally experienced: {'; '.join(bio['battlescars'])}.\n"
        f"You think: {bio['thinking_style']}.\n"
        f"You believe firmly that: {bio['strong_opinion']}"
    )
    return (
        f"{identity_block}\n\n"
        f"YOUR TASK:\n{task.question}\n\n"
        f"APPROACH:\n"
        f"- Identify the governing principle or failure mode first\n"
        f"- Evaluate alternatives or counterarguments before concluding\n"
        f"- State your confidence level with explicit reasoning\n"
        f"- Flag information that would change your conclusion\n\n"
        f"OUTPUT FORMAT:\n"
        f"1. Your answer or recommendation\n"
        f"2. Step-by-step reasoning or justification\n"
        f"3. Confidence level (0-100%)\n"
        f"4. Any assumptions or uncertainties clearly stated"
    )

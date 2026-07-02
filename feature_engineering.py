"""
feature_engineering.py — Candidate feature extraction for the AI Recruiter pipeline.

Computes structured signals used by the scoring engine:
    - Career progression score
    - Founder mindset score
    - Product DNA fraction
    - Skill freshness coefficient
    - Notice period and location modifiers
    - Behavioral engagement multiplier
"""

# Standard library
from datetime import datetime

# Local
from utils import parse_date


# IT services companies (career at these penalises product DNA score)
SERVICES_COMPANIES = {
    "tcs", "wipro", "infosys", "accenture", "cognizant", "capgemini",
    "tech mahindra", "hcl", "mphasis", "mindtree", "genpact", "l&t",
    "tata consultancy",
}

# Maps title keywords to an ordinal seniority rank
TITLE_RANKS = {
    "intern":     1,
    "associate":  2,
    "junior":     2,
    "engineer":   3,
    "developer":  3,
    "analyst":    3,
    "senior":     4,
    "lead":       5,
    "manager":    5,
    "staff":      5,
    "principal":  5,
    "director":   6,
    "founder":    6,
    "cto":        6,
    "ceo":        6,
    "head":       6,
}


def get_title_rank(title):
    """Returns the ordinal seniority rank for a job title string."""
    if not title:
        return 2
    title_lower = title.lower()
    for keyword, rank in TITLE_RANKS.items():
        if keyword in title_lower:
            return rank
    return 3  # default: standard IC engineer


def calculate_career_progression(history):
    """
    Computes a career progression score (0–100).

    Rewards upward title trajectory and penalises extreme job-hopping
    (average tenure < 18 months). Bonuses for average tenure > 36 months.
    """
    if not history:
        return 50

    # Sort career history chronologically (oldest first)
    sorted_history = sorted(
        [(parse_date(j.get("start_date")), j) for j in history if j.get("start_date")],
        key=lambda x: x[0] or datetime.min,
    )

    if not sorted_history:
        return 50

    ranks = [get_title_rank(item[1].get("title")) for item in sorted_history]

    rank_growth = 0.0
    if len(ranks) > 1:
        growth_steps = sum(
            1 if ranks[i] > ranks[i - 1] else (-1 if ranks[i] < ranks[i - 1] else 0)
            for i in range(1, len(ranks))
        )
        rank_growth = growth_steps / (len(ranks) - 1)

    total_months = sum(j.get("duration_months", 0) for j in history)
    avg_tenure_months = total_months / len(history) if history else 0

    prog_score = 50 + (rank_growth * 30)

    # Penalise frequent job-hopping; reward loyalty
    if avg_tenure_months < 18:
        prog_score -= (18 - avg_tenure_months) * 2
    elif avg_tenure_months > 36:
        prog_score += min(15, (avg_tenure_months - 36) * 0.5)

    return max(0, min(100, prog_score))


def calculate_founder_mindset(candidate):
    """
    Computes a Founder Mindset score (0–100).

    Based on leadership titles, startup/hacker vocabulary in job descriptions,
    and GitHub activity. Missing GitHub linkage applies a small penalty.
    """
    score = 40  # baseline

    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})

    has_leadership_title = any(
        kw in job.get("title", "").lower()
        for job in history
        for kw in ("founder", "cto", "co-founder", "founding", "lead")
    )
    if has_leadership_title:
        score += 20

    startup_keywords = [
        "startup", "founding", "prototype", "shipped", "launched",
        "mvp", "zero to one", "ownership", "scale-up", "open-source", "diy",
    ]
    keyword_matches = sum(
        kw in job.get("description", "").lower()
        for job in history
        for kw in startup_keywords
    )
    score += min(20, keyword_matches * 3)

    github_score = signals.get("github_activity_score", -1)
    if github_score > 0:
        score += (github_score / 100.0) * 20
    elif github_score == -1:
        score -= 10  # no GitHub profile linked

    return max(0, min(100, score))


def calculate_product_dna(history):
    """
    Returns the fraction of career duration (0–1) spent at product companies.

    Companies present in SERVICES_COMPANIES are treated as IT services and
    excluded from the product-months count.
    """
    if not history:
        return 0.5

    total_months = 0
    product_months = 0

    for job in history:
        comp = job.get("company", "").lower()
        dur = job.get("duration_months", 0)
        total_months += dur

        is_service = any(s in comp for s in SERVICES_COMPANIES)
        if not is_service:
            product_months += dur

    return product_months / total_months if total_months > 0 else 0.5


def get_skill_freshness(skills, history):
    """
    Returns a freshness coefficient (0.6–1.0) indicating how recently
    the candidate has applied their listed skills.

    Checks whether skill names appear in the current (or most recent) job
    description and title.
    """
    if not skills:
        return 0.5

    current_job = next((j for j in history if j.get("is_current")), None)
    if not current_job and history:
        current_job = sorted(history, key=lambda x: x.get("start_date", ""), reverse=True)[0]

    recent_skills = set()
    if current_job:
        desc  = current_job.get("description", "").lower()
        title = current_job.get("title", "").lower()
        for sk in skills:
            name = sk.get("name", "").lower()
            if name in desc or name in title:
                recent_skills.add(name)

    fresh_ratio = len(recent_skills) / len(skills) if skills else 0
    return 0.6 + (fresh_ratio * 0.4)


def calculate_notice_period_modifier(notice_days):
    """
    Returns a multiplier (0.1–1.0) based on notice period.

    ≤ 30 days → 1.0 (best), > 90 days → 0.1 (heavy penalty).
    """
    if notice_days <= 30:
        return 1.0
    elif notice_days <= 60:
        return 0.7
    elif notice_days <= 90:
        return 0.4
    else:
        return 0.1


def calculate_location_modifier(profile):
    """
    Returns a location preference multiplier (0.3–1.0).

    Noida / Pune / NCR preferred (1.0). Tier-1 Indian cities eligible for
    relocation (0.8). Outside India penalised (0.3).
    """
    loc     = profile.get("location", "").lower()
    country = profile.get("country", "").lower()

    preferred_cities = ["noida", "pune", "delhi", "gurgaon", "ncr", "ghaziabad", "faridabad"]
    tier1_cities     = ["bangalore", "bengaluru", "hyderabad", "mumbai", "chennai", "kolkata"]

    if any(city in loc for city in preferred_cities):
        return 1.0
    elif any(city in loc for city in tier1_cities):
        return 0.8  # relocation feasible
    elif country != "india":
        return 0.3
    else:
        return 0.5


def calculate_behavioral_multiplier(signals, benchmark_date_str="2026-06-22"):
    """
    Computes a behavioral engagement multiplier (0.1–1.3).

    Factors:
        - Login recency (inactive > 6 months → 0.3×)
        - Recruiter response rate (< 15% → 0.4×)
        - Open-to-work flag (active → 1.15×)
    """
    mult = 1.0

    last_act = signals.get("last_active_date")
    if last_act:
        try:
            days_inactive = (
                datetime.strptime(benchmark_date_str, "%Y-%m-%d")
                - datetime.strptime(last_act, "%Y-%m-%d")
            ).days
            if days_inactive > 180:
                mult *= 0.3
            elif days_inactive > 90:
                mult *= 0.6
            elif days_inactive > 30:
                mult *= 0.85
        except Exception:
            pass

    rrr = signals.get("recruiter_response_rate", 0)
    if rrr < 0.15:
        mult *= 0.4
    elif rrr < 0.40:
        mult *= 0.75

    if signals.get("open_to_work_flag"):
        mult *= 1.15

    return min(1.3, max(0.1, mult))

"""
scoring_engine.py — Multi-signal scoring aggregation for the AI Recruiter pipeline.

Combines feature scores into a final relevance score using configurable
persona weights, then computes independent risk and confidence signals.
"""

# Third-party
import numpy as np

# Local
import config
from feature_engineering import (
    calculate_behavioral_multiplier,
    calculate_career_progression,
    calculate_founder_mindset,
    calculate_location_modifier,
    calculate_product_dna,
    get_skill_freshness,
)


def evaluate_skill_match(candidate_skills, target_skills):
    """
    Scores a candidate's skill alignment against the JD target skills.

    Proficiency multipliers: beginner 0.5 × | intermediate 0.8 × |
                             advanced 1.2 × | expert 1.5 ×
    Duration multiplier: < 12 months → 0.8 ×, > 36 months → 1.2 ×.

    Returns:
        skill_score (float): Normalised score out of 100.
        matched_list (list[str]): Names of matched target skills.
    """
    if not candidate_skills or not target_skills:
        return 0.0, []

    # Maximum achievable score: all target skills at expert + long duration
    max_possible = sum(target_skills.values()) * 1.5 * 1.2

    score        = 0.0
    matched_list = []

    for sk in candidate_skills:
        name = sk.get("name", "")
        if not name:
            continue

        prof = sk.get("proficiency", "beginner")
        dur  = sk.get("duration_months", 0) or 0

        target_wt = next(
            (wt for t_sk, wt in target_skills.items() if t_sk.lower() == name.lower()),
            None,
        )
        if target_wt is None:
            continue

        matched_list.append(name)

        prof_mult = {"beginner": 0.5, "intermediate": 0.8,
                     "advanced": 1.2, "expert": 1.5}.get(prof, 0.5)
        dur_mult  = 1.2 if dur > 36 else (0.8 if dur < 12 else 1.0)

        score += target_wt * prof_mult * dur_mult

    normalized_score = min(100.0, (score / max_possible) * 100) if max_possible > 0 else 0.0
    return normalized_score, matched_list


def evaluate_experience_score(years_exp):
    """
    Maps years of experience to a score (0–100).

    Sweet spot: 5–10 years → 100.
    < 5 years → linear ramp from 10 (0 yrs) to 100 (5 yrs).
    > 10 years → overqualification decay; floor at 40.
    """
    if not years_exp or years_exp < 0:
        return 10.0
    if 5.0 <= years_exp <= 10.0:
        return 100.0
    elif years_exp < 5.0:
        return max(10.0, 10.0 + (years_exp / 5.0) * 90.0)
    else:
        return max(40.0, 100.0 - (years_exp - 10.0) * 7.0)


def compute_risk_score(candidate, matched_skills):
    """
    Computes a risk score (0–100) from role-relevant friction signals.

    Components (max pts):
        Notice period    30  — penalised only when > 30 days
        Location         20  — penalised for non-preferred cities
        Job-hopping      25  — average tenure < 18 months
        Skill freshness  15  — skills not used in recent roles
        Profile gaps     10  — severely incomplete profiles
    """
    risk = 0.0

    profile = candidate.get("profile",         {}) or {}
    history = candidate.get("career_history",  []) or []
    skills  = candidate.get("skills",          []) or []
    signals = candidate.get("redrob_signals",  {}) or {}

    # Notice period risk (max 30 pts)
    notice_days = signals.get("notice_period_days", 0) or 0
    if notice_days > 90:
        risk += 30.0
    elif notice_days > 60:
        risk += 15.0
    elif notice_days > 30:
        risk += 5.0

    # Location / relocation friction (max 20 pts)
    location_score = calculate_location_modifier(profile)
    if location_score < 0.5:
        risk += 20.0
    elif location_score < 1.0:
        risk += 8.0

    # Job-hopping instability (max 25 pts)
    if history:
        durations  = [j.get("duration_months", 0) or 0 for j in history]
        avg_tenure = sum(durations) / len(durations)
        if len(history) >= 3 and avg_tenure < 18:
            risk += 25.0
        elif len(history) >= 2 and avg_tenure < 24:
            risk += 10.0

    # Skill freshness gap (max 15 pts)
    if skills:
        freshness = get_skill_freshness(skills, history)
        if freshness < 0.7:
            risk += 15.0
        elif freshness < 0.85:
            risk += 5.0

    # Profile incompleteness (max 10 pts)
    completeness = signals.get("profile_completeness_score", 50) or 50
    if completeness < 40:
        risk += 10.0
    elif completeness < 60:
        risk += 4.0

    return min(100.0, risk)


def compute_confidence_score(candidate):
    """
    Computes system confidence in the quality of the match data (0–100%).

    Intentionally DECOUPLED from relevance — a low-relevance candidate
    can have HIGH confidence if their profile is complete and verified.

    Components (max pts):
        Profile completeness   35
        Verified identity      35  (email 15, phone 12, LinkedIn 8)
        Career data richness   20  (job descriptions + skills depth)
        Social signals         10  (connections + endorsements)
    """
    score   = 0.0
    signals = candidate.get("redrob_signals",  {}) or {}
    history = candidate.get("career_history",  []) or []
    skills  = candidate.get("skills",          []) or []

    completeness = signals.get("profile_completeness_score", 0) or 0
    score += (completeness / 100.0) * 35.0

    if signals.get("verified_email"):   score += 15.0
    if signals.get("verified_phone"):   score += 12.0
    if signals.get("linkedin_connected"): score += 8.0

    jobs_with_desc = sum(1 for j in history if j.get("description", ""))
    score += min(12.0, jobs_with_desc * 3.0)
    score += 8.0 if len(skills) >= 5 else (4.0 if len(skills) >= 2 else 0.0)

    connections  = signals.get("connection_count",      0) or 0
    endorsements = signals.get("endorsements_received", 0) or 0
    score += 5.0 if connections > 100  else (2.0 if connections > 20  else 0.0)
    score += 5.0 if endorsements > 10  else (2.0 if endorsements > 2  else 0.0)

    return min(100.0, score)


def compute_candidate_score(candidate, hybrid_similarity, target_skills,
                            persona="startup_founder", jd_weights=None):
    """
    Full scoring aggregation.

    JD-parsed weights take precedence over persona defaults.
    Final score = weighted relevance × behavioral_mult − risk_penalty.

    Returns a dict with keys:
        relevance, risk, confidence, raw_relevance_minus_risk,
        matched_skills, components.
    """
    profile  = candidate.get("profile",         {}) or {}
    history  = candidate.get("career_history",  []) or []
    skills   = candidate.get("skills",          []) or []
    signals  = candidate.get("redrob_signals",  {}) or {}

    # ── Feature computations ─────────────────────────────────────────────────
    progression_score    = calculate_career_progression(history)
    founder_mindset_score = calculate_founder_mindset(candidate)
    product_dna          = calculate_product_dna(history)       # fraction 0–1

    years_exp        = float(profile.get("years_of_experience", 0) or 0)
    experience_score = evaluate_experience_score(years_exp)

    skill_score, matched_skills = evaluate_skill_match(skills, target_skills)

    similarity_score = hybrid_similarity * 100.0
    product_score    = product_dna * 100.0

    # ── Weight selection: JD-parsed weights > persona defaults ───────────────
    weights = jd_weights if jd_weights else config.PERSONAS[persona]["weights"]

    components = {
        "semantic_match":     similarity_score,
        "skills":             skill_score,
        "career_growth":      progression_score,
        "founder_mindset":    founder_mindset_score,
        "product_experience": product_score,
    }

    # ── Weighted relevance (normalised to 0–100) ─────────────────────────────
    relevance_score   = 0.0
    active_weight_sum = 0.0
    for key, wt in weights.items():
        if key in components:
            relevance_score   += components[key] * wt
            active_weight_sum += wt

    if active_weight_sum > 0:
        relevance_score /= active_weight_sum

    # ── Behavioral multiplier (activity, response rate, open-to-work) ────────
    behavioral_mult  = calculate_behavioral_multiplier(signals)
    relevance_score *= behavioral_mult

    # ── Risk & Confidence (independently computed) ───────────────────────────
    risk_score       = compute_risk_score(candidate, matched_skills)
    confidence_score = compute_confidence_score(candidate)

    # ── Final: relevance penalised by risk (cap at 15% of max relevance) ─────
    risk_penalty          = risk_score * 0.15
    relevance_minus_risk  = max(0.0, relevance_score - risk_penalty)

    return {
        "relevance":               relevance_score,
        "risk":                    risk_score,
        "confidence":              confidence_score,
        "raw_relevance_minus_risk": relevance_minus_risk,
        "matched_skills":          matched_skills,
        "components": {
            "semantic":    similarity_score,
            "skills":      skill_score,
            "progression": progression_score,
            "founder":     founder_mindset_score,
            "product":     product_score,
            "experience":  experience_score,
        },
    }

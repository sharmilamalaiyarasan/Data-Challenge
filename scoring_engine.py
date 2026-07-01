import numpy as np
import config
from feature_engineering import (
    calculate_career_progression, calculate_founder_mindset, calculate_product_dna,
    get_skill_freshness, calculate_notice_period_modifier, calculate_location_modifier,
    calculate_behavioral_multiplier
)


def evaluate_skill_match(candidate_skills, target_skills):
    """
    Evaluates candidate's skill alignment with target skills from Job Description.
    Returns:
        skill_score (float): Score out of 100.
        matched_list (list of str): List of matching skills.
    """
    if not candidate_skills or not target_skills:
        return 0.0, []

    score = 0.0
    matched_list = []

    # Maximum possible score (all target skills at expert proficiency + long duration)
    max_possible = sum(target_skills.values()) * 1.5 * 1.2  # expert * long-duration

    for sk in candidate_skills:
        name = sk.get("name", "")
        if not name:
            continue
        prof = sk.get("proficiency", "beginner")
        dur = sk.get("duration_months", 0) or 0

        # Case-insensitive match
        target_wt = None
        for t_sk, wt in target_skills.items():
            if t_sk.lower() == name.lower():
                target_wt = wt
                matched_list.append(name)
                break

        if target_wt is not None:
            # Proficiency multipliers
            prof_mult = {"beginner": 0.5, "intermediate": 0.8,
                         "advanced": 1.2, "expert": 1.5}.get(prof, 0.5)

            # Duration multiplier — capped at 1.2x for 36+ months
            dur_mult = 1.0
            if dur > 36:
                dur_mult = 1.2
            elif dur < 12:
                dur_mult = 0.8

            score += target_wt * prof_mult * dur_mult

    normalized_score = min(100.0, (score / max_possible) * 100) if max_possible > 0 else 0.0
    return normalized_score, matched_list


def evaluate_experience_score(years_exp):
    """
    Score experience: sweet spot 5-10 years = 100.
    Gracefully handles 0 or None.
    """
    if not years_exp or years_exp < 0:
        return 10.0
    if 5.0 <= years_exp <= 10.0:
        return 100.0
    elif years_exp < 5.0:
        # Linear: 0yr=10, 5yr=100
        return max(10.0, 10.0 + (years_exp / 5.0) * 90.0)
    else:
        # Overqualification decay: 15yr ~ 50, never below 40
        return max(40.0, 100.0 - (years_exp - 10.0) * 7.0)


def compute_risk_score(candidate, matched_skills):
    """
    Computes a risk score (0-100).
    Reflects only factors supported by the dataset and relevant to the role.
    """
    risk = 0.0

    profile = candidate.get("profile", {}) or {}
    history = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    signals = candidate.get("redrob_signals", {}) or {}

    # 1. Notice period risk (max 30 pts) — only if > 30 days
    notice_days = signals.get("notice_period_days", 0) or 0
    if notice_days > 90:
        risk += 30.0
    elif notice_days > 60:
        risk += 15.0
    elif notice_days > 30:
        risk += 5.0

    # 2. Location/relocation friction (max 20 pts)
    location_score = calculate_location_modifier(profile)
    if location_score < 0.5:
        risk += 20.0
    elif location_score < 1.0:
        risk += 8.0

    # 3. Job-hopping instability (max 25 pts)
    if history:
        durations = [j.get("duration_months", 0) or 0 for j in history]
        avg_tenure = sum(durations) / len(durations)
        if len(history) >= 3 and avg_tenure < 18:
            risk += 25.0
        elif len(history) >= 2 and avg_tenure < 24:
            risk += 10.0

    # 4. Skill freshness (max 15 pts)
    if skills:
        freshness = get_skill_freshness(skills, history)
        if freshness < 0.7:
            risk += 15.0
        elif freshness < 0.85:
            risk += 5.0

    # 5. Profile incompleteness (max 10 pts) — only penalize severely incomplete
    completeness = signals.get("profile_completeness_score", 50) or 50
    if completeness < 40:
        risk += 10.0
    elif completeness < 60:
        risk += 4.0

    return min(100.0, risk)


def compute_confidence_score(candidate):
    """
    Computes system confidence in the quality of data behind the match (0-100%).
    Intentionally DECOUPLED from relevance score — a low-relevance candidate
    can have HIGH confidence if their profile is complete and verified.
    """
    score = 0.0

    signals = candidate.get("redrob_signals", {}) or {}
    history = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []

    # Profile completeness (max 35 pts)
    completeness = signals.get("profile_completeness_score", 0) or 0
    score += (completeness / 100.0) * 35.0

    # Verified identity signals (max 35 pts)
    if signals.get("verified_email"):
        score += 15.0
    if signals.get("verified_phone"):
        score += 12.0
    if signals.get("linkedin_connected"):
        score += 8.0

    # Career data richness (max 20 pts)
    # At least 2 jobs with descriptions = good evidence base
    jobs_with_desc = sum(1 for j in history if j.get("description", ""))
    score += min(12.0, jobs_with_desc * 3.0)
    # Skills richness
    if len(skills) >= 5:
        score += 8.0
    elif len(skills) >= 2:
        score += 4.0

    # Social signals (max 10 pts) — connections & endorsements
    connections = signals.get("connection_count", 0) or 0
    endorsements = signals.get("endorsements_received", 0) or 0
    if connections > 100:
        score += 5.0
    elif connections > 20:
        score += 2.0
    if endorsements > 10:
        score += 5.0
    elif endorsements > 2:
        score += 2.0

    return min(100.0, score)


def compute_candidate_score(candidate, hybrid_similarity, target_skills,
                             persona="startup_founder", jd_weights=None):
    """
    Full scoring aggregation.
    Returns a dict of relevance, risk, confidence, components, and final score.
    """
    profile  = candidate.get("profile", {}) or {}
    history  = candidate.get("career_history", []) or []
    skills   = candidate.get("skills", []) or []
    signals  = candidate.get("redrob_signals", {}) or {}

    # ── Feature computations ─────────────────────────────────────────────────
    progression_score    = calculate_career_progression(history)
    founder_mindset_score = calculate_founder_mindset(candidate)
    product_dna          = calculate_product_dna(history)     # fraction 0..1

    years_exp       = float(profile.get("years_of_experience", 0) or 0)
    experience_score = evaluate_experience_score(years_exp)

    skill_score, matched_skills = evaluate_skill_match(skills, target_skills)

    similarity_score = hybrid_similarity * 100.0
    product_score    = product_dna * 100.0

    # ── Weight selection: JD-parsed weights > persona defaults ───────────────
    weights = jd_weights if jd_weights else config.PERSONAS[persona]["weights"]

    components = {
        "semantic_match":    similarity_score,
        "skills":            skill_score,
        "career_growth":     progression_score,
        "founder_mindset":   founder_mindset_score,
        "product_experience": product_score,
    }

    # ── Weighted relevance score ─────────────────────────────────────────────
    relevance_score  = 0.0
    active_weight_sum = 0.0
    for key, wt in weights.items():
        if key in components:
            relevance_score += components[key] * wt
            active_weight_sum += wt

    if active_weight_sum > 0:
        relevance_score /= active_weight_sum   # normalize to 0-100

    # ── Behavioral multiplier (activity, response rate, open-to-work) ────────
    behavioral_mult = calculate_behavioral_multiplier(signals)
    relevance_score *= behavioral_mult

    # ── Risk & Confidence (independently computed) ───────────────────────────
    risk_score       = compute_risk_score(candidate, matched_skills)
    confidence_score = compute_confidence_score(candidate)

    # ── Final score: relevance penalised by risk ─────────────────────────────
    # Risk contribution capped at 15% of max relevance (prevents over-penalisation)
    risk_penalty = risk_score * 0.15
    relevance_minus_risk = max(0.0, relevance_score - risk_penalty)

    return {
        "relevance":              relevance_score,
        "risk":                   risk_score,
        "confidence":             confidence_score,
        "raw_relevance_minus_risk": relevance_minus_risk,
        "matched_skills":         matched_skills,
        "components": {
            "semantic":    similarity_score,
            "skills":      skill_score,
            "progression": progression_score,
            "founder":     founder_mindset_score,
            "product":     product_score,
            "experience":  experience_score,
        }
    }

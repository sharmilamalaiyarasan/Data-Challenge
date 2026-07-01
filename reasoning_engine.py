"""
reasoning_engine.py — Explainability layer for the AI Recruiter pipeline.

Produces:
  1. A structured JSON breakdown (for Streamlit XAI UI)
  2. A factual, specific CSV reasoning string derived only from candidate data
"""

import json


def get_star_rating(score_out_of_100: float) -> int:
    """Converts a 0-100 score into 1-5 integer stars."""
    val = round(score_out_of_100 / 20.0)
    return max(1, min(5, int(val)))


def _bar(value: float, max_val: float = 50.0, width: int = 10) -> str:
    """ASCII progress bar for feature contribution display."""
    if max_val <= 0:
        filled = 0
    else:
        filled = min(width, int(round((value / max_val) * width)))
    return "#" * filled + "." * (width - filled)


def build_feature_contribution(scores_dict: dict, weights: dict = None) -> list:
    """
    Returns an ordered list of (label, raw_contribution, pct_of_total) tuples
    for the feature contribution chart in the Streamlit UI.
    """
    comps = scores_dict.get("components", {})
    relevance = scores_dict.get("relevance", 0.0)
    risk = scores_dict.get("risk", 0.0)

    # Approximate per-component raw contribution
    # (component_score * weight) / sum_weights, then scale to relevance
    labels = [
        ("Semantic Match",    comps.get("semantic", 0)),
        ("Technical Skills",  comps.get("skills", 0)),
        ("Experience",        comps.get("experience", 0)),
        ("Career Growth",     comps.get("progression", 0)),
        ("Founder Mindset",   comps.get("founder", 0)),
        ("Product Pedigree",  comps.get("product", 0)),
    ]

    total_raw = sum(v for _, v in labels) or 1.0
    result = []
    for label, val in labels:
        contribution = round((val / total_raw) * relevance, 1)
        result.append({
            "label":        label,
            "value":        round(val, 1),
            "contribution": contribution,
        })

    # Risk penalty as a negative entry
    risk_penalty = round(risk * 0.15, 1)
    result.append({
        "label":        "Risk Penalty",
        "value":        -risk_penalty,
        "contribution": -risk_penalty,
    })

    return result


def generate_explainability(candidate: dict, scores_dict: dict,
                             validity_score: float, invalid_reasons: list) -> dict:
    """
    Constructs a structured JSON API output and a factual CSV reasoning string.
    """
    cid     = candidate.get("candidate_id", "unknown")
    profile = candidate.get("profile", {}) or {}
    signals = candidate.get("redrob_signals", {}) or {}
    history = candidate.get("career_history", []) or []
    skills  = candidate.get("skills", []) or []

    # ── Candidate facts ──────────────────────────────────────────────────────
    years_exp   = float(profile.get("years_of_experience", 0) or 0)
    curr_title  = profile.get("current_title", "Engineer") or "Engineer"
    notice_days = int(signals.get("notice_period_days", 0) or 0)
    location    = profile.get("location", "") or ""
    matched     = scores_dict.get("matched_skills", [])

    # ── Invalid profile shortcut ─────────────────────────────────────────────
    if validity_score == 0.0:
        reason_str = (
            f"Profile failed logical consistency validation: "
            f"{'; '.join(invalid_reasons[:2])}."
            if invalid_reasons else
            "Profile failed logical consistency validation."
        )
        return {
            "candidate_id":  cid,
            "overall_score": 0.0,
            "confidence":    0.0,
            "breakdown": {
                "skills": 1, "experience": 1, "career": 1,
                "culture": 1, "availability": 1, "risk": "High"
            },
            "strengths":          [],
            "weaknesses":         invalid_reasons or ["Logical consistency failure"],
            "recommendation":     "Not Recommended",
            "feature_contributions": [],
            "csv_reasoning":      reason_str,
        }

    # ── Valid candidate ──────────────────────────────────────────────────────
    relevance  = scores_dict.get("relevance", 0.0)
    risk       = scores_dict.get("risk", 0.0)
    confidence = scores_dict.get("confidence", 0.0)
    final      = scores_dict.get("raw_relevance_minus_risk", 0.0)
    comps      = scores_dict.get("components", {})

    # Star ratings
    skills_stars  = get_star_rating(comps.get("skills", 0))
    exp_stars     = get_star_rating(comps.get("experience", 0))
    career_stars  = get_star_rating(comps.get("progression", 0))
    culture_stars = get_star_rating(comps.get("founder", 0))

    # Availability (notice period → score)
    avail_raw   = max(0.0, 100.0 - (notice_days / 150.0) * 100.0)
    if signals.get("open_to_work_flag"):
        avail_raw = min(100.0, avail_raw + 10.0)
    avail_stars = get_star_rating(avail_raw)

    # Risk label
    risk_label = "Low" if risk < 20.0 else ("High" if risk > 50.0 else "Medium")

    # ── Strengths (evidence-backed only) ─────────────────────────────────────
    strengths = []
    if comps.get("semantic", 0) > 60:
        strengths.append(
            f"Strong semantic alignment with JD ({comps['semantic']:.0f}/100)"
        )
    if matched:
        skill_str = ", ".join(matched[:3])
        strengths.append(f"Matched target skills: {skill_str}")
    if comps.get("progression", 0) > 70:
        strengths.append("Upward career trajectory across roles")
    if comps.get("founder", 0) > 70:
        strengths.append("Demonstrated startup/leadership mindset")
    if comps.get("product", 0) > 75:
        strengths.append("Predominantly product-company background")
    if signals.get("open_to_work_flag"):
        strengths.append("Actively open to opportunities")
    if notice_days <= 30 and notice_days >= 0:
        strengths.append(f"Immediate or short notice ({notice_days} days)")

    # ── Weaknesses (evidence-backed only) ────────────────────────────────────
    weaknesses = []
    if notice_days > 90:
        weaknesses.append(f"Long notice period of {notice_days} days")
    elif notice_days > 60:
        weaknesses.append(f"Notice period of {notice_days} days")
    if risk_label == "High":
        avg_t = (sum(j.get("duration_months", 0) or 0 for j in history)
                 / len(history)) if history else 0
        if avg_t < 18:
            weaknesses.append(
                f"Frequent job changes (avg tenure {avg_t:.0f} months)"
            )
    if not matched:
        weaknesses.append("No direct skill overlap with JD requirements")
    if comps.get("experience", 0) < 40:
        weaknesses.append(
            f"{years_exp:.1f} years experience outside preferred 5-10 year range"
        )

    # ── Recommendation ───────────────────────────────────────────────────────
    if final >= 75:
        recommendation = "Highly Recommended"
    elif final >= 55:
        recommendation = "Recommended"
    elif final >= 35:
        recommendation = "Consider with Reservations"
    else:
        recommendation = "Not Recommended"

    # ── Feature contributions (for bar chart) ────────────────────────────────
    feature_contribs = build_feature_contribution(scores_dict)

    # ── CSV reasoning — factual and specific ─────────────────────────────────
    # Pattern: <Title> with <X> yrs exp; <primary strength>; <primary concern if any>.
    title_part    = f"{curr_title} with {years_exp:.1f} yrs experience"
    strength_part = strengths[0].lower() if strengths else "profile data available"
    concern_part  = (f"; concern: {weaknesses[0].lower()}"
                     if weaknesses else "")
    avail_part    = (f"; notice period {notice_days}d"
                     if notice_days > 0 else "; immediately available")

    csv_reasoning = (
        f"{title_part}; {strength_part}{avail_part}{concern_part}."
    )
    # Ensure first character is uppercase
    csv_reasoning = csv_reasoning[0].upper() + csv_reasoning[1:]
    # Truncate to 500 chars to stay well within CSV limits
    if len(csv_reasoning) > 500:
        csv_reasoning = csv_reasoning[:497] + "..."

    return {
        "candidate_id":          cid,
        "overall_score":         round(final, 2),
        "confidence":            round(confidence, 2),
        "breakdown": {
            "skills":       skills_stars,
            "experience":   exp_stars,
            "career":       career_stars,
            "culture":      culture_stars,
            "availability": avail_stars,
            "risk":         risk_label,
        },
        "strengths":              strengths,
        "weaknesses":             weaknesses,
        "recommendation":         recommendation,
        "feature_contributions":  feature_contribs,
        "csv_reasoning":          csv_reasoning,
    }

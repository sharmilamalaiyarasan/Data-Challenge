"""
validator.py — Logical consistency checks for candidate profiles.

Detects honeypot / fabricated profiles by flagging date contradictions,
skill-duration fraud, and timeline inflations.
"""

from datetime import datetime

from utils import parse_date


def validate_candidate(candidate, benchmark_date_str="2026-06-22"):
    """
    Performs logical consistency checks on a candidate profile.

    Returns:
        validity_score (float): 1.0 if valid, 0.0 if contradictions are found.
        reasons (list[str]): Detailed list of detected issues.
    """
    reasons = []

    profile   = candidate.get("profile", {})
    history   = candidate.get("career_history", [])
    skills    = candidate.get("skills", [])
    education = candidate.get("education", [])
    signals   = candidate.get("redrob_signals", {})

    benchmark_date = datetime.strptime(benchmark_date_str, "%Y-%m-%d")
    years_exp = float(profile.get("years_of_experience", 0))

    # 1. Current job duration mismatch (allow 3-month buffer)
    for idx, job in enumerate(history):
        if job.get("is_current"):
            start_str = job.get("start_date")
            stated_dur = job.get("duration_months", 0)
            if start_str:
                try:
                    s_dt = datetime.strptime(start_str, "%Y-%m-%d")
                    calculated_dur = (
                        (benchmark_date.year - s_dt.year) * 12
                        + (benchmark_date.month - s_dt.month)
                    )
                    if abs(calculated_dur - stated_dur) > 3:
                        reasons.append(
                            f"Current job duration mismatch: stated {stated_dur} mos vs "
                            f"calculated {calculated_dur} mos (start {start_str} → benchmark)"
                        )
                except Exception:
                    pass

    # 2. Skill duration fraud: "expert" proficiency with 0 months of usage
    zero_dur_expert_skills = [
        sk.get("name")
        for sk in skills
        if sk.get("proficiency") == "expert" and sk.get("duration_months", -1) == 0
    ]
    if len(zero_dur_expert_skills) >= 3:
        reasons.append(
            f"Skill duration fraud: Expert level claimed in {len(zero_dur_expert_skills)} "
            f"skills with exactly 0 months of usage: {zero_dur_expert_skills}"
        )

    # 3. Total career history exceeds stated experience by > 3 years
    total_history_yrs = sum(j.get("duration_months", 0) for j in history) / 12.0
    if total_history_yrs > years_exp + 3.0:
        reasons.append(
            f"Stated experience timeline inflation: total history ({total_history_yrs:.1f} yrs) "
            f"exceeds stated experience ({years_exp:.1f} yrs)"
        )

    # 4. Single job duration exceeds total stated experience
    for idx, job in enumerate(history):
        job_dur_yrs = job.get("duration_months", 0) / 12.0
        if job_dur_yrs > years_exp + 0.5:
            reasons.append(
                f"Job {idx} duration ({job_dur_yrs:.1f} yrs) exceeds total "
                f"stated experience ({years_exp:.1f} yrs)"
            )

    # 5. Future date anomalies and reversed date ranges
    for idx, job in enumerate(history):
        start_str = job.get("start_date")
        end_str   = job.get("end_date")

        if start_str:
            s_dt = parse_date(start_str)
            if s_dt and s_dt > benchmark_date:
                reasons.append(f"Job {idx} start date {start_str} is in the future")
            if end_str:
                e_dt = parse_date(end_str)
                if e_dt and e_dt > benchmark_date:
                    reasons.append(f"Job {idx} end date {end_str} is in the future")
                if s_dt and e_dt and s_dt > e_dt:
                    reasons.append(f"Job {idx} start date {start_str} is after end date {end_str}")

    # 6. Education graduation timeline contradictions
    for idx, edu in enumerate(education):
        start = edu.get("start_year")
        end   = edu.get("end_year")
        if start and end and start > end:
            reasons.append(f"Education {idx} start year {start} is after graduation year {end}")

    validity_score = 0.0 if reasons else 1.0
    return validity_score, reasons

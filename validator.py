from datetime import datetime
from utils import parse_date, get_months_between

def validate_candidate(candidate, benchmark_date_str="2026-06-22"):
    """
    Performs logical consistency checks on candidate profiles to filter out
    invalid / simulated profiles (honeypots).
    Returns:
        validity_score (float): 1.0 if valid, 0.0 if logical inconsistencies are found.
        reasons (list of str): Detailed list of detected contradictions.
    """
    reasons = []
    
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])
    signals = candidate.get("redrob_signals", {})
    
    benchmark_date = datetime.strptime(benchmark_date_str, "%Y-%m-%d")
    
    # 1. Stated Years of Experience vs. Details
    years_exp = float(profile.get("years_of_experience", 0))
    
    # 2. Check current job duration mismatch
    for idx, job in enumerate(history):
        if job.get("is_current"):
            start_str = job.get("start_date")
            stated_dur = job.get("duration_months", 0)
            if start_str:
                try:
                    s_dt = datetime.strptime(start_str, "%Y-%m-%d")
                    calculated_dur = (benchmark_date.year - s_dt.year) * 12 + (benchmark_date.month - s_dt.month)
                    # Allow 3 months of buffer/grace
                    if abs(calculated_dur - stated_dur) > 3:
                        reasons.append(
                            f"Current job duration mismatch: stated {stated_dur} mos vs "
                            f"calculated {calculated_dur} mos (from start date {start_str} to benchmark)"
                        )
                except Exception as e:
                    pass

    # 3. Check for skill duration fraud: "expert" proficiency with 0 months used
    zero_dur_expert_count = 0
    zero_dur_expert_skills = []
    for sk in skills:
        if sk.get("proficiency") == "expert" and sk.get("duration_months", -1) == 0:
            zero_dur_expert_count += 1
            zero_dur_expert_skills.append(sk.get("name"))
            
    if zero_dur_expert_count >= 3:
        reasons.append(
            f"Skill duration fraud: Expert level claimed in {zero_dur_expert_count} "
            f"skills with exactly 0 months of usage: {zero_dur_expert_skills}"
        )

    # 4. Experience timeline vs Stated overall experience
    total_history_months = sum(j.get("duration_months", 0) for j in history)
    total_history_yrs = total_history_months / 12.0
    if total_history_yrs > years_exp + 3.0:
        reasons.append(
            f"Stated experience timeline inflation: total history duration ({total_history_yrs:.1f} yrs) "
            f"exceeds stated years of experience ({years_exp:.1f} yrs)"
        )
        
    for idx, job in enumerate(history):
        job_dur_yrs = job.get("duration_months", 0) / 12.0
        if job_dur_yrs > years_exp + 0.5:
            reasons.append(
                f"Job {idx} duration ({job_dur_yrs:.1f} yrs) exceeds total "
                f"stated years of experience ({years_exp:.1f} yrs)"
            )

    # 5. Future date anomalies
    for idx, job in enumerate(history):
        start_str = job.get("start_date")
        end_str = job.get("end_date")
        
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
        end = edu.get("end_year")
        if start and end and start > end:
            reasons.append(f"Education {idx} start year {start} is after graduation year {end}")

    # Calculate final validity score
    validity_score = 0.0 if len(reasons) > 0 else 1.0
    return validity_score, reasons

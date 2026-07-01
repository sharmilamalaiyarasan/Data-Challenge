from datetime import datetime
import re
from utils import parse_date

# Services firms list for DNA checking
SERVICES_COMPANIES = {
    "tcs", "wipro", "infosys", "accenture", "cognizant", "capgemini", 
    "tech mahindra", "hcl", "mphasis", "mindtree", "genpact", "l&t", "tata consultancy"
}

TITLE_RANKS = {
    "intern": 1,
    "associate": 2,
    "junior": 2,
    "engineer": 3,
    "developer": 3,
    "analyst": 3,
    "senior": 4,
    "lead": 5,
    "manager": 5,
    "staff": 5,
    "principal": 5,
    "director": 6,
    "founder": 6,
    "cto": 6,
    "ceo": 6,
    "head": 6
}

def get_title_rank(title):
    if not title:
        return 2
    title_lower = title.lower()
    for keyword, rank in TITLE_RANKS.items():
        if keyword in title_lower:
            return rank
    return 3 # default rank for standard engineers

def calculate_career_progression(history):
    """
    Computes a career progression score (0-100).
    Looks at title rank increases and average job tenure.
    """
    if not history:
        return 50
    
    # Sort history by start date (oldest first)
    sorted_history = []
    for job in history:
        start_str = job.get("start_date")
        if start_str:
            dt = parse_date(start_str)
            if dt:
                sorted_history.append((dt, job))
    sorted_history.sort(key=lambda x: x[0])
    
    if not sorted_history:
        return 50
        
    ranks = [get_title_rank(item[1].get("title")) for item in sorted_history]
    
    # Career progression logic
    rank_growth = 0
    if len(ranks) > 1:
        # Check if rank increases over time
        growth_steps = 0
        for i in range(1, len(ranks)):
            if ranks[i] > ranks[i-1]:
                growth_steps += 1
            elif ranks[i] < ranks[i-1]:
                growth_steps -= 1
        rank_growth = growth_steps / (len(ranks) - 1)
        
    # Average job tenure in months
    total_months = sum(j.get("duration_months", 0) for j in history)
    avg_tenure_months = total_months / len(history) if len(history) > 0 else 0
    
    # Compute base progression score
    prog_score = 50 + (rank_growth * 30)
    
    # Tenure/stability adjustment
    # Penalize extreme job hopping (< 18 months average tenure)
    if avg_tenure_months < 18:
        prog_score -= (18 - avg_tenure_months) * 2
    # Reward long stays (> 36 months average tenure)
    elif avg_tenure_months > 36:
        prog_score += min(15, (avg_tenure_months - 36) * 0.5)
        
    return max(0, min(100, prog_score))

def calculate_founder_mindset(candidate):
    """
    Computes a Founder Mindset score (0-100) based on startup key terms,
    GitHub contributions, leadership roles, and side projects.
    """
    score = 40 # baseline
    
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})
    
    # Check job titles for founder/leadership
    has_leadership_title = False
    for job in history:
        t = job.get("title", "").lower()
        if "founder" in t or "cto" in t or "co-founder" in t or "founding" in t or "lead" in t:
            has_leadership_title = True
            
    if has_leadership_title:
        score += 20
        
    # Check descriptions for hacker/startup keywords
    startup_keywords = [
        "startup", "founding", "prototype", "shipped", "launched", 
        "mvp", "zero to one", "ownership", "scale-up", "open-source", "diy"
    ]
    keyword_matches = 0
    for job in history:
        desc = job.get("description", "").lower()
        for kw in startup_keywords:
            if kw in desc:
                keyword_matches += 1
                
    score += min(20, keyword_matches * 3)
    
    # Github activity contribution
    github_score = signals.get("github_activity_score", -1)
    if github_score > 0:
        score += (github_score / 100.0) * 20
    elif github_score == -1:
        # no Github linked penalty
        score -= 10
        
    return max(0, min(100, score))

def calculate_product_dna(history):
    """
    Returns the percentage of candidate's career duration spent at product companies
    versus IT services companies.
    """
    if not history:
        return 0.5
        
    total_months = 0
    product_months = 0
    
    for job in history:
        comp = job.get("company", "").lower()
        dur = job.get("duration_months", 0)
        total_months += dur
        
        # Check if company is in services list
        is_service = False
        for s_comp in SERVICES_COMPANIES:
            if s_comp in comp:
                is_service = True
                break
                
        if not is_service:
            product_months += dur
            
    if total_months == 0:
        return 0.5
        
    return product_months / total_months

def get_skill_freshness(skills, history):
    """
    Evaluates skill freshness by checking the overlap of key skills in recent jobs.
    """
    # Simply mapping: if candidate has skill assessments or has used skills recently
    # For this challenge, we check if skills have positive duration and endorsements.
    # Return a freshness coefficient (0.5 to 1.0)
    if not skills:
        return 0.5
    
    recent_skills = set()
    # Sort history to find the current or last job
    if history:
        # Get the current job or the one with latest start_date
        current_job = next((j for j in history if j.get("is_current")), None)
        if not current_job:
            # get job with max start date
            sorted_jobs = sorted(history, key=lambda x: x.get("start_date", ""), reverse=True)
            current_job = sorted_jobs[0]
            
        desc = current_job.get("description", "").lower()
        title = current_job.get("title", "").lower()
        
        # Look for skills in current job description/title
        for sk in skills:
            name = sk.get("name", "").lower()
            if name in desc or name in title:
                recent_skills.add(name)
                
    fresh_ratio = len(recent_skills) / len(skills) if len(skills) > 0 else 0
    return 0.6 + (fresh_ratio * 0.4)

def calculate_notice_period_modifier(notice_days):
    """Score notice period: sub-30 days is best (1.0). 150 days is worst (0.1)."""
    if notice_days <= 30:
        return 1.0
    elif notice_days <= 60:
        return 0.7
    elif notice_days <= 90:
        return 0.4
    else:
        # 120 or 150 days notice period gets heavily penalized
        return 0.1

def calculate_location_modifier(profile):
    """
    Noida/Pune preferred. Relocation candidates from Tier-1 welcome.
    """
    loc = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    
    preferred_cities = ["noida", "pune", "delhi", "gurgaon", "ncr", "ghaziabad", "faridabad"]
    tier1_cities = ["bangalore", "bengaluru", "hyderabad", "mumbai", "chennai", "kolkata"]
    
    # Noida/Pune/NCR
    if any(city in loc for city in preferred_cities):
        return 1.0
    # Tier-1 Indian cities
    elif any(city in loc for city in tier1_cities):
        # Relocation potential
        return 0.8
    # Outside India
    elif country != "india":
        return 0.3
    else:
        return 0.5

def calculate_behavioral_multiplier(signals, benchmark_date_str="2026-06-22"):
    """
    Computes a multiplier based on platform log-in active status, response rate, etc.
    """
    mult = 1.0
    
    # 1. Login Recency
    last_act = signals.get("last_active_date")
    if last_act:
        try:
            la_dt = datetime.strptime(last_act, "%Y-%m-%d")
            benchmark = datetime.strptime(benchmark_date_str, "%Y-%m-%d")
            days_inactive = (benchmark - la_dt).days
            
            if days_inactive > 180: # > 6 months inactive
                mult *= 0.3
            elif days_inactive > 90: # > 3 months inactive
                mult *= 0.6
            elif days_inactive > 30: # > 1 month inactive
                mult *= 0.85
        except:
            pass
            
    # 2. Recruiter Response Rate
    rrr = signals.get("recruiter_response_rate", 0)
    # Low response rates down-weight availability
    if rrr < 0.15:
        mult *= 0.4
    elif rrr < 0.40:
        mult *= 0.75
        
    # 3. Open to work flag
    if signals.get("open_to_work_flag"):
        mult *= 1.15
        
    return min(1.3, max(0.1, mult))

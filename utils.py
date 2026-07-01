from datetime import datetime
import re

def parse_date(date_str):
    """Safely parses a YYYY-MM-DD date string."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d")
    except ValueError:
        return None

def get_months_between(start_str, end_str, benchmark_str="2026-06-22"):
    """Calculates duration in months between two date strings."""
    s_dt = parse_date(start_str)
    if not s_dt:
        return 0
    
    if not end_str:
        e_dt = parse_date(benchmark_str)
    else:
        e_dt = parse_date(end_str)
        
    if not e_dt or s_dt > e_dt:
        return 0
        
    return (e_dt.year - s_dt.year) * 12 + (e_dt.month - s_dt.month)

def clean_text(text):
    """Cleans and normalizes text for indexing or similarity engines."""
    if not text:
        return ""
    text = text.lower()
    # Replace non-alphanumeric chars with spaces (preserving some tech chars like . or - or #)
    text = re.sub(r"[^a-zA-Z0-9\.\-\#\s]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_years_from_education(start_year, end_year):
    """Returns the duration of education program in years."""
    if start_year and end_year:
        return max(0, int(end_year) - int(start_year))
    return 0

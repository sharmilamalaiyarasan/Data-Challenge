import os

# System seed for reproducibility
RANDOM_SEED = 42

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "candidates.jsonl")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
SUBMISSION_FILE = os.path.join(BASE_DIR, "submission.csv")

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cache file paths
EMBEDDING_CACHE_FILE = os.path.join(CACHE_DIR, "candidate_embeddings.npy")
ROLE_RELEVANCE_CACHE_FILE = os.path.join(CACHE_DIR, "role_embeddings.npy")
FEATURES_CACHE_FILE = os.path.join(CACHE_DIR, "candidate_features.pkl")
SKILLS_CACHE_FILE = os.path.join(CACHE_DIR, "normalized_skills.pkl")
BM25_CACHE_FILE = os.path.join(CACHE_DIR, "bm25_index.pkl")

# Benchmarking Date (used for experience/duration calculations)
BENCHMARK_DATE_STR = "2026-06-22"

# Recruiter Personas & Weights
PERSONAS = {
    "startup_founder": {
        "description": "Prioritizes startup pedigree, growth velocity, and a hacker/builder mindset.",
        "weights": {
            "semantic_match": 0.20,
            "skills": 0.20,
            "career_growth": 0.15,
            "founder_mindset": 0.15,
            "product_experience": 0.15,
            "role_relevance": 0.15
        }
    },
    "enterprise_recruiter": {
        "description": "Prioritizes stability, tenure, certifications, and traditional education tiering.",
        "weights": {
            "semantic_match": 0.15,
            "skills": 0.25,
            "career_growth": 0.10,
            "founder_mindset": 0.05,
            "product_experience": 0.10,
            "stability": 0.20,
            "role_relevance": 0.15
        }
    }
}


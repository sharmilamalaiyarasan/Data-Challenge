"""
parser.py — Job Description parser for the AI Recruiter pipeline.

Extracts target skills (with dynamic weight multipliers) and scoring
weight adjustments by analysing keyword frequency in the JD text.
"""

import re


def parse_job_description(jd_text):
    """
    Parses the job description text and returns core parameters.

    Returns a dict with:
        target_skills (dict): Skill name → weight multiplier.
        weights (dict): Component weights for the scoring engine.
        weights_adjustment (dict): Alias of weights (backwards compat).
        title_keywords (list[str]): Role-relevant title terms.
    """
    jd_lower = jd_text.lower()

    # Skill category keyword counts drive dynamic weight boosts below.
    skill_categories = {
        "embeddings":        ["embedding", "sentence-transformers", "openai embeddings",
                              "bge", "e5", "retrieval"],
        "vector_db":         ["pinecone", "weaviate", "qdrant", "milvus", "opensearch",
                              "elasticsearch", "faiss", "vector database", "hybrid search"],
        "evaluation":        ["ndcg", "mrr", "map", "evaluation framework",
                              "a/b testing", "offline-to-online"],
        "fine_tuning":       ["fine-tuning", "lora", "qlora", "peft", "llm"],
        "learning_to_rank":  ["learning-to-rank", "xgboost", "neural ranking"],
    }

    focus_scores = {
        category: sum(1 for term in terms if term in jd_lower)
        for category, terms in skill_categories.items()
    }

    # Baseline skill weights for standard AI engineering skills
    target_skills = {
        "Embeddings":     1.0,
        "Vector Search":  1.0,
        "Python":         1.0,
        "NDCG":           1.0,
        "NLP":            0.8,
        "Fine-tuning LLMs": 0.8,
        "LoRA":           0.8,
        "PEFT":           0.8,
        "XGBoost":        0.6,
        "SQL":            0.5,
        "Airflow":        0.5,
        "Kafka":          0.5,
        "Spark":          0.5,
    }

    # Dynamically boost weights based on JD keyword frequency
    if focus_scores.get("embeddings", 0) > 0:
        target_skills["Embeddings"] = 1.2
        target_skills["NLP"] = 1.0
    if focus_scores.get("vector_db", 0) > 0:
        target_skills["Vector Search"] = 1.2
    if focus_scores.get("evaluation", 0) > 0:
        target_skills["NDCG"] = 1.2
    if focus_scores.get("fine_tuning", 0) > 0:
        target_skills["Fine-tuning LLMs"] = 1.1
        target_skills["LoRA"] = 1.0

    # Detect startup/high-velocity culture signals; boost founder mindset weight
    # when ≥ 3 of these terms appear (e.g. "founding team", "scrappy", "seed").
    founder_counts = sum(
        1 for w in ["founding", "startup", "scrappy", "shipper", "fast", "seed", "series a"]
        if w in jd_lower
    )

    if founder_counts >= 3:
        weights_adjustment = {
            "founder_mindset":    0.25,
            "semantic_match":     0.25,
            "skills":             0.25,
            "career_growth":      0.15,
            "product_experience": 0.10,
        }
    else:
        weights_adjustment = {
            "founder_mindset":    0.15,
            "semantic_match":     0.25,
            "skills":             0.30,
            "career_growth":      0.15,
            "product_experience": 0.15,
        }

    title_keywords = [
        "ai", "ml", "machine learning", "nlp", "search",
        "retrieval", "ranking", "data scientist",
    ]

    return {
        "target_skills":      target_skills,
        "weights":            weights_adjustment,   # primary key read by pipeline.py
        "weights_adjustment": weights_adjustment,   # kept for backwards compatibility
        "title_keywords":     title_keywords,
    }


# Alias so pipeline.py can do: from parser import parse_jd
parse_jd = parse_job_description

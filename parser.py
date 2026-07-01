import re

def parse_job_description(jd_text):
    """
    Parses the job description text and returns core parameters:
    - target_skills (dict of skill name -> weight multiplier)
    - weights_adjustment (dict of weights adjusting scorer components)
    - title_keywords (list of strings)
    """
    jd_lower = jd_text.lower()
    
    # 1. Target Skills and dynamic weights
    # We identify categories of skills and count their occurrences in JD
    skill_categories = {
        "embeddings": ["embedding", "sentence-transformers", "openai embeddings", "bge", "e5", "retrieval"],
        "vector_db": ["pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", "faiss", "vector database", "hybrid search"],
        "evaluation": ["ndcg", "mrr", "map", "evaluation framework", "a/b testing", "offline-to-online"],
        "fine_tuning": ["fine-tuning", "lora", "qlora", "peft", "llm"],
        "learning_to_rank": ["learning-to-rank", "xgboost", "neural ranking"]
    }
    
    focus_scores = {}
    for category, terms in skill_categories.items():
        count = sum(1 for term in terms if term in jd_lower)
        focus_scores[category] = count
        
    # Standard AI engineering skills
    target_skills = {
        "Embeddings": 1.0,
        "Vector Search": 1.0,
        "Python": 1.0,
        "NDCG": 1.0,
        "NLP": 0.8,
        "Fine-tuning LLMs": 0.8,
        "LoRA": 0.8,
        "PEFT": 0.8,
        "XGBoost": 0.6,
        "SQL": 0.5,
        "Airflow": 0.5,
        "Kafka": 0.5,
        "Spark": 0.5
    }
    
    # Dynamically scale skill weights based on JD counts
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
        
    # Dynamic Scoring Weights adjustments based on JD focus
    # If the JD mentions "founding team", "startup", "fast-paced" frequently, boost founder mindset
    founder_counts = sum(1 for w in ["founding", "startup", "scrappy", "shipper", "fast", "seed", "series a"] if w in jd_lower)
    
    weights_adjustment = {}
    if founder_counts >= 3:
        # Increase founder mindset weight, reduce academic or stable weights
        weights_adjustment["founder_mindset"] = 0.25
        weights_adjustment["semantic_match"] = 0.25
        weights_adjustment["skills"] = 0.25
        weights_adjustment["career_growth"] = 0.15
        weights_adjustment["product_experience"] = 0.10
    else:
        # standard balanced profile
        weights_adjustment["founder_mindset"] = 0.15
        weights_adjustment["semantic_match"] = 0.25
        weights_adjustment["skills"] = 0.30
        weights_adjustment["career_growth"] = 0.15
        weights_adjustment["product_experience"] = 0.15
        
    title_keywords = ["ai", "ml", "machine learning", "nlp", "search", "retrieval", "ranking", "data scientist"]
    
    return {
        "target_skills": target_skills,
        "weights": weights_adjustment,          # pipeline.py reads "weights"
        "weights_adjustment": weights_adjustment,  # kept for backwards compat
        "title_keywords": title_keywords
    }


# Alias so pipeline.py can do: from parser import parse_jd
parse_jd = parse_job_description

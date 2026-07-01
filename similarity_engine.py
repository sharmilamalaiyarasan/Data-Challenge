import os
import pickle
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import config
from utils import clean_text

def tokenize_corpus(texts):
    """Simple whitespace tokenization of cleaned text."""
    return [clean_text(t).split() for t in texts]

def build_or_load_bm25(candidates_texts):
    """Loads a cached BM25 index or builds and caches it."""
    cache_path = config.BM25_CACHE_FILE
    
    if os.path.exists(cache_path):
        print(f"Loading BM25 index from cache: {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading BM25 index: {e}. Rebuilding...")
            
    # Rebuild index
    print("Building BM25 Okapi index...")
    tokenized_corpus = tokenize_corpus(candidates_texts)
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open(cache_path, 'wb') as f:
        pickle.dump(bm25, f)
    print(f"Saved BM25 index to {cache_path}")
    return bm25

def extract_bm25_query_terms(jd_text, target_skills, title_keywords):
    """
    Constructs a highly focused list of search terms from target skills,
    title keywords, and top unique terms in the JD to speed up BM25.
    """
    # Start with title keywords and target skills (lowercased)
    query_terms = []
    seen = set()
    
    for kw in title_keywords:
        kw_clean = kw.lower()
        if kw_clean not in seen:
            query_terms.append(kw_clean)
            seen.add(kw_clean)
            
    for skill in target_skills.keys():
        skill_clean = skill.lower()
        # Handle multi-word skills
        for word in re.findall(r'[a-zA-Z0-9\-\#\+]+', skill_clean):
            if word not in seen and len(word) > 1:
                query_terms.append(word)
                seen.add(word)
                
    # Also extract some content terms from JD, skipping common stop words
    words = re.findall(r'[a-zA-Z0-9\-\#\+]+', jd_text.lower())
    stop_words = {
        "the", "and", "a", "of", "to", "in", "is", "for", "that", "with", "on", "as", "by", "at", 
        "an", "be", "this", "from", "are", "it", "you", "we", "our", "us", "their", "they", "or", 
        "will", "have", "has", "had", "can", "could", "would", "should", "your", "my", "me", 
        "experience", "work", "job", "description", "candidate", "role", "team", "engineer", 
        "skills", "knowledge", "ability", "responsibilities", "requirements", 
        "qualifications", "company", "position", "preferred", "required", "years", "working", 
        "strong", "excellent", "good", "great", "fast", "environment", "growth", "opportunity",
        "about", "all", "also", "any", "but", "do", "if", "into", "more", "no", "not", "other", 
        "some", "such", "than", "then", "there", "these", "up", "very", "who", "which", "design",
        "develop", "build", "create", "implement", "manage", "lead", "support", "maintain", 
        "collaborate", "deliver", "highly", "hands-on", "relevant", "field", "degree", "bs", "ms",
        "phd", "computer", "science", "engineering", "technology", "systems", "solutions",
        "applications", "projects", "production", "tools", "methods", "best", "practices"
    }
    
    for w in words:
        if len(w) > 2 and w not in stop_words and w not in seen:
            query_terms.append(w)
            seen.add(w)
            if len(query_terms) >= 40:  # limit to top 40 terms total
                break
                
    return query_terms

def get_hybrid_similarities(candidates_texts, candidate_embeddings, jd_text, embedding_model, 
                            target_skills=None, title_keywords=None):
    """
    Computes a hybrid similarity score using:
    - 50% Sentence Transformers similarity
    - 30% BM25 similarity (using focused query terms)
    - 20% TF-IDF similarity
    """
    num_candidates = len(candidates_texts)
    
    # 1. Sentence Transformers Embedding Cosine Similarity (50%)
    print("Computing Sentence Transformer similarity...")
    jd_embedding = embedding_model.encode([jd_text], convert_to_numpy=True).astype(np.float32)
    # Cosine similarity
    jd_norm = jd_embedding / np.linalg.norm(jd_embedding, axis=1, keepdims=True)
    cand_norms = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    transformer_scores = np.dot(cand_norms, jd_norm.T).squeeze()
    
    # Normalize score to 0..1
    transformer_scores = (transformer_scores - transformer_scores.min()) / (transformer_scores.max() - transformer_scores.min() + 1e-8)
    
    # 2. BM25 Okapi Search Score (30%)
    print("Computing BM25 similarity...")
    bm25 = build_or_load_bm25(candidates_texts)
    
    # Use target skills and title keywords if available to keep query short & fast
    if target_skills is not None and title_keywords is not None:
        tokenized_query = extract_bm25_query_terms(jd_text, target_skills, title_keywords)
    else:
        # Fallback to simple extraction
        cleaned_jd = clean_text(jd_text)
        tokenized_query = cleaned_jd.split()[:40]
        
    print(f"  BM25 query terms count: {len(tokenized_query)}")
    bm25_scores = np.array(bm25.get_scores(tokenized_query))
    
    # Normalize BM25 score to 0..1
    if bm25_scores.max() > bm25_scores.min():
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
    else:
        bm25_scores = np.zeros(num_candidates)
        
    # 3. TF-IDF Cosine Similarity (20%)
    print("Computing TF-IDF similarity...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(candidates_texts)
    
    # Construct a cleaned version of the query for TF-IDF as well
    cleaned_jd = clean_text(jd_text)
    query_tfidf = vectorizer.transform([cleaned_jd])
    tfidf_scores = cosine_similarity(tfidf_matrix, query_tfidf).squeeze()
    
    # Normalize TF-IDF score to 0..1
    if tfidf_scores.max() > tfidf_scores.min():
        tfidf_scores = (tfidf_scores - tfidf_scores.min()) / (tfidf_scores.max() - tfidf_scores.min() + 1e-8)
    else:
        tfidf_scores = np.zeros(num_candidates)
        
    # Combine scores with weights
    hybrid_scores = (0.5 * transformer_scores) + (0.3 * bm25_scores) + (0.2 * tfidf_scores)
    return hybrid_scores

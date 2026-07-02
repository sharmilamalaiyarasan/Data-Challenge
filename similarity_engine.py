"""
similarity_engine.py — Hybrid similarity scoring for the AI Recruiter pipeline.

Blends three complementary retrieval signals:
    50%  Sentence Transformers (dense semantic similarity)
    30%  BM25 Okapi           (lexical term frequency)
    20%  TF-IDF cosine         (sparse bag-of-words)
"""

# Standard library
import os
import pickle
import re

# Third-party
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Local
import config
from utils import clean_text


def tokenize_corpus(texts):
    """Whitespace-tokenises a list of cleaned text strings."""
    return [clean_text(t).split() for t in texts]


def build_or_load_bm25(candidates_texts):
    """Loads a cached BM25Okapi index from disk, or builds and caches it."""
    cache_path = config.BM25_CACHE_FILE

    if os.path.exists(cache_path):
        print(f"Loading BM25 index from cache: {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"BM25 cache load error: {e}. Rebuilding…")

    print("Building BM25 Okapi index…")
    bm25 = BM25Okapi(tokenize_corpus(candidates_texts))

    with open(cache_path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"Saved BM25 index → {cache_path}")
    return bm25


def extract_bm25_query_terms(jd_text, target_skills, title_keywords):
    """
    Builds a focused BM25 query (≤ 40 terms) from target skills, title keywords,
    and high-signal JD content words (after stop-word removal).
    """
    query_terms = []
    seen = set()

    for kw in title_keywords:
        kw_clean = kw.lower()
        if kw_clean not in seen:
            query_terms.append(kw_clean)
            seen.add(kw_clean)

    for skill in target_skills.keys():
        for word in re.findall(r"[a-zA-Z0-9\-\#\+]+", skill.lower()):
            if word not in seen and len(word) > 1:
                query_terms.append(word)
                seen.add(word)

    stop_words = {
        "the", "and", "a", "of", "to", "in", "is", "for", "that", "with", "on", "as",
        "by", "at", "an", "be", "this", "from", "are", "it", "you", "we", "our", "us",
        "their", "they", "or", "will", "have", "has", "had", "can", "could", "would",
        "should", "your", "my", "me", "experience", "work", "job", "description",
        "candidate", "role", "team", "engineer", "skills", "knowledge", "ability",
        "responsibilities", "requirements", "qualifications", "company", "position",
        "preferred", "required", "years", "working", "strong", "excellent", "good",
        "great", "fast", "environment", "growth", "opportunity", "about", "all", "also",
        "any", "but", "do", "if", "into", "more", "no", "not", "other", "some", "such",
        "than", "then", "there", "these", "up", "very", "who", "which", "design",
        "develop", "build", "create", "implement", "manage", "lead", "support",
        "maintain", "collaborate", "deliver", "highly", "hands-on", "relevant", "field",
        "degree", "bs", "ms", "phd", "computer", "science", "engineering", "technology",
        "systems", "solutions", "applications", "projects", "production", "tools",
        "methods", "best", "practices",
    }

    for w in re.findall(r"[a-zA-Z0-9\-\#\+]+", jd_text.lower()):
        if len(w) > 2 and w not in stop_words and w not in seen:
            query_terms.append(w)
            seen.add(w)
            if len(query_terms) >= 40:
                break

    return query_terms


def get_hybrid_similarities(candidates_texts, candidate_embeddings, jd_text,
                            embedding_model, target_skills=None, title_keywords=None):
    """
    Computes per-candidate hybrid similarity scores (blended 50/30/20).

    Args:
        candidates_texts:      List of assembled candidate text strings.
        candidate_embeddings:  Pre-computed float32 embedding matrix (N × D).
        jd_text:               Raw job description text.
        embedding_model:       Loaded SentenceTransformer instance.
        target_skills:         Dict of skill → weight (from parser).
        title_keywords:        List of role-relevant title strings (from parser).

    Returns:
        np.ndarray of shape (N,) with normalised hybrid scores in [0, 1].
    """
    num_candidates = len(candidates_texts)

    # ── 50%: Sentence Transformer dense similarity ────────────────────────────
    print("Computing Sentence Transformer similarity…")
    jd_embedding = embedding_model.encode([jd_text], convert_to_numpy=True).astype(np.float32)

    jd_norm   = jd_embedding / np.linalg.norm(jd_embedding, axis=1, keepdims=True)
    cand_norm = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    transformer_scores = np.dot(cand_norm, jd_norm.T).squeeze()
    transformer_scores = (
        (transformer_scores - transformer_scores.min())
        / (transformer_scores.max() - transformer_scores.min() + 1e-8)
    )

    # ── 30%: BM25 lexical similarity ─────────────────────────────────────────
    print("Computing BM25 similarity…")
    bm25 = build_or_load_bm25(candidates_texts)

    if target_skills is not None and title_keywords is not None:
        tokenized_query = extract_bm25_query_terms(jd_text, target_skills, title_keywords)
    else:
        tokenized_query = clean_text(jd_text).split()[:40]

    print(f"  BM25 query term count: {len(tokenized_query)}")
    bm25_scores = np.array(bm25.get_scores(tokenized_query))
    if bm25_scores.max() > bm25_scores.min():
        bm25_scores = (
            (bm25_scores - bm25_scores.min())
            / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        )
    else:
        bm25_scores = np.zeros(num_candidates)

    # ── 20%: TF-IDF sparse cosine similarity ─────────────────────────────────
    print("Computing TF-IDF similarity…")
    vectorizer   = TfidfVectorizer(stop_words="english", max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(candidates_texts)
    query_tfidf  = vectorizer.transform([clean_text(jd_text)])
    tfidf_scores = cosine_similarity(tfidf_matrix, query_tfidf).squeeze()
    if tfidf_scores.max() > tfidf_scores.min():
        tfidf_scores = (
            (tfidf_scores - tfidf_scores.min())
            / (tfidf_scores.max() - tfidf_scores.min() + 1e-8)
        )
    else:
        tfidf_scores = np.zeros(num_candidates)

    return (0.5 * transformer_scores) + (0.3 * bm25_scores) + (0.2 * tfidf_scores)

"""
pipeline.py — Master orchestrator for the AI Recruiter Ranking System.

Coordinates:
  1. Loading candidates from JSONL / JSON
  2. Logical validation (validity_score: 1.0 or 0.0)
  3. Candidate text assembly with disk caching
  4. Pre-computed or on-demand Sentence Transformer embeddings
  5. Hybrid similarity scoring (ST 50% + BM25 30% + TF-IDF 20%)
  6. Feature extraction and dynamic scoring
  7. Explainability / reasoning generation
  8. Deterministic sort and return of ranked result dicts
"""

# Standard library
import json
import os
import pickle

# Third-party
import numpy as np
from tqdm import tqdm

# Local
import config
from embedding_engine import assemble_candidate_text, build_or_load_embeddings, build_or_load_role_embeddings

from parser import parse_jd
from reasoning_engine import generate_explainability
from scoring_engine import compute_candidate_score
from similarity_engine import get_hybrid_similarities
from utils import clean_text
from validator import validate_candidate


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_candidates(candidates_path: str) -> list:
    """Reads a .jsonl or .json file and returns a list of candidate dicts."""
    candidates = []
    if candidates_path.endswith(".json"):
        with open(candidates_path, "r", encoding="utf-8") as f:
            candidates = json.load(f)
    else:
        with open(candidates_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        candidates.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    print(f"  Loaded {len(candidates):,} candidates from {candidates_path}")
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Candidate Text Cache
# ─────────────────────────────────────────────────────────────────────────────

def build_or_load_candidate_texts(candidates: list) -> list:
    """
    Assembles a textual representation for every candidate.

    Results are pickled to disk so subsequent runs skip this step entirely.
    """
    texts_cache = os.path.join(config.CACHE_DIR, "candidate_texts.pkl")

    if os.path.exists(texts_cache):
        print("  Loading candidate texts from cache…")
        with open(texts_cache, "rb") as f:
            return pickle.load(f)

    print("  Assembling candidate text representations…")
    texts = [assemble_candidate_text(c) for c in tqdm(candidates, desc="  Texts")]

    with open(texts_cache, "wb") as f:
        pickle.dump(texts, f)
    print(f"  Saved candidate texts cache → {texts_cache}")
    return texts


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    candidates_path: str,
    jd_text: str,
    persona: str = "startup_founder",
    top_n: int = 100,
) -> list:
    """
    Full ranking pipeline.

    Returns a list of result dicts sorted by:
      1. final_score descending
      2. candidate_id ascending (deterministic tie-break)
    """
    print("\n" + "=" * 60)
    print("  AI RECRUITER — RANKING PIPELINE")
    print("=" * 60)

    # ── Step 1: Load Candidates ──────────────────────────────────────────────
    print("\n[1/7] Loading candidates…")
    candidates = load_candidates(candidates_path)

    # ── Step 2: Parse JD ────────────────────────────────────────────────────
    print("\n[2/7] Parsing Job Description…")
    jd_config     = parse_jd(jd_text)
    target_skills = jd_config.get("target_skills", {})
    jd_weights    = jd_config.get("weights", None)
    print(f"  Target skills identified: {list(target_skills.keys())[:10]}")

    # ── Step 3: Logical Validation ───────────────────────────────────────────
    print("\n[3/7] Validating candidate profiles…")
    validity_results = {}
    for c in tqdm(candidates, desc="  Validating"):
        cid = c.get("candidate_id")
        validity_results[cid] = validate_candidate(c)

    invalid_count = sum(1 for v, _ in validity_results.values() if v == 0.0)
    print(f"  {invalid_count:,} candidates flagged as logically inconsistent")

    # ── Step 4: Assemble Candidate Texts ─────────────────────────────────────
    print("\n[4/7] Assembling candidate text representations…")
    candidate_texts = build_or_load_candidate_texts(candidates)

    # ── Step 5: Embeddings & Hybrid Similarity ───────────────────────────────
    print("\n[5/7] Computing hybrid similarity scores…")
    model, candidate_embeddings = build_or_load_embeddings(candidates, candidate_texts)

    hybrid_scores = get_hybrid_similarities(
        candidates_texts=candidate_texts,
        candidate_embeddings=candidate_embeddings,
        jd_text=jd_text,
        embedding_model=model,
        target_skills=target_skills,
        title_keywords=jd_config.get("title_keywords", []),
    )
    print(f"  Hybrid similarity computed for {len(hybrid_scores):,} candidates.")

    # ── Step 5.5: Compute Role Relevance ─────────────────────────────────────
    print("\n[5.5/7] Computing semantic role relevance scores…")
    role_embeddings = build_or_load_role_embeddings(candidates, model=model)
    target_role_descriptor = jd_config.get("target_role_descriptor", "Senior AI Engineer")
    print(f"  Target role descriptor for embedding comparison: '{target_role_descriptor}'")
    target_role_emb = model.encode([target_role_descriptor], convert_to_numpy=True).astype(np.float32)

    # Cosine similarity
    target_role_norm = target_role_emb / np.linalg.norm(target_role_emb, axis=1, keepdims=True)
    role_norms = role_embeddings / np.linalg.norm(role_embeddings, axis=1, keepdims=True)
    role_relevance_scores = np.dot(role_norms, target_role_norm.T).squeeze()
    role_relevance_scores = np.clip(role_relevance_scores, 0.0, 1.0)

    # ── Step 6: Scoring & Feature Extraction ─────────────────────────────────
    print("\n[6/7] Scoring and extracting features…")
    results = []

    for i, candidate in enumerate(tqdm(candidates, desc="  Scoring")):
        cid = candidate.get("candidate_id")
        validity_score, invalid_reasons = validity_results.get(cid, (1.0, []))

        if validity_score == 0.0:
            # Invalid candidates are force-zeroed; skip expensive feature computation
            explanation = generate_explainability(
                candidate=candidate,
                scores_dict={
                    "relevance": 0.0,
                    "risk": 100.0,
                    "confidence": 0.0,
                    "raw_relevance_minus_risk": 0.0,
                    "matched_skills": [],
                    "components": {
                        "semantic": 0.0, "skills": 0.0, "progression": 0.0,
                        "founder": 0.0,  "product": 0.0, "experience":  0.0,
                        "role_relevance": 0.0,
                    },
                },
                validity_score=0.0,
                invalid_reasons=invalid_reasons,
                target_skills=target_skills,
            )
            results.append({
                "candidate_id":   cid,
                "final_score":    0.0,
                "role_relevance": 0.0,
                "semantic_match": 0.0,
                "confidence":     0.0,
                "reasoning":      explanation["csv_reasoning"],
                "explainability": explanation,
            })
            continue

        scores = compute_candidate_score(
            candidate=candidate,
            hybrid_similarity=float(hybrid_scores[i]),
            target_skills=target_skills,
            role_relevance=float(role_relevance_scores[i]),
            persona=persona,
            jd_weights=jd_weights,
        )

        # Final score = validity (1.0) × (relevance − risk_penalty)
        final_score = validity_score * scores["raw_relevance_minus_risk"]

        # Tiny tie-breaker adjustment to make candidate scores slightly distinct based on role and semantic fit
        if final_score > 0.0:
            final_score += (scores.get("components", {}).get("role_relevance", 0.0) * 1e-6 +
                            scores.get("components", {}).get("semantic", 0.0) * 1e-8)

        final_score = round(final_score, 4)

        explanation = generate_explainability(
            candidate=candidate,
            scores_dict=scores,
            validity_score=validity_score,
            invalid_reasons=[],
            target_skills=target_skills,
        )

        results.append({
            "candidate_id":   cid,
            "final_score":    final_score,
            "role_relevance": scores.get("components", {}).get("role_relevance", 0.0),
            "semantic_match": scores.get("components", {}).get("semantic", 0.0),
            "confidence":     round(scores["confidence"], 2),
            "reasoning":      explanation["csv_reasoning"],
            "explainability": explanation,
        })

    # ── Step 7: Deterministic Sort ───────────────────────────────────────────
    print("\n[7/7] Sorting results…")
    results.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
    print(f"  Top candidate: {results[0]['candidate_id']} "
          f"(score={results[0]['final_score']:.2f})")

    return results

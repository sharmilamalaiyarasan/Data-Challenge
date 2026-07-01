"""
pipeline.py — Master Orchestrator for the AI Recruiter Ranking System.

Coordinates:
  1. Loading candidates from JSONL
  2. Logical validation (validity_score: 1.0 or 0.0)
  3. Candidate text assembly for BM25 / TF-IDF
  4. Pre-computed or on-demand Sentence Transformer embeddings
  5. Hybrid similarity scoring
  6. Feature extraction + dynamic scoring
  7. Explainability / reasoning generation
  8. Returns sorted list of result dicts ready for CSV output
"""

import os
import json
import pickle
import numpy as np
from tqdm import tqdm

import config
from utils import clean_text
from validator import validate_candidate
from feature_engineering import (
    calculate_career_progression, calculate_founder_mindset,
    calculate_product_dna, get_skill_freshness,
    calculate_notice_period_modifier, calculate_location_modifier,
    calculate_behavioral_multiplier
)
from parser import parse_jd
from embedding_engine import build_or_load_embeddings, assemble_candidate_text
from similarity_engine import get_hybrid_similarities
from scoring_engine import compute_candidate_score
from reasoning_engine import generate_explainability


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_candidates(candidates_path: str) -> list:
    """Reads a .jsonl file and returns a list of candidate dicts."""
    candidates = []
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
# Candidate ID Cache (candidate_id -> candidate_texts mapping)
# ─────────────────────────────────────────────────────────────────────────────

def build_or_load_candidate_texts(candidates: list) -> list:
    """
    Assembles textual representation of every candidate.
    Results are cached to disk so re-runs are instant.
    """
    texts_cache = os.path.join(config.CACHE_DIR, "candidate_texts.pkl")
    if os.path.exists(texts_cache):
        print("  Loading candidate texts from cache...")
        with open(texts_cache, "rb") as f:
            return pickle.load(f)

    print("  Assembling candidate text representations...")
    texts = [assemble_candidate_text(c) for c in tqdm(candidates, desc="  Texts")]
    with open(texts_cache, "wb") as f:
        pickle.dump(texts, f)
    print(f"  Saved candidate texts cache -> {texts_cache}")
    return texts


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    candidates_path: str,
    jd_text: str,
    persona: str = "startup_founder",
    top_n: int = 100
) -> list:
    """
    Full ranking pipeline. Returns a list of result dicts sorted by:
      1. final_score descending
      2. candidate_id ascending (deterministic tie-break)
    """
    print("\n" + "="*60)
    print("  AI RECRUITER — RANKING PIPELINE")
    print("="*60)

    # ── Step 1: Load Candidates ──────────────────────────────────────────────
    print("\n[1/7] Loading candidates...")
    candidates = load_candidates(candidates_path)

    # ── Step 2: Parse JD ────────────────────────────────────────────────────
    print("\n[2/7] Parsing Job Description...")
    jd_config = parse_jd(jd_text)
    target_skills  = jd_config.get("target_skills", {})
    jd_weights     = jd_config.get("weights", None)
    print(f"  Target skills identified: {list(target_skills.keys())[:10]}")

    # ── Step 3: Logical Validation ───────────────────────────────────────────
    print("\n[3/7] Validating candidate profiles...")
    validity_results = {}  # candidate_id -> (score, reasons)
    for c in tqdm(candidates, desc="  Validating"):
        cid = c.get("candidate_id")
        v_score, reasons = validate_candidate(c)
        validity_results[cid] = (v_score, reasons)

    invalid_count = sum(1 for v, _ in validity_results.values() if v == 0.0)
    print(f"  {invalid_count:,} candidates flagged as logically inconsistent (validity=0)")

    # ── Step 4: Assemble Candidate Texts ─────────────────────────────────────
    print("\n[4/7] Assembling candidate text representations...")
    candidate_texts = build_or_load_candidate_texts(candidates)

    # ── Step 5: Embeddings & Hybrid Similarity ───────────────────────────────
    print("\n[5/7] Computing hybrid similarity scores...")
    model, candidate_embeddings = build_or_load_embeddings(candidates, candidate_texts)

    hybrid_scores = get_hybrid_similarities(
        candidates_texts=candidate_texts,
        candidate_embeddings=candidate_embeddings,
        jd_text=jd_text,
        embedding_model=model,
        target_skills=target_skills,
        title_keywords=jd_config.get("title_keywords", [])
    )
    print(f"  Hybrid similarity computed for {len(hybrid_scores):,} candidates.")

    # ── Step 6: Scoring & Feature Extraction ─────────────────────────────────
    print("\n[6/7] Scoring and extracting features...")
    results = []
    for i, candidate in enumerate(tqdm(candidates, desc="  Scoring")):
        cid = candidate.get("candidate_id")
        validity_score, invalid_reasons = validity_results.get(cid, (1.0, []))

        if validity_score == 0.0:
            # Invalid candidates: forced to zero score, skip heavy computation
            explanation = generate_explainability(
                candidate=candidate,
                scores_dict={
                    "relevance": 0.0,
                    "risk": 100.0,
                    "confidence": 0.0,
                    "raw_relevance_minus_risk": 0.0,
                    "matched_skills": [],
                    "components": {
                        "semantic": 0.0,
                        "skills": 0.0,
                        "progression": 0.0,
                        "founder": 0.0,
                        "product": 0.0,
                        "experience": 0.0,
                    }
                },
                validity_score=0.0,
                invalid_reasons=invalid_reasons
            )
            results.append({
                "candidate_id":  cid,
                "final_score":   0.0,
                "confidence":    0.0,
                "reasoning":     explanation["csv_reasoning"],
                "explainability": explanation,
            })
            continue

        # Valid candidate -> full scoring
        scores = compute_candidate_score(
            candidate=candidate,
            hybrid_similarity=float(hybrid_scores[i]),
            target_skills=target_skills,
            persona=persona,
            jd_weights=jd_weights
        )

        # Final score = validity × (relevance − risk_penalty)
        # validity is 1.0 here; included for formula clarity
        final_score = round(validity_score * scores["raw_relevance_minus_risk"], 4)

        explanation = generate_explainability(
            candidate=candidate,
            scores_dict=scores,
            validity_score=validity_score,
            invalid_reasons=[]
        )

        results.append({
            "candidate_id":   cid,
            "final_score":    final_score,
            "confidence":     round(scores["confidence"], 2),
            "reasoning":      explanation["csv_reasoning"],
            "explainability": explanation,
        })

    # ── Step 7: Deterministic Sort ───────────────────────────────────────────
    print("\n[7/7] Sorting results...")
    results.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
    print(f"  Top candidate: {results[0]['candidate_id']} "
          f"(score={results[0]['final_score']:.2f})")

    return results

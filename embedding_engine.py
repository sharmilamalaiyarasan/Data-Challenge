"""
embedding_engine.py — Sentence Transformer embedding management.

Handles model loading (local cache or HuggingFace download) and
candidate text assembly / embedding computation with disk caching.
"""

# Standard library
from datetime import datetime
import os

# Third-party
import numpy as np
from sentence_transformers import SentenceTransformer

# Local
import config
from utils import parse_date


def get_embedding_model():

    """Loads the embedding model from the local cache, or downloads it on first run."""
    model_name = "all-MiniLM-L6-v2"
    model_path = os.path.join(config.BASE_DIR, "models", model_name)

    if os.path.exists(model_path):
        return SentenceTransformer(model_path)

    print(f"Embedding model not found at {model_path}. Downloading from HuggingFace…")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model = SentenceTransformer(model_name)
    model.save(model_path)
    return model


def assemble_candidate_text(cand: dict) -> str:
    """Concatenates headline, summary, skills, and recent job descriptions into a single string."""
    profile  = cand.get("profile", {})
    headline = profile.get("headline", "")
    summary  = profile.get("summary", "")

    skill_names = " ".join(s.get("name", "") for s in cand.get("skills", []))

    job_descs = [
        f"{job.get('title', '')}: {job.get('description', '')}"
        for job in cand.get("career_history", [])[:3]   # most recent 3 jobs
    ]

    return f"{headline}. {summary}. {skill_names}. {' '.join(job_descs)}"


def build_candidate_texts(candidates: list) -> list:
    """Returns text representation list for all candidates (kept for backwards compatibility)."""
    return [assemble_candidate_text(c) for c in candidates]


def get_or_build_embeddings(candidates: list) -> np.ndarray:
    """
    Returns pre-computed candidate embeddings, loading from disk cache when available.

    Falls back to computing and caching them if the cache file is missing or corrupt.
    """
    cache_path = config.EMBEDDING_CACHE_FILE

    if os.path.exists(cache_path):
        print(f"Loading candidate embeddings from cache: {cache_path}")
        try:
            return np.load(cache_path).astype(np.float32)
        except Exception as e:
            print(f"Embedding cache load error: {e}. Rebuilding…")

    print("Rebuilding candidate embeddings… (first run may take several minutes on CPU)")
    model = get_embedding_model()
    texts = build_candidate_texts(candidates)

    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embeddings.astype(np.float32))
    print(f"Cached {len(embeddings):,} embeddings → {cache_path}")
    return embeddings


def build_or_load_embeddings(candidates: list, candidate_texts: list = None):
    """
    Public interface used by pipeline.py.

    Returns:
        (model, embeddings_array): Tuple of the loaded ST model and the float32 embeddings.
    """
    model      = get_embedding_model()
    cache_path = config.EMBEDDING_CACHE_FILE

    if os.path.exists(cache_path):
        print(f"  Loading embeddings from cache: {cache_path}")
        try:
            embeddings = np.load(cache_path).astype(np.float32)
            print(f"  Loaded {len(embeddings):,} cached embeddings.")
            return model, embeddings
        except Exception as e:
            print(f"  Cache load error: {e}. Rebuilding…")

    if candidate_texts is None:
        candidate_texts = build_candidate_texts(candidates)

    print("  Building Sentence Transformer embeddings (first run — will be cached)…")
    embeddings = model.encode(
        candidate_texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embeddings)
    print(f"  Saved {len(embeddings):,} embeddings → {cache_path}")
    return model, embeddings


def assemble_candidate_roles(cand: dict) -> str:
    """
    Concatenates the unique titles of a candidate's recent jobs (up to 3) in reverse chronological order
    to construct a trajectory representation.
    """
    history = cand.get("career_history", []) or []

    # Sort career history by start date (newest first)
    sorted_history = sorted(
        [job for job in history if job.get("start_date")],
        key=lambda x: parse_date(x.get("start_date")) or datetime.min,
        reverse=True,
    )
    if not sorted_history:
        sorted_history = history

    titles = [job.get("title", "") for job in sorted_history if job.get("title")]
    
    unique_titles = []
    for t in titles:
        t_clean = " ".join(t.split())
        if t_clean and t_clean not in unique_titles:
            unique_titles.append(t_clean)
            if len(unique_titles) >= 3:
                break

    if not unique_titles:
        profile_title = cand.get("profile", {}).get("current_title", "")
        if profile_title:
            return profile_title
        return "Engineer"

    return " / ".join(unique_titles)


def build_or_load_role_embeddings(candidates: list, model=None):
    """
    Loads pre-computed candidate role embeddings, or computes and caches them.
    Returns:
        role_embeddings (np.ndarray): Matrix of embedded candidate titles/trajectory.
    """
    cache_path = config.ROLE_RELEVANCE_CACHE_FILE

    if os.path.exists(cache_path):
        print(f"  Loading role embeddings from cache: {cache_path}")
        try:
            embeddings = np.load(cache_path).astype(np.float32)
            if len(embeddings) == len(candidates):
                print(f"  Loaded {len(embeddings):,} cached role embeddings.")
                return embeddings
            else:
                print(f"  Role embedding cache size mismatch ({len(embeddings)} vs {len(candidates)}). Rebuilding…")
        except Exception as e:
            print(f"  Role embedding cache load error: {e}. Rebuilding…")

    print("  Assembling candidate role representations…")
    role_texts = [assemble_candidate_roles(c) for c in candidates]

    print("  Building Sentence Transformer role embeddings (first run — will be cached)…")
    if model is None:
        model = get_embedding_model()
    
    embeddings = model.encode(
        role_texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embeddings)
    print(f"  Saved {len(embeddings):,} role embeddings → {cache_path}")
    return embeddings



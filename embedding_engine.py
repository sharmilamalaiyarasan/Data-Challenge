import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import config

def get_embedding_model():
    """Loads model from local cache directory, or downloads if missing."""
    model_name = "all-MiniLM-L6-v2"
    models_dir = os.path.join(config.BASE_DIR, "models")
    model_path = os.path.join(models_dir, model_name)
    
    if os.path.exists(model_path):
        # Load local model
        return SentenceTransformer(model_path)
    else:
        # Download and cache locally (requires network, only during dev/prep)
        print(f"Embedding model not found locally at {model_path}. Downloading from HuggingFace...")
        os.makedirs(models_dir, exist_ok=True)
        model = SentenceTransformer(model_name)
        model.save(model_path)
        return model

def assemble_candidate_text(cand: dict) -> str:
    """Concatenates profile text features for a single candidate."""
    profile = cand.get("profile", {})
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")

    # Collect skills text
    skills = cand.get("skills", [])
    skill_names = " ".join(s.get("name", "") for s in skills)

    # Pull text from current/recent job descriptions
    history = cand.get("career_history", [])
    job_descs = []
    for job in history[:3]:  # take recent 3 jobs
        title = job.get("title", "")
        desc = job.get("description", "")
        job_descs.append(f"{title}: {desc}")

    full_text = f"{headline}. {summary}. {skill_names}. {' '.join(job_descs)}"
    return full_text


def build_candidate_texts(candidates: list) -> list:
    """Builds text representation list for all candidates (kept for backwards compat)."""
    return [assemble_candidate_text(c) for c in candidates]

def get_or_build_embeddings(candidates: list) -> np.ndarray:
    """
    Checks if pre-computed embeddings exist in cache directory.
    If yes, loads them. If no, computes and saves them.
    Returns only the embeddings array (model is created internally).
    """
    cache_path = config.EMBEDDING_CACHE_FILE

    if os.path.exists(cache_path):
        print(f"Loading candidate embeddings from cache: {cache_path}")
        try:
            return np.load(cache_path).astype(np.float32)
        except Exception as e:
            print(f"Error loading embedding cache: {e}. Rebuilding...")

    # Fallback: rebuild from scratch
    print("Rebuilding candidate embeddings... This will take a few minutes on CPU.")
    model = get_embedding_model()
    texts = build_candidate_texts(candidates)

    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embeddings.astype(np.float32))
    print(f"Successfully cached {len(embeddings)} candidate embeddings -> {cache_path}")
    return embeddings


def build_or_load_embeddings(candidates: list, candidate_texts: list = None):
    """
    Public interface used by pipeline.py.
    Returns (model, embeddings_array) tuple.
    """
    model = get_embedding_model()
    cache_path = config.EMBEDDING_CACHE_FILE

    if os.path.exists(cache_path):
        print(f"  Loading embeddings from cache: {cache_path}")
        try:
            embeddings = np.load(cache_path).astype(np.float32)
            print(f"  Loaded {len(embeddings):,} cached embeddings.")
            return model, embeddings
        except Exception as e:
            print(f"  Cache load error: {e}. Rebuilding...")

    # Build embeddings from candidate_texts (or assemble if not provided)
    if candidate_texts is None:
        candidate_texts = build_candidate_texts(candidates)

    print("  Building Sentence Transformer embeddings (first-run, will be cached)...")
    embeddings = model.encode(
        candidate_texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype(np.float32)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embeddings)
    print(f"  Saved {len(embeddings):,} embeddings -> {cache_path}")
    return model, embeddings

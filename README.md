# 🤖 AI Recruiter — Hybrid Semantic Ranking System

> **Redrob × India Runs — Data & AI Challenge**  
> Ranking 100,000+ candidates against a Senior AI Engineer job description using a fully offline, deterministic, explainable AI pipeline.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Architecture](#architecture)
4. [Scoring Methodology](#scoring-methodology)
5. [Honeypot / Fraud Detection](#honeypot--fraud-detection)
6. [Performance Benchmarks](#performance-benchmarks)
7. [Folder Structure](#folder-structure)
8. [Installation](#installation)
9. [How to Run `rank.py`](#how-to-run-rankpy)
10. [How to Launch the Streamlit App](#how-to-launch-the-streamlit-app)
11. [Team Information](#team-information)
12. [License](#license)

---

## Project Overview

This system ranks a pool of 100,000+ candidate profiles against a job description for a **Senior AI Engineer** role. It combines three complementary NLP similarity signals — Sentence Transformers, BM25 Okapi, and TF-IDF — into a single hybrid score, then applies a multi-dimensional feature-scoring layer and a logical consistency validator to produce a deterministic, explainable Top-100 shortlist.

**Key design goals:**
- ✅ No external API calls during ranking (fully offline)
- ✅ Deterministic: byte-identical output across repeated runs
- ✅ Runtime < 60 seconds on CPU (after first-run embedding cache)
- ✅ Honeypot-aware: invalid profiles are scored `0.0` and filtered out
- ✅ Explainable: every ranked candidate has a structured XAI breakdown

---

## Problem Statement

Given:
- `candidates.jsonl` — 100,000+ candidate profiles in JSONL format
- `job_description.md` — a detailed Senior AI Engineer job description

Produce:
- `submission.csv` — ranked Top 100 candidates with columns: `candidate_id`, `rank`, `score`, `reasoning`

Constraints: no GPU, no network during ranking, ≤ 5 minutes end-to-end on CPU with 16 GB RAM.

---

## Architecture

```
candidates.jsonl
       │
       ▼
┌─────────────────────┐
│   1. Data Loading   │  load_candidates() — streams JSONL line-by-line
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  2. JD Parsing      │  parse_jd() — extracts target_skills + title keywords
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────────────────────┐
│  3. Logical Consistency Validator               │
│     • Future dates in career history            │
│     • Expert skills with 0 months experience   │
│     • Duration inflation vs. stated yrs_exp     │
│     • Overlapping full-time employment          │
│  → validity_score = 1.0 (valid) | 0.0 (reject) │
└─────────┬───────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────┐
│  4. Hybrid Similarity Engine (3-signal fusion)               │
│                                                              │
│   ┌─────────────────────┐  weight  ┌──────────────────────┐ │
│   │ Sentence Transformers│  50%  ──▶│                      │ │
│   │ (all-MiniLM-L6-v2)  │          │  Hybrid Similarity   │ │
│   ├─────────────────────┤          │  Score (0–1)         │ │
│   │ BM25 Okapi          │  30%  ──▶│                      │ │
│   ├─────────────────────┤          │                      │ │
│   │ TF-IDF Cosine       │  20%  ──▶│                      │ │
│   └─────────────────────┘          └──────────────────────┘ │
└─────────┬────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────┐
│  5. Feature Scoring Engine (5 components)                    │
│                                                              │
│   Component          Weight (startup_founder persona)        │
│   ─────────────────  ──────────────────────────────         │
│   semantic_match     25%                                     │
│   skills             25%  (proficiency × duration weighted)  │
│   career_growth      15%  (title rank trajectory)            │
│   founder_mindset    20%  (GitHub + startup DNA signals)     │
│   product_experience 15%  (product vs service ratio)         │
│                                                              │
│   × behavioral_multiplier (activity, open-to-work, etc.)    │
└─────────┬────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────┐
│  6. Risk Scoring (subtracted as penalty)                     │
│                                                              │
│   • Notice period > 90 days          → up to −30 pts        │
│   • Job-hopping (avg tenure < 18 mo) → up to −25 pts        │
│   • Relocation friction              → up to −20 pts        │
│   • Stale skills                     → up to −15 pts        │
│   • Incomplete profile               → up to −10 pts        │
│                                                              │
│   final_score = validity × max(0, relevance − risk × 0.15)  │
└─────────┬────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────┐
│  7. Explainability / Reasoning Engine                        │
│     • 1–5 ⭐ star ratings across 6 dimensions               │
│     • Factual one-sentence CSV reasoning string             │
│     • Structured JSON XAI breakdown per candidate           │
└─────────┬────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────┐
│  8. Deterministic   │  sort(-score, candidate_id ASC)
│     Sort & Output   │  → submission.csv (Top 100)
└─────────────────────┘
```

---

## Scoring Methodology

### Final Score Formula

```
final_score = validity_score × max(0, relevance_score − risk_penalty)

where:
  validity_score  = 1.0 (passes all consistency checks) | 0.0 (honeypot/invalid)
  relevance_score = weighted_sum(components) × behavioral_multiplier
  risk_penalty    = risk_score × 0.15
```

### Component Weights — `startup_founder` Persona

| Component | Weight | Signal |
|---|---|---|
| `semantic_match` | 25% | Sentence Transformers + BM25 + TF-IDF fusion |
| `skills` | 25% | Proficiency level (0.5–1.5×) × duration (0.8–1.2×) × JD weight |
| `career_growth` | 15% | Title seniority trajectory over career history |
| `founder_mindset` | 20% | GitHub presence, startup company detection, builder signals |
| `product_experience` | 15% | Ratio of product vs service-company roles |

### Experience Score

| Years | Score |
|---|---|
| 0 | 10 |
| 5–10 (sweet spot) | 100 |
| 15+ | ~50 (overqualification decay) |

### Risk Penalties

| Factor | Max Penalty |
|---|---|
| Notice period > 90 days | 30 pts |
| Job-hopping (avg tenure < 18 mo) | 25 pts |
| Relocation friction | 20 pts |
| Stale skills | 15 pts |
| Incomplete profile | 10 pts |

---

## Honeypot / Fraud Detection

The `validator.py` module applies **logical consistency checks** before scoring:

| Check | Condition | Action |
|---|---|---|
| Current job duration mismatch | Stated vs calculated duration differs by > 3 months | Flag |
| Skill duration fraud | `expert` proficiency with `0` months on ≥ 3 skills | Flag |
| Experience timeline inflation | Sum of job durations > stated years + 3 years | Flag |
| Future date anomalies | Start/end dates after benchmark date (2026-06-22) | Flag |
| Education timeline contradiction | `start_year > end_year` | Flag |

Any flagged profile receives `validity_score = 0.0` → `final_score = 0.0`.  
This ensures honeypots **cannot** enter the Top 100 regardless of keyword overlap.

---

## Performance Benchmarks

| Run | Description | Time |
|---|---|---|
| First run | Builds embedding cache (~100K profiles) | ~3–4 min |
| Subsequent runs | Loads from `cache/candidate_embeddings.npy` | **~48 seconds** |
| BM25 query | Optimized to top-40 keyword terms | ~1 second |
| Validation | 100K profiles logical checks | ~8 seconds |
| End-to-end (cached) | Full pipeline | **< 60 seconds** |

Tested on: Windows 11, 8-core CPU, 16 GB RAM, Python 3.11, no GPU.

---

## Folder Structure

```
.
├── rank.py                    # CLI entry point — produces submission.csv
├── app.py                     # Streamlit UI dashboard
├── pipeline.py                # Master pipeline orchestrator
├── scoring_engine.py          # Relevance, risk & confidence scoring
├── similarity_engine.py       # BM25 + TF-IDF + Sentence Transformers fusion
├── embedding_engine.py        # Embedding builder & cache manager
├── feature_engineering.py     # Career progression, founder mindset, etc.
├── reasoning_engine.py        # XAI explainability & CSV reasoning generator
├── validator.py               # Logical consistency / honeypot detector
├── parser.py                  # Job description parser
├── config.py                  # Paths, seeds, persona weight configs
├── utils.py                   # Date parsing & text cleaning helpers
│
├── job_description.md         # Target job description (Senior AI Engineer)
├── candidate_schema.json      # Schema definition for candidates.jsonl
├── submission_metadata.yaml   # Hackathon submission metadata
├── validate_submission.py     # Official submission validator
│
├── cache/                     # Auto-generated on first run (gitignored)
│   ├── candidate_embeddings.npy
│   ├── candidate_texts.pkl
│   └── bm25_index.pkl
│
├── models/                    # Local Sentence Transformer model (gitignored)
│   └── all-MiniLM-L6-v2/
│
├── output/                    # Intermediate outputs (gitignored)
└── submission.csv             # Final ranked output (100 rows)
```

---

## Installation

**Prerequisites:** Python 3.9+

```bash
# 1. Clone the repository
git clone https://github.com/sharmilamalaiyarasan/Data-Challenge.git
cd Data-Challenge

# 2. (Recommended) Create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** The `sentence-transformers` model (`all-MiniLM-L6-v2`) is downloaded automatically on the first run and cached under `models/`. Subsequent runs are fully offline.

---

## How to Run `rank.py`

```bash
# Standard submission command (produces submission.csv)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# With all optional flags
python rank.py \
  --candidates ./candidates.jsonl \
  --out        ./submission.csv \
  --jd         ./job_description.md \
  --persona    startup_founder \
  --top        100
```

**Available flags:**

| Flag | Default | Description |
|---|---|---|
| `--candidates` | *(required)* | Path to `candidates.jsonl` |
| `--out` | `./submission.csv` | Output CSV file path |
| `--jd` | `./job_description.md` | Path to job description file |
| `--persona` | `startup_founder` | `startup_founder` or `enterprise_recruiter` |
| `--top` | `100` | Number of top candidates to output |

**Validate the output:**
```bash
python validate_submission.py submission.csv
```

---

## How to Launch the Streamlit App

```bash
streamlit run app.py
```

The dashboard provides:
- 🔍 **Candidate Search** — search by ID or name across the full dataset
- 📊 **Score Breakdown** — interactive 5-dimension radar chart
- ⭐ **Star Ratings** — Skills, Experience, Career, Culture Fit, Availability
- 📈 **Feature Contribution** — ASCII bar charts showing per-component scores
- 🏆 **Top-100 Leaderboard** — ranked table with confidence scores and reasoning
- ⚠️ **Honeypot Report** — flagged profiles and detected inconsistencies

---

## Team Information

| Field | Value |
|---|---|
| GitHub | [github.com/sharmilamalaiyarasan](https://github.com/sharmilamalaiyarasan) |
| Repository | [Data-Challenge](https://github.com/sharmilamalaiyarasan/Data-Challenge) |
| Team Name | TBD — update in `submission_metadata.yaml` before submission |
| Contact | TBD — update in `submission_metadata.yaml` before submission |

---

## License

This project is submitted as part of the **Redrob × India Runs — Data & AI Challenge** hackathon.  
All code is original work by the team. See `submission_metadata.yaml` for full declarations.

---

*Built with ❤️ using Python, Sentence Transformers, BM25, TF-IDF, and Streamlit.*

"""
rank.py — CLI entry point for the AI Recruiter Ranking System.

Usage (submission format):
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Optional flags:
    --jd       path to job description file (default: ./job_description.md)
    --persona  startup_founder | enterprise_recruiter (default: startup_founder)
    --top      number of candidates to output (default: 100)
"""

import argparse
import csv
import os
import sys
import time

# Force UTF-8 stdout/stderr on Windows to avoid cp1252 encoding errors
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import config
from pipeline import run_pipeline


def read_jd(jd_path: str) -> str:
    """Reads the job description from a text or markdown file."""
    if not os.path.exists(jd_path):
        print(f"[WARN] JD file not found at {jd_path}. Using empty JD.")
        return ""
    with open(jd_path, "r", encoding="utf-8") as f:
        return f.read()


def write_csv(results: list, out_path: str) -> None:
    """
    Writes ranked results to CSV with the exact required columns:
      candidate_id, rank, score, reasoning
    (as specified by validate_submission.py REQUIRED_HEADER)
    """
    fieldnames = ["candidate_id", "rank", "score", "reasoning"]
    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for rank_num, row in enumerate(results, start=1):
            writer.writerow({
                "candidate_id": row["candidate_id"],
                "rank":         rank_num,
                "score":        f"{row['final_score']:.4f}",
                "reasoning":    row["reasoning"],
            })
    print(f"\n  Submission CSV written -> {out_path}")
    print(f"  Total rows: {len(results)}")



def main():
    parser = argparse.ArgumentParser(
        description="AI Recruiter — Rank candidates against a Job Description."
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to candidates.jsonl file."
    )
    parser.add_argument(
        "--out",
        default=config.SUBMISSION_FILE,
        help="Path for output submission.csv (default: ./submission.csv)."
    )
    parser.add_argument(
        "--jd",
        default=os.path.join(config.BASE_DIR, "job_description.md"),
        help="Path to the job description text/markdown file."
    )
    parser.add_argument(
        "--persona",
        default="startup_founder",
        choices=list(config.PERSONAS.keys()),
        help="Recruiter persona to weight scores."
    )
    parser.add_argument(
        "--top",
        type=int,
        default=100,
        help="Number of top candidates to output."
    )

    args = parser.parse_args()

    # ── Validate inputs ──────────────────────────────────────────────────────
    if not os.path.exists(args.candidates):
        print(f"[ERROR] candidates file not found: {args.candidates}")
        sys.exit(1)

    # ── Load JD ──────────────────────────────────────────────────────────────
    jd_text = read_jd(args.jd)
    if jd_text:
        print(f"  JD loaded from: {args.jd} ({len(jd_text)} chars)")
    else:
        print("  [WARN] JD is empty — semantic match will be weak.")

    # ── Run Pipeline ─────────────────────────────────────────────────────────
    start_time = time.time()

    ranked_results = run_pipeline(
        candidates_path=args.candidates,
        jd_text=jd_text,
        persona=args.persona,
        top_n=args.top
    )

    elapsed = time.time() - start_time
    print(f"\n  Pipeline completed in {elapsed:.1f}s")

    # Trim to top_n
    top_results = ranked_results[: args.top]

    # ── Write CSV ─────────────────────────────────────────────────────────────
    write_csv(top_results, args.out)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  TOP 5 CANDIDATES")
    print("="*60)
    for i, r in enumerate(top_results[:5], start=1):
        print(f"  #{i}  {r['candidate_id']}  score={r['final_score']:.2f}  "
              f"conf={r['confidence']:.0f}%  | {r['reasoning'][:80]}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

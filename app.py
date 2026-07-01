"""
app.py — Streamlit Sandbox & XAI Dashboard for the AI Recruiter system.

Run with:
    streamlit run app.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st

import config
from pipeline import run_pipeline, load_candidates


# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Recruiter — Candidate Discovery",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

.score-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    border: 1px solid rgba(139,92,246,0.35);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 4px 24px rgba(139,92,246,0.15);
    transition: transform 0.2s, box-shadow 0.2s;
}
.score-card:hover { transform: translateY(-2px); box-shadow: 0 8px 32px rgba(139,92,246,0.3); }

.rank-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 36px; height: 36px; border-radius: 50%;
    font-weight: 700; font-size: 14px; margin-right: 12px;
}
.rank-1 { background: linear-gradient(135deg, #f59e0b, #d97706); color: #1c1917; }
.rank-2 { background: linear-gradient(135deg, #94a3b8, #64748b); color: #0f172a; }
.rank-3 { background: linear-gradient(135deg, #cd7c2e, #92400e); color: #fef3c7; }
.rank-n { background: rgba(139,92,246,0.25); color: #c4b5fd; }

.stars { color: #f59e0b; font-size: 18px; letter-spacing: 2px; }

.pill { display:inline-block; padding:3px 12px; border-radius:999px; font-size:12px; font-weight:600; }
.pill-green { background:rgba(16,185,129,0.2);color:#6ee7b7;border:1px solid rgba(16,185,129,0.4); }
.pill-yellow { background:rgba(245,158,11,0.2);color:#fcd34d;border:1px solid rgba(245,158,11,0.4); }
.pill-red { background:rgba(239,68,68,0.2);color:#fca5a5;border:1px solid rgba(239,68,68,0.4); }

.section-header { font-size:11px; font-weight:600; letter-spacing:1.5px; text-transform:uppercase; color:#7c3aed; margin-bottom:4px; }

.metric-row { display:flex; gap:12px; margin-bottom:12px; }
.metric-box { flex:1; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:12px; text-align:center; }
.metric-val { font-size:22px; font-weight:700; color:#a78bfa; }
.metric-lbl { font-size:11px; color:#94a3b8; margin-top:2px; }

.risk-low  { color: #6ee7b7; font-weight: 600; }
.risk-med  { color: #fcd34d; font-weight: 600; }
.risk-high { color: #fca5a5; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def stars(n: int) -> str:
    n = max(1, min(5, int(n)))
    return "★" * n + "☆" * (5 - n)

def risk_html(label: str) -> str:
    css = {"Low": "risk-low", "Medium": "risk-med", "High": "risk-high"}.get(label, "risk-low")
    return f'<span class="{css}">{label}</span>'

def rec_pill(rec: str) -> str:
    if "Highly" in rec:
        return f'<span class="pill pill-green">{rec}</span>'
    elif "Recommended" in rec and "Not" not in rec:
        return f'<span class="pill pill-yellow">{rec}</span>'
    else:
        return f'<span class="pill pill-red">{rec}</span>'

def rank_badge(rank: int) -> str:
    cls = {1: "rank-1", 2: "rank-2", 3: "rank-3"}.get(rank, "rank-n")
    return f'<span class="rank-badge {cls}">#{rank}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Configuration
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🤖 AI Recruiter")
    st.markdown("*Intelligent Candidate Discovery*")
    st.markdown("---")
    st.markdown("### ⚙️ Configuration")

    candidates_path = st.text_input("Candidates file (.jsonl)", value=config.DATA_FILE)
    jd_path = st.text_input("Job Description file",
                             value=os.path.join(config.BASE_DIR, "job_description.md"))
    persona = st.selectbox(
        "Recruiter Persona",
        options=list(config.PERSONAS.keys()),
        format_func=lambda k: {
            "startup_founder": "🚀 Startup Founder",
            "enterprise_recruiter": "🏢 Enterprise Recruiter"
        }.get(k, k)
    )
    top_n = st.slider("Top N candidates to rank", 10, 200, 100, step=10)

    st.markdown("---")
    st.markdown("### 🎛️ Score Weights")
    st.caption("Override the persona defaults")

    w_semantic = st.slider("Semantic Match",     0.0, 1.0, 0.25, 0.05)
    w_skills   = st.slider("Skills",             0.0, 1.0, 0.25, 0.05)
    w_growth   = st.slider("Career Growth",      0.0, 1.0, 0.15, 0.05)
    w_founder  = st.slider("Founder Mindset",    0.0, 1.0, 0.20, 0.05)
    w_product  = st.slider("Product Experience", 0.0, 1.0, 0.15, 0.05)

    custom_weights = {
        "semantic_match":    w_semantic,
        "skills":            w_skills,
        "career_growth":     w_growth,
        "founder_mindset":   w_founder,
        "product_experience": w_product,
    }

    run_btn = st.button("🚀 Run Ranking", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("### 🔍 Filter")
    search_query = st.text_input("Filter by candidate ID / reasoning")
    min_score    = st.slider("Minimum score", 0.0, 100.0, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Main Area
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="padding: 8px 0 24px 0;">
  <h1 style="margin:0;font-size:2rem;font-weight:700;
             background:linear-gradient(135deg, #a78bfa, #60a5fa);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    🤖 AI Recruiter Dashboard
  </h1>
  <p style="color:#94a3b8;margin-top:6px;font-size:14px;">
    Rank candidates by true job fit — not keyword count
  </p>
</div>
""", unsafe_allow_html=True)

if "results"  not in st.session_state: st.session_state.results  = None
if "run_time" not in st.session_state: st.session_state.run_time = None

# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    if not os.path.exists(candidates_path):
        st.error(f"Candidates file not found: `{candidates_path}`")
    else:
        with st.spinner("Running ranking pipeline… (first run builds embedding cache ~3 min)"):
            jd_text = ""
            if os.path.exists(jd_path):
                with open(jd_path, "r", encoding="utf-8") as f:
                    jd_text = f.read()

            config.PERSONAS[persona]["weights"] = custom_weights

            t0 = time.time()
            results = run_pipeline(
                candidates_path=candidates_path,
                jd_text=jd_text,
                persona=persona,
                top_n=top_n
            )
            elapsed = time.time() - t0

        st.session_state.results  = results
        st.session_state.run_time = elapsed
        st.success(
            f"✅ Ranking complete in **{elapsed:.1f}s** — "
            f"{len(results):,} candidates processed."
        )


# ── Display Results ───────────────────────────────────────────────────────────
if st.session_state.results:
    results = st.session_state.results

    filtered = [
        r for r in results
        if r["final_score"] >= min_score
        and (
            not search_query
            or search_query.lower() in r["candidate_id"].lower()
            or search_query.lower() in r["reasoning"].lower()
        )
    ]

    tab_cards, tab_table, tab_analytics = st.tabs([
        "🃏 Candidate Cards", "📊 Data Table", "📈 Analytics"
    ])

    # ─── TAB 1: Candidate Cards ───────────────────────────────────────────────
    with tab_cards:
        st.markdown(f"**Showing {len(filtered)} candidates** (score ≥ {min_score:.0f})")

        for rank, row in enumerate(filtered, start=1):
            exp       = row.get("explainability", {}) or {}
            bd        = exp.get("breakdown", {}) or {}
            rec       = exp.get("recommendation", "—")
            strengths  = exp.get("strengths", []) or []
            weaknesses = exp.get("weaknesses", []) or []
            contribs   = exp.get("feature_contributions", []) or []

            risk_lbl   = bd.get("risk", "Low")
            badge_html = rank_badge(rank)
            pill_html  = rec_pill(rec)
            risk_h     = risk_html(risk_lbl)

            with st.container():
                st.markdown(f"""
<div class="score-card">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
    <div style="display:flex;align-items:center;">
      {badge_html}
      <div>
        <div style="font-weight:700;font-size:16px;color:#e2e8f0;">{row['candidate_id']}</div>
        <div style="font-size:12px;color:#94a3b8;margin-top:2px;">
          {row['reasoning'][:120]}{"…" if len(row['reasoning']) > 120 else ""}
        </div>
      </div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:28px;font-weight:800;color:#a78bfa;">{row['final_score']:.1f}</div>
      <div style="font-size:11px;color:#94a3b8;">/ 100</div>
    </div>
  </div>

  <div class="metric-row">
    <div class="metric-box"><div class="metric-val">{row['confidence']:.0f}%</div><div class="metric-lbl">Confidence</div></div>
    <div class="metric-box"><div class="metric-val" style="color:#f59e0b;">{row['final_score']:.1f}</div><div class="metric-lbl">Final Score</div></div>
    <div class="metric-box"><div class="metric-val">{risk_h}</div><div class="metric-lbl">Risk Level</div></div>
    <div class="metric-box"><div>{pill_html}</div><div class="metric-lbl">Verdict</div></div>
  </div>

  <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:4px;">
    <div><div class="section-header">Skills</div><div class="stars">{stars(bd.get('skills',3))}</div></div>
    <div><div class="section-header">Experience</div><div class="stars">{stars(bd.get('experience',3))}</div></div>
    <div><div class="section-header">Career</div><div class="stars">{stars(bd.get('career',3))}</div></div>
    <div><div class="section-header">Culture</div><div class="stars">{stars(bd.get('culture',3))}</div></div>
    <div><div class="section-header">Availability</div><div class="stars">{stars(bd.get('availability',3))}</div></div>
  </div>

  {"<div style='margin-top:10px;font-size:12px;'><span style='color:#6ee7b7;'>&#10004; " + " &middot; ".join(strengths[:2]) + "</span></div>" if strengths else ""}
  {"<div style='font-size:12px;'><span style='color:#fca5a5;'>&#9888; " + " &middot; ".join(weaknesses[:2]) + "</span></div>" if weaknesses else ""}
</div>
""", unsafe_allow_html=True)

                # ── Feature Contribution Bar Chart ─────────────────────────────
                if contribs:
                    with st.expander("📊 Score Breakdown — Why this rank?",
                                     expanded=(rank == 1)):
                        st.markdown(
                            f"**Overall Score: {row['final_score']:.1f}** &nbsp;|&nbsp; "
                            f"**Confidence: {row['confidence']:.0f}%**"
                        )

                        # Table-style bar chart (always works, no JS needed)
                        max_c = max((abs(c.get("contribution", 0)) for c in contribs), default=1)
                        rows_html = ""
                        for item in contribs:
                            lbl  = item.get("label", "")
                            val  = item.get("value", 0.0)
                            cont = item.get("contribution", 0.0)
                            bar  = int(abs(cont) / max(max_c, 0.01) * 25)
                            col  = "#6ee7b7" if cont >= 0 else "#fca5a5"
                            sign = "+" if cont >= 0 else ""
                            rows_html += (
                                f"<tr>"
                                f"<td style='padding:3px 10px 3px 0;color:#cbd5e1;"
                                f"font-size:13px;white-space:nowrap;min-width:160px'>{lbl}</td>"
                                f"<td style='font-family:monospace;color:{col};"
                                f"font-size:13px;letter-spacing:-0.5px;min-width:180px'>"
                                f"{'&#9608;' * bar}{'&#9617;' * (25-bar)}</td>"
                                f"<td style='padding:3px 0 3px 10px;color:{col};"
                                f"font-weight:700;font-size:13px'>{sign}{cont:.1f}</td>"
                                f"</tr>"
                            )
                        st.markdown(
                            f"<table style='border-collapse:collapse;margin-top:8px'>"
                            f"{rows_html}</table>",
                            unsafe_allow_html=True
                        )

            if rank >= 30:
                st.caption(
                    f"… and {len(filtered) - 30} more candidates. "
                    f"Adjust the minimum score filter to see fewer."
                )
                break

    # ─── TAB 2: Data Table ────────────────────────────────────────────────────
    with tab_table:
        df = pd.DataFrame([
            {
                "Rank":         i + 1,
                "Candidate ID": r["candidate_id"],
                "Score":        round(r["final_score"], 2),
                "Confidence":   round(r["confidence"], 1),
                "Risk":         (r.get("explainability") or {}).get("breakdown", {}).get("risk", "—"),
                "Verdict":      (r.get("explainability") or {}).get("recommendation", "—"),
                "Reasoning":    r["reasoning"],
            }
            for i, r in enumerate(filtered)
        ])
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score":      st.column_config.ProgressColumn("Score",      min_value=0, max_value=100),
                "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100),
            }
        )
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv_bytes,
                           file_name="ai_recruiter_results.csv", mime="text/csv")

    # ─── TAB 3: Analytics ─────────────────────────────────────────────────────
    with tab_analytics:
        all_scores = [r["final_score"] for r in filtered]
        risks = [(r.get("explainability") or {}).get("breakdown", {}).get("risk", "Low")
                 for r in filtered]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Top Score",     f"{max(all_scores):.1f}"       if all_scores else "—")
        col2.metric("Avg Score",     f"{np.mean(all_scores):.1f}"   if all_scores else "—")
        col3.metric("High Risk",     risks.count("High"))
        col4.metric("Low Risk",      risks.count("Low"))

        st.markdown("#### Score Distribution")
        hist_df = pd.DataFrame({"Score": all_scores})
        st.bar_chart(hist_df["Score"].value_counts(bins=20).sort_index())

        st.markdown("#### Risk Breakdown")
        st.bar_chart(pd.DataFrame({
            "Count": [risks.count("Low"), risks.count("Medium"), risks.count("High")]
        }, index=["Low", "Medium", "High"]))

        st.markdown("#### Recommendation Breakdown")
        verdicts = pd.Series([
            (r.get("explainability") or {}).get("recommendation", "Unknown")
            for r in filtered
        ]).value_counts()
        st.bar_chart(verdicts)

        # Avg feature contribution across top candidates
        st.markdown("#### Avg Feature Contributions (top 20)")
        all_contribs: dict[str, list] = {}
        for r in filtered[:20]:
            for item in (r.get("explainability") or {}).get("feature_contributions", []):
                lbl = item.get("label", "")
                all_contribs.setdefault(lbl, []).append(item.get("contribution", 0.0))
        if all_contribs:
            avg_df = pd.DataFrame({
                "Avg Contribution": {k: round(sum(v)/len(v), 1)
                                     for k, v in all_contribs.items()}
            })
            st.bar_chart(avg_df)

else:
    st.markdown("""
<div style="text-align:center;padding:80px 0;color:#64748b;">
  <div style="font-size:64px;margin-bottom:16px;">🎯</div>
  <h3 style="color:#94a3b8;font-weight:600;">Ready to Rank Candidates</h3>
  <p style="max-width:420px;margin:0 auto;line-height:1.6;">
    Configure parameters in the sidebar and click <strong>Run Ranking</strong>
    to discover best-fit candidates using hybrid AI scoring.
  </p>
</div>
""", unsafe_allow_html=True)

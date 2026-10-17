"""
app.py — CEM-RAG IDS Streamlit Demo

Run:  streamlit run app.py
Requires: inference.py, scenario_generator.py, netslab_episode_kb_v3/ artifacts
"""

import time
import json
import io
import os
import threading
from collections import deque

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="CEM-RAG IDS Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background-color: #0f1117; }

.metric-card {
    background: linear-gradient(135deg, #1e2433 0%, #252d3d 100%);
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
}
.metric-label  { color: #94a3b8; font-size: 0.78rem; font-weight: 500;
                  text-transform: uppercase; letter-spacing: 0.08em; }
.metric-value  { color: #f1f5f9; font-size: 1.8rem; font-weight: 700; margin-top: 2px; }
.metric-sub    { color: #64748b; font-size: 0.8rem; margin-top: 2px; }

.pred-benign  { color: #22c55e !important; }
.pred-attack  { color: #ef4444 !important; }
.pred-warning { color: #f59e0b !important; }

.alert-box {
    border-radius: 8px; padding: 1rem 1.2rem; margin: 0.6rem 0;
    border-left: 4px solid;
}
.alert-danger  { background: #1f0f0f; border-color: #ef4444; color: #fca5a5; }
.alert-warning { background: #1f1a0f; border-color: #f59e0b; color: #fcd34d; }
.alert-success { background: #0f1f12; border-color: #22c55e; color: #86efac; }
.alert-info    { background: #0f1520; border-color: #3b82f6; color: #93c5fd; }

.section-header {
    color: #e2e8f0; font-size: 1rem; font-weight: 600;
    border-bottom: 1px solid #2d3748; padding-bottom: 0.4rem;
    margin: 1.2rem 0 0.8rem 0;
}
.kpi-badge {
    display: inline-block; background: #1e293b; border: 1px solid #334155;
    border-radius: 6px; padding: 2px 8px; font-size: 0.75rem; color: #94a3b8;
    margin: 2px;
}
.llm-report {
    background: #131921; border: 1px solid #1e3a5f; border-radius: 10px;
    padding: 1.2rem; font-family: monospace; font-size: 0.82rem;
    color: #c9d1d9; line-height: 1.7;
}
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: #1e2433; border-radius: 8px 8px 0 0;
    padding: 0.5rem 1.2rem; color: #94a3b8;
}
.stTabs [aria-selected="true"] { background: #2d3748; color: #f1f5f9; }
</style>
""", unsafe_allow_html=True)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(BASE_DIR, "netslab_episode_kb_v3")

# ── Load model (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading CEM-RAG model...")
def load_model():
    from inference import CEMRAGInference
    return CEMRAGInference(kb_dir=KB_DIR)


@st.cache_data(show_spinner=False)
def load_kpi_names():
    with open(os.path.join(KB_DIR, "kpi_names.json")) as f:
        return json.load(f)


# ── Plotly helpers ─────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="#0f1117",
    plot_bgcolor="#131921",
    font=dict(family="Inter", color="#94a3b8", size=11),
    margin=dict(l=40, r=20, t=40, b=40),
)


def kpi_timeseries_fig(kpi_z: np.ndarray, kpi_names, top_n: int = 8):
    """Multi-line time series of z-scored KPI values."""
    z_means = np.abs(kpi_z.mean(axis=0))
    top_idx = np.argsort(z_means)[::-1][:top_n]

    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for ci, i in enumerate(top_idx):
        fig.add_trace(go.Scatter(
            x=list(range(24)),
            y=kpi_z[:, i].tolist(),
            name=kpi_names[i],
            mode="lines",
            line=dict(width=2, color=colors[ci % len(colors)]),
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="#2d3748", line_width=1)
    fig.add_hline(y=2, line_dash="dot", line_color="#ef4444", line_width=1,
                  annotation_text="2σ", annotation_position="right")
    fig.add_hline(y=-2, line_dash="dot", line_color="#ef4444", line_width=1)
    fig.update_layout(
        **PLOT_LAYOUT,
        title="KPI Time Series (z-scored, 24 timestamps)",
        xaxis_title="Timestamp",
        yaxis_title="z-score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"),
        height=320,
    )
    return fig


def radar_fig(z_scores: dict):
    """Spider / radar chart of per-KPI z-score magnitude."""
    names = list(z_scores.keys())
    vals  = [abs(v) for v in z_scores.values()]
    # Close the loop
    names.append(names[0])
    vals.append(vals[0])

    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=names,
        fill="toself",
        fillcolor="rgba(59,130,246,0.15)",
        line=dict(color="#3b82f6", width=2),
        name="Anomaly magnitude",
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        polar=dict(
            bgcolor="#131921",
            radialaxis=dict(visible=True, range=[0, max(vals) * 1.2 + 0.5],
                            gridcolor="#2d3748", linecolor="#2d3748",
                            tickfont=dict(size=9)),
            angularaxis=dict(gridcolor="#2d3748", linecolor="#2d3748",
                             tickfont=dict(size=9)),
        ),
        showlegend=False,
        title="KPI Anomaly Radar",
        height=360,
    )
    return fig


def prediction_bar_fig(scores: dict, pred_key: str, title: str):
    """Horizontal confidence bars."""
    total = sum(scores.values()) + 1e-9
    labels = list(scores.keys())
    values = [v / total * 100 for v in scores.values()]
    colors = ["#3b82f6" if l == pred_key else "#2d3748" for l in labels]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(size=10, color="#94a3b8"),
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        title=title,
        xaxis=dict(range=[0, 115], showgrid=False, visible=False),
        yaxis=dict(autorange="reversed"),
        height=max(180, 40 * len(labels) + 60),
        bargap=0.35,
    )
    return fig


def retrieval_fig(retrieved_labels):
    """Bar chart of retrieved episodes colored by label."""
    cats  = [r["category"] for r in retrieved_labels]
    subs  = [r["subcategory"] for r in retrieved_labels]
    sims  = [r["score"] for r in retrieved_labels]
    idxs  = list(range(1, len(sims) + 1))

    color_map = {}
    palette = px.colors.qualitative.Pastel
    for i, c in enumerate(sorted(set(cats))):
        color_map[c] = palette[i % len(palette)]

    fig = go.Figure(go.Bar(
        x=idxs, y=sims,
        marker_color=[color_map[c] for c in cats],
        text=subs,
        textposition="inside",
        textfont=dict(size=9),
        customdata=cats,
        hovertemplate="Rank %{x}<br>%{customdata} / %{text}<br>Score: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        title="Retrieved Historical Episodes (top-10)",
        xaxis_title="Retrieval rank",
        yaxis_title="Similarity score",
        height=240,
    )
    return fig


def get_alert_class(pred_category: str) -> str:
    if pred_category == "Benign":
        return "alert-success"
    if pred_category in ("DDoS", "DoS"):
        return "alert-danger"
    return "alert-warning"


def render_prediction(result, kpi_names):
    """Render prediction metrics + charts."""
    is_attack = result.pred_category != "Benign"
    conf_pct  = result.subcategory_conf * 100

    # Top banner
    alert_cls = get_alert_class(result.pred_category)
    icon = "🔴" if is_attack else "🟢"
    st.markdown(f"""
    <div class="alert-box {alert_cls}">
        <strong>{icon} {result.pred_category} / {result.pred_subcategory}</strong>
        &nbsp;&nbsp;—&nbsp;&nbsp;Confidence: <strong>{conf_pct:.1f}%</strong>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Predicted Category</div>
            <div class="metric-value {'pred-attack' if is_attack else 'pred-benign'}">{result.pred_category}</div>
            <div class="metric-sub">Conf: {result.category_conf*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Predicted Attack Type</div>
            <div class="metric-value">{result.pred_subcategory}</div>
            <div class="metric-sub">Conf: {conf_pct:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        top_kpi, top_z = result.anomalous_kpis[0] if result.anomalous_kpis else ("N/A", 0)
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Root KPI (most anomalous)</div>
            <div class="metric-value">{top_kpi}</div>
            <div class="metric-sub">z-score: {top_z:+.2f}σ</div>
        </div>""", unsafe_allow_html=True)

    # Charts row 1
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(
            kpi_timeseries_fig(result.raw_kpi_array, kpi_names),
            use_container_width=True, theme=None, key="ts")
    with c2:
        st.plotly_chart(
            radar_fig(result.kpi_z_scores),
            use_container_width=True, theme=None, key="radar")

    # Charts row 2
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(
            prediction_bar_fig(result.top_categories, result.pred_category,
                               "Category Vote Weights"),
            use_container_width=True, theme=None, key="cat_bars")
    with c4:
        st.plotly_chart(
            prediction_bar_fig(result.top_subcategories, result.pred_subcategory,
                               "Subcategory Vote Weights"),
            use_container_width=True, theme=None, key="sub_bars")

    # Retrieval chart
    st.plotly_chart(retrieval_fig(result.retrieved_labels),
                    use_container_width=True, theme=None, key="retrieval")

    # Anomalous KPIs list
    st.markdown('<div class="section-header">Top Anomalous KPIs</div>',
                unsafe_allow_html=True)
    cols = st.columns(6)
    for ci, (kpi, z) in enumerate(result.anomalous_kpis[:6]):
        arrow = "▲" if z > 0 else "▼"
        color = "#ef4444" if abs(z) > 2 else ("#f59e0b" if abs(z) > 1 else "#22c55e")
        cols[ci % 6].markdown(f"""
        <div class="metric-card" style="padding:0.7rem 0.9rem;">
            <div class="metric-label">{kpi}</div>
            <div class="metric-value" style="font-size:1.2rem;color:{color};">{arrow} {z:+.2f}σ</div>
            <div class="metric-sub">{result.kpi_trends.get(kpi,'flat')}</div>
        </div>""", unsafe_allow_html=True)

    # Graph edges
    if result.graph_edges:
        st.markdown('<div class="section-header">Strongest KPI Correlations</div>',
                    unsafe_allow_html=True)
        for a, b, r in result.graph_edges:
            bar_w = int(abs(r) * 100)
            color = "#3b82f6" if r > 0 else "#ef4444"
            sign  = "+" if r > 0 else ""
            st.markdown(
                f'<span class="kpi-badge">{a}</span> → '
                f'<span class="kpi-badge">{b}</span> &nbsp;'
                f'<span style="color:{color};font-size:0.85rem;">'
                f'r = {sign}{r:.2f}</span>',
                unsafe_allow_html=True
            )


def call_llm(result, model) -> str:
    """Call the 3-stage LLM chain (blocking, shown in spinner)."""
    try:
        import requests, json as _json

        def _chat(prompt: str, timeout: int = 180) -> str:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3.2:3b", "prompt": prompt,
                      "stream": False, "options": {"temperature": 0.2}},
                timeout=timeout,
            )
            return resp.json().get("response", "").strip()

        kpi_anom = "; ".join(f"{k}: {v:+.2f}σ" for k, v in result.anomalous_kpis[:5])
        edges    = "; ".join(f"{a}→{b} (r={r:+.2f})" for a, b, r in result.graph_edges[:3])
        trends   = "; ".join(f"{k}: {v}"
                             for k, v in result.kpi_trends.items()
                             if v != "flat")[:200]

        # Stage 1 — Evidence Interpreter
        s1_prompt = f"""You are a 5G RAN security expert analyzing KPI data.

ANOMALOUS KPIs: {kpi_anom}
CORRELATION EDGES: {edges if edges else 'none detected'}
KPI TRENDS: {trends if trends else 'all flat'}

Do NOT use the attack label. Reason purely from evidence.
Respond in JSON with keys: "radio_layer_interpretation", "linked_effects", "network_layer_context"
Keep each value to 2 sentences maximum."""

        s1_raw  = _chat(s1_prompt)
        try:
            s1 = _json.loads(s1_raw[s1_raw.find("{"):s1_raw.rfind("}")+1])
        except Exception:
            s1 = {"radio_layer_interpretation": s1_raw[:300],
                  "linked_effects": "", "network_layer_context": ""}

        # Stage 2 — Historical Analyst
        top3 = result.retrieved_labels[:3]
        hist = "\n".join(
            f"  Ep {i+1}: {r['category']}/{r['subcategory']} (sim={r['score']:.3f})"
            for i, r in enumerate(top3)
        )
        s2_prompt = f"""You are comparing a current incident to historical episodes.

CURRENT RADIO INTERPRETATION: {s1.get('radio_layer_interpretation','')}

RETRIEVED HISTORICAL INCIDENTS:
{hist}

Compare and respond in JSON with keys: "historical_comparison", "key_differences", "escalation_signal"
Keep each value to 2 sentences maximum."""

        s2_raw = _chat(s2_prompt)
        try:
            s2 = _json.loads(s2_raw[s2_raw.find("{"):s2_raw.rfind("}")+1])
        except Exception:
            s2 = {"historical_comparison": s2_raw[:300],
                  "key_differences": "", "escalation_signal": "Unknown"}

        # Stage 3 — Operator Report + Confidence Review
        conf = result.subcategory_conf
        s3_prompt = f"""You are a NOC operator writing a security incident report.

INCIDENT: {result.pred_category} / {result.pred_subcategory} (conf={conf:.1%})
RADIO FINDING: {s1.get('radio_layer_interpretation','')}
HISTORICAL CONTEXT: {s2.get('historical_comparison','')}
ESCALATION SIGNAL: {s2.get('escalation_signal','')}

Write a report in JSON with keys:
  "severity" (1-5 integer),
  "severity_rationale" (1 sentence),
  "affected_components" (list of strings),
  "immediate_actions" (list of 3 strings),
  "escalation_triggers" (1 sentence),
  "prediction_audit" (2 sentences: {'confirm evidence' if conf >= 0.85 else 'CHALLENGE prediction, propose alternative'}),
  "requires_human_review" (boolean)"""

        s3_raw = _chat(s3_prompt)
        try:
            s3 = _json.loads(s3_raw[s3_raw.find("{"):s3_raw.rfind("}")+1])
        except Exception:
            s3 = {"severity": 3, "severity_rationale": s3_raw[:200],
                  "affected_components": [], "immediate_actions": [],
                  "escalation_triggers": "", "prediction_audit": "",
                  "requires_human_review": conf < 0.85}

        # Format report
        sev = s3.get("severity", 3)
        sev_bar = "█" * sev + "░" * (5 - sev)
        actions = "\n  ".join(f"{i+1}. {a}"
                               for i, a in enumerate(s3.get("immediate_actions", [])))
        comps   = ", ".join(s3.get("affected_components", []))
        review  = "YES — human review recommended" \
                  if s3.get("requires_human_review") else \
                  "NO — automated classification sufficient"

        return f"""{'─'*60}
  INCIDENT REPORT  |  Severity: {sev}/5  {sev_bar}
  Attack: {result.pred_category} / {result.pred_subcategory}
  Confidence: {conf:.1%}
{'─'*60}

[1] Radio Layer Interpretation
  {s1.get('radio_layer_interpretation','')}
  {s1.get('linked_effects','')}

[2] Historical Evidence
  {s2.get('historical_comparison','')}
  Differences: {s2.get('key_differences','')}
  Escalation signal: {s2.get('escalation_signal','')}

[3] Operational Actions
  Affected: {comps or 'RAN cell, UE uplink channel'}
  {actions}
  Escalate if: {s3.get('escalation_triggers','')}

[4] Prediction Audit
  {s3.get('prediction_audit','')}
  Human review: {review}
{'─'*60}"""

    except Exception as e:
        return f"[LLM error: {e}]"


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ CEM-RAG IDS")
    st.markdown("**Two-Tower Contrastive IDS**\n\nNetSlab 5G Dataset · 14 attack types")
    st.divider()

    mode = st.radio(
        "Demo Mode",
        ["🎮 Scenario Explorer", "📝 Manual Input", "📡 Live Monitor"],
        label_visibility="collapsed",
    )
    st.divider()

    enable_llm = st.toggle("Enable LLM Report", value=False)
    st.caption("ℹ️ Standalone IDS mode active. LLM Analyst (Qwen-4B fine-tuned) integration in progress.")
    st.divider()

    st.markdown("**Model Performance**")
    st.markdown("""
    | Metric | Score |
    |--------|-------|
    | Subcategory Acc | **99.45%** |
    | Macro F1 | **0.9956** |
    | KB Vectors | ~8,000 |
    """)


# ── Load model ─────────────────────────────────────────────────────────────
try:
    model = load_model()
    kpi_names = load_kpi_names()
    model_ok = True
except Exception as e:
    st.error(f"Failed to load model: {e}\n\nMake sure you have run "
             "`python complete_pipeline.py --skip-training` first.")
    model_ok = False
    st.stop()

# Pre-load exact test windows for the scenarios
from scenario_generator import load_demo_scenarios
load_demo_scenarios(scenario_file=os.path.join(BASE_DIR, "demo_scenarios.json"))


# ===========================================================================
# MODE 1: SCENARIO EXPLORER
# ===========================================================================
if mode == "🎮 Scenario Explorer":
    from scenario_generator import SCENARIO_REGISTRY

    st.title("🎮 Scenario Explorer")
    st.markdown(
        "Pick an attack scenario, adjust the intensity sliders, and see how "
        "the CEM-RAG model responds in real time."
    )

    col_sel, col_desc = st.columns([1, 2])
    with col_sel:
        scenario_key = st.selectbox(
            "Attack Scenario",
            list(SCENARIO_REGISTRY.keys()),
            index=1,
        )

    scenario = SCENARIO_REGISTRY[scenario_key]
    with col_desc:
        st.markdown(f"""
        <div class="alert-box alert-info" style="margin-top:1.8rem;">
            <strong>{scenario_key}</strong><br>
            {scenario['description']}
        </div>
        """, unsafe_allow_html=True)

    # Sliders
    st.markdown('<div class="section-header">Attack Parameters</div>',
                unsafe_allow_html=True)
    slider_cols = st.columns(len(scenario["sliders"]))
    slider_vals = {}
    for ci, (key, (label, mn, mx, default)) in enumerate(scenario["sliders"].items()):
        slider_vals[key] = slider_cols[ci].slider(label, mn, mx, default, step=0.05)

    if st.button("▶  Run Analysis", type="primary", use_container_width=True):
        with st.spinner("Generating synthetic trace and running inference..."):
            arr = scenario["fn"](**slider_vals)
            result = model.predict(arr, col_names=kpi_names)

        render_prediction(result, kpi_names)

        if enable_llm:
            st.markdown('<div class="section-header">🧠 LLM Incident Report</div>',
                        unsafe_allow_html=True)
            with st.spinner("Generating 3-stage LLM report... (this takes ~60-120s)"):
                report = call_llm(result, model)
            st.markdown(f'<div class="llm-report">{report}</div>',
                        unsafe_allow_html=True)


# ===========================================================================
# MODE 2: MANUAL INPUT
# ===========================================================================
elif mode == "📝 Manual Input":
    from scenario_generator import SCENARIO_REGISTRY

    st.title("📝 Manual KPI Input")
    st.markdown(
        "Type your own KPI values **or** auto-fill from a real test scenario. "
        "The model needs 24 timesteps — demo scenarios load the full real window; "
        "manual values are projected as a steady-state 24-step window."
    )

    # Only arr24 and source need explicit state; the widget values live in
    # st.session_state[f"manual_in_{kpi}"] automatically via the key= param.
    if "manual_arr24"  not in st.session_state:
        st.session_state.manual_arr24  = None
    if "manual_source" not in st.session_state:
        st.session_state.manual_source = "manual"

    # Initialise widget keys to 0.0 on first load
    for kpi in kpi_names:
        if f"manual_in_{kpi}" not in st.session_state:
            st.session_state[f"manual_in_{kpi}"] = 0.0

    # ── Demo Fill buttons ────────────────────────────────────────────────────
    st.markdown("#### ⚡ Demo Auto-Fill (uses full real 24-step window)")
    demo_cols = st.columns(4)
    demo_options = [
        ("DDoS / DDoS_UDP",             "💉 DDoS UDP",      demo_cols[0]),
        ("DDoS / DDoS_SYN",             "💉 DDoS SYN",      demo_cols[1]),
        ("BruteForce / BruteForce_SSH", "🔑 BruteForce SSH", demo_cols[2]),
        ("Benign",                       "✅ Benign",         demo_cols[3]),
    ]
    for scen_key, label, col in demo_options:
        if col.button(label, use_container_width=True):
            full_arr = SCENARIO_REGISTRY[scen_key]["fn"](noise=0.0)  # (24, 16)
            st.session_state.manual_arr24  = full_arr
            st.session_state.manual_source = scen_key
            # Write directly into the widget keys so inputs update on rerun
            mean_row = full_arr.mean(axis=0)
            for i, kpi in enumerate(kpi_names):
                st.session_state[f"manual_in_{kpi}"] = float(mean_row[i])
            st.rerun()

    # Source banner
    if st.session_state.manual_source != "manual":
        st.success(
            f"✅ Loaded real 24-step window: **{st.session_state.manual_source}**  "
            "— boxes show the time-averaged mean of each KPI."
        )
    else:
        st.info("ℹ️ Manual mode — values will be tiled 24× as a steady-state window.")

    st.divider()
    st.markdown("#### 🔢 KPI Values")

    # 4-column input grid — key= lets Streamlit own the widget value
    grid = st.columns(4)
    for i, kpi in enumerate(kpi_names):
        grid[i % 4].number_input(
            kpi,
            format="%.4f",
            key=f"manual_in_{kpi}",   # Streamlit reads/writes state here directly
        )

    st.divider()

    if st.button("▶  Run Analysis", type="primary", use_container_width=True):
        if st.session_state.manual_arr24 is not None:
            arr_24 = st.session_state.manual_arr24.astype(np.float32)
            st.caption(f"Using real 24-step window: **{st.session_state.manual_source}**")
        else:
            arr_1d = np.array(
                [float(st.session_state[f"manual_in_{k}"]) for k in kpi_names],
                dtype=np.float32,
            )
            arr_24 = np.tile(arr_1d, (24, 1)).astype(np.float32)
            st.warning(
                "Using a tiled steady-state approximation. "
                "For best results, use one of the Demo Auto-Fill buttons above."
            )

        with st.spinner("Running inference..."):
            result = model.predict(arr_24, col_names=kpi_names)

        render_prediction(result, kpi_names)

        if enable_llm:
            st.markdown('<div class="section-header">🧠 LLM Incident Report</div>',
                        unsafe_allow_html=True)
            with st.spinner("Generating 3-stage LLM report..."):
                report = call_llm(result, model)
            st.markdown(f'<div class="llm-report">{report}</div>', unsafe_allow_html=True)



# ===========================================================================
# MODE 3: LIVE MONITOR
# ===========================================================================
elif mode == "📡 Live Monitor":
    from scenario_generator import SCENARIO_REGISTRY, generate_benign

    st.title("📡 Live Network Monitor")
    st.markdown(
        "Simulates continuous KPI monitoring. Click **Inject Attack** to switch "
        "the signal to an attack pattern and watch the model raise an alert."
    )

    # Session state
    if "live_running"  not in st.session_state: st.session_state.live_running  = False
    if "live_history"  not in st.session_state: st.session_state.live_history  = deque(maxlen=80)
    if "live_attack"   not in st.session_state: st.session_state.live_attack   = None
    if "live_pred"     not in st.session_state: st.session_state.live_pred     = None
    if "live_t"        not in st.session_state: st.session_state.live_t        = 0
    if "attack_start"  not in st.session_state: st.session_state.attack_start  = None

    # Controls row
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 2, 1])
    with ctrl1:
        if st.button("▶ Start", use_container_width=True, type="primary"):
            st.session_state.live_running = True
            st.session_state.live_history.clear()
            st.session_state.live_t = 0
            st.session_state.live_attack = None
    with ctrl2:
        if st.button("🛑 Stop", use_container_width=True):
            st.session_state.live_running = False
    with ctrl3:
        inject_key = st.selectbox(
            "Attack to inject",
            [k for k in SCENARIO_REGISTRY if k != "Benign"],
            label_visibility="collapsed",
        )
    with ctrl4:
        if st.button("💉 Inject", use_container_width=True):
            st.session_state.live_attack  = inject_key
            st.session_state.attack_start = st.session_state.live_t

    # Live chart placeholder
    chart_ph  = st.empty()
    pred_ph   = st.empty()
    alert_ph  = st.empty()

    if st.session_state.live_running:
        rng = np.random.default_rng(st.session_state.live_t)

        # Generate one new row
        t = st.session_state.live_t
        if st.session_state.live_attack is not None:
            scen = SCENARIO_REGISTRY[st.session_state.live_attack]
            full_arr = scen["fn"](noise=0.0)
            new_row = full_arr[t % 24]
        else:
            benign_arr = generate_benign(noise=0.0)
            new_row = benign_arr[t % 24]

        st.session_state.live_history.append(new_row)
        st.session_state.live_t += 1

        hist_arr = np.array(list(st.session_state.live_history), dtype=np.float32)

        # Run inference every 4 steps
        if t % 4 == 0 and len(hist_arr) >= 4:
            result = model.predict(hist_arr, col_names=kpi_names)
            st.session_state.live_pred = result

        # Draw chart (top 4 KPIs by variance in recent history)
        if len(hist_arr) > 1:
            var_idx = np.argsort(hist_arr.var(axis=0))[::-1][:4]
            fig = go.Figure()
            colors = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b"]
            for ci, ki in enumerate(var_idx):
                fig.add_trace(go.Scatter(
                    y=hist_arr[:, ki].tolist(),
                    name=kpi_names[ki],
                    mode="lines",
                    line=dict(color=colors[ci], width=2),
                ))
            if st.session_state.attack_start is not None:
                fig.add_vline(
                    x=st.session_state.attack_start,
                    line_dash="dash", line_color="#ef4444",
                    annotation_text="Attack injected",
                    annotation_position="top right",
                )
            fig.update_layout(
                **PLOT_LAYOUT,
                title=f"Live KPI Stream (t={t})",
                height=300,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            bgcolor="rgba(0,0,0,0)"),
            )
            chart_ph.plotly_chart(fig, use_container_width=True, theme=None, key=f"live_{t}")

        # Prediction banner
        if st.session_state.live_pred is not None:
            r = st.session_state.live_pred
            is_att = r.pred_category != "Benign"
            cls    = get_alert_class(r.pred_category)
            icon   = "🔴" if is_att else "🟢"
            pred_ph.markdown(f"""
            <div class="alert-box {cls}">
                {icon} <strong>{r.pred_category} / {r.pred_subcategory}</strong>
                &nbsp;—&nbsp; Confidence: <strong>{r.subcategory_conf*100:.1f}%</strong>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                Root KPI: <strong>{r.anomalous_kpis[0][0] if r.anomalous_kpis else 'N/A'}</strong>
                ({r.anomalous_kpis[0][1]:+.2f}σ if r.anomalous_kpis else '')
            </div>
            """, unsafe_allow_html=True)

        time.sleep(0.4)
        st.rerun()
    else:
        st.info("Press **▶ Start** to begin live monitoring simulation.")


"""
FIA Cost Cap Risk Simulator — Streamlit Dashboard
==================================================
Run with: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from src.simulation import (
    run_simulation,
    DEFAULT_CATEGORIES,
    SimulationConfig,
    CostCategory,
    scenario_comparison,
    COST_CAP_2025,
)

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FIA Cost Cap Risk Simulator",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styling ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px;
    }
    h1, h2, h3 { color: #E5E7EB !important; }
    .stMetric label { color: #9CA3AF !important; font-size: 11px; }
</style>
""", unsafe_allow_html=True)

CATEGORY_COLORS = {
    "Race Operations":           "#E8002D",
    "Aerodynamic Development":   "#FF8700",
    "Chassis & Bodywork":        "#FFC906",
    "Power Unit (Leased)":       "#00D2BE",
    "Manufacturing & Tooling":   "#0067FF",
    "Transmission & Suspension": "#9B59B6",
    "Electronics & Software":    "#2ECC71",
    "Contingency & Risk Reserve":"#95A5A6",
}

# ─── Sidebar — Controls ───────────────────────────────────────────────────────

st.sidebar.title("🎛️ Cost Category Controls")
st.sidebar.caption("Adjust baseline spend per category. Monte Carlo samples within uncertainty bounds.")

n_sims = st.sidebar.select_slider(
    "Simulation runs", options=[1000, 5000, 10000, 25000], value=10000
)

custom_categories = []
for cat in DEFAULT_CATEGORIES:
    st.sidebar.markdown(f"**{cat.name}**")
    baseline = st.sidebar.slider(
        f"Baseline ($M)", cat.lower_bound_usd_m, cat.upper_bound_usd_m,
        cat.baseline_usd_m, 0.5,
        key=f"baseline_{cat.name}"
    )
    custom_categories.append(CostCategory(
        name=cat.name,
        baseline_usd_m=baseline,
        lower_bound_usd_m=cat.lower_bound_usd_m,
        upper_bound_usd_m=cat.upper_bound_usd_m,
        distribution=cat.distribution,
        description=cat.description,
    ))

# ─── Run Simulation ───────────────────────────────────────────────────────────

config = SimulationConfig(n_simulations=n_sims, random_seed=42)
results = run_simulation(custom_categories, config)

# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown("# 🏎️ FIA Cost Cap Risk Simulator")
st.markdown(
    f"Monte Carlo analysis · **{n_sims:,} simulations** · "
    f"FIA Financial Regulations 2025 · **${COST_CAP_2025}M cap**"
)
st.divider()

# ─── KPI Row ─────────────────────────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)

breach_pct = results["breach_prob"] * 100
risk_emoji = "🟢" if breach_pct < 10 else "🟡" if breach_pct < 25 else "🔴"

col1.metric("Breach Probability", f"{breach_pct:.1f}%", f"{risk_emoji} P(Total > $135M)")
col2.metric("Minor Breach", f"{results['minor_breach_prob']*100:.1f}%", "0–5% over cap")
col3.metric("Material Breach", f"{results['material_breach_prob']*100:.1f}%", ">5% over cap")
col4.metric("Median Spend", f"${np.percentile(results['total_spend'], 50):.1f}M",
            f"P90: ${np.percentile(results['total_spend'], 90):.1f}M")
col5.metric("Expected Excess", f"${results['expected_excess']:.2f}M", "E[overspend | breach]")

st.divider()

# ─── Charts ───────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📊 Distribution", "🌪️ Sensitivity", "🎯 Scenarios"])

# ── Tab 1: Distribution ───────────────────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns([2, 1])

    with col_a:
        total = results["total_spend"]
        counts, edges = np.histogram(total, bins=60)
        mids = (edges[:-1] + edges[1:]) / 2

        colors = ["#E8002D" if m > COST_CAP_2025 else "#00D2BE" for m in mids]

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Bar(
            x=mids, y=counts, marker_color=colors,
            hovertemplate="Spend: $%{x:.1f}M<br>Simulations: %{y}<extra></extra>",
        ))
        fig_hist.add_vline(
            x=COST_CAP_2025, line_color="#E8002D", line_width=2,
            annotation_text="FIA CAP $135M", annotation_font_color="#E8002D",
        )
        fig_hist.update_layout(
            title="Season Spend Distribution", template="plotly_dark",
            xaxis_title="Total Season Spend ($M)", yaxis_title="Simulation Count",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=380,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        pcts = [5, 10, 25, 50, 75, 90, 95]
        pct_vals = [np.percentile(total, p) for p in pcts]

        fig_pct = go.Figure()
        fig_pct.add_trace(go.Scatter(
            x=pcts, y=pct_vals, mode="lines+markers",
            line=dict(color="#00D2BE", width=2),
            marker=dict(color=["#E8002D" if v > COST_CAP_2025 else "#00D2BE" for v in pct_vals], size=10),
        ))
        fig_pct.add_hline(y=COST_CAP_2025, line_color="#E8002D", line_dash="dash")
        fig_pct.update_layout(
            title="Percentile Curve", template="plotly_dark",
            xaxis_title="Percentile", yaxis_title="Spend ($M)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=380,
        )
        st.plotly_chart(fig_pct, use_container_width=True)

# ── Tab 2: Sensitivity ────────────────────────────────────────────────────────
with tab2:
    sens = results["sensitivity"]
    names = list(sens.keys())
    values = list(sens.values())
    sorted_pairs = sorted(zip(values, names), reverse=True)
    s_vals, s_names = zip(*sorted_pairs)
    colors_sens = [CATEGORY_COLORS.get(n, "#00D2BE") for n in s_names]

    fig_tornado = go.Figure()
    fig_tornado.add_trace(go.Bar(
        x=s_vals, y=s_names, orientation="h",
        marker_color=colors_sens,
        hovertemplate="%{y}<br>Correlation: %{x:.3f}<extra></extra>",
    ))
    fig_tornado.update_layout(
        title="Tornado Chart — Cost Driver Sensitivity (Pearson Correlation with Total Spend)",
        template="plotly_dark",
        xaxis_title="Correlation Coefficient",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=420, margin=dict(l=200),
    )
    st.plotly_chart(fig_tornado, use_container_width=True)

    st.info(
        f"**Top cost driver:** {s_names[0]} (correlation: {s_vals[0]:.3f})  \n"
        f"This category has the strongest influence on total season spend and therefore breach probability. "
        f"A 10% reduction in uncertainty here would have the highest impact on cost cap risk."
    )

# ── Tab 3: Scenarios ─────────────────────────────────────────────────────────
with tab3:
    scenario_cats = {
        "Conservative": [
            CostCategory(c.name, max(c.lower_bound_usd_m, c.baseline_usd_m * 0.90),
                         c.lower_bound_usd_m, c.upper_bound_usd_m * 0.95,
                         c.distribution) for c in custom_categories
        ],
        "Baseline": custom_categories,
        "Aggressive": [
            CostCategory(c.name, min(c.upper_bound_usd_m, c.baseline_usd_m * 1.08),
                         c.lower_bound_usd_m * 1.02, c.upper_bound_usd_m * 1.08,
                         c.distribution) for c in custom_categories
        ],
    }
    sc_df = scenario_comparison(scenario_cats)
    st.dataframe(sc_df, use_container_width=True, hide_index=True)

    breach_vals = [float(v.strip("%")) for v in sc_df["Breach Probability"]]
    fig_sc = go.Figure()
    fig_sc.add_trace(go.Bar(
        x=sc_df["Scenario"].tolist(),
        y=[float(v) for v in sc_df["P50 Spend (£M)"]],
        name="Median Spend", marker_color="#00D2BE",
    ))
    fig_sc.add_trace(go.Bar(
        x=sc_df["Scenario"].tolist(),
        y=[float(v) for v in sc_df["P90 Spend (£M)"]],
        name="P90 Spend", marker_color="#FF8700",
    ))
    fig_sc.add_hline(y=COST_CAP_2025, line_color="#E8002D", line_dash="dash",
                     annotation_text="Cost Cap", annotation_font_color="#E8002D")
    fig_sc.update_layout(
        title="Scenario Comparison vs FIA Cost Cap", template="plotly_dark",
        yaxis_title="Spend ($M)", barmode="group",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=380,
    )
    st.plotly_chart(fig_sc, use_container_width=True)

# ─── Summary Table ────────────────────────────────────────────────────────────

st.divider()
st.markdown("### Percentile Summary Table")
st.dataframe(results["summary"], use_container_width=True, hide_index=True)

st.caption(
    "FIA Financial Regulations 2025 · Distributions: Triangular (default), Normal (fixed costs), "
    "PERT (contingency) · Figures are illustrative estimates for analytical demonstration"
)
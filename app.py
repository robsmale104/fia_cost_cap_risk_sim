"""
FIA Cost Cap Risk Simulator — Streamlit Dashboard
==================================================
Run with: streamlit run app.py

Phase 2 Updates:
- Correct 2025 adjusted cap ($165M) vs baseline ($135M)
- Correlation toggle (Cholesky decomposition)
- Correlation impact analysis tab
- Regulatory context panel
- 2026 cap scenario ($215M) with correct 2026 cost category distributions
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from src.simulation import (
    run_simulation,
    DEFAULT_CATEGORIES,
    DEFAULT_CATEGORIES_2026,
    CATEGORIES_BY_YEAR,
    SimulationConfig,
    CostCategory,
    scenario_comparison,
    correlation_impact_analysis,
    COST_CAP_BASELINE,
    COST_CAP_2025_ADJUSTED,
    COST_CAP_2026_BASELINE,
    PU_CAP_BASELINE,
)

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FIA Cost Cap Risk Simulator",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0d1117; }
    h1, h2, h3 { color: #E5E7EB !important; }
    .stMetric label { color: #9CA3AF !important; font-size: 11px; }
    .regulatory-box {
        background: rgba(0,111,98,0.1);
        border: 1px solid rgba(0,111,98,0.3);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

CATEGORY_COLORS = {
    "Race Operations":            "#E8002D",
    "Aerodynamic Development":    "#FF8700",
    "Chassis & Bodywork":         "#FFC906",
    "Power Unit (Leased)":        "#00D2BE",
    "Manufacturing & Tooling":    "#0067FF",
    "Transmission & Suspension":  "#9B59B6",
    "Electronics & Software":     "#2ECC71",
    "Contingency & Risk Reserve": "#95A5A6",
}

# ─── Sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.title("🎛️ Simulation Controls")
st.sidebar.caption("Adjust parameters and cost category baselines.")

# Cap / regulation year selector
cap_choice = st.sidebar.radio(
    "Regulatory Scenario",
    options=["2025 Adjusted ($165M)", "Baseline ($135M)", "2026 New Regs ($215M)"],
    index=0,
    help=(
        "2025 adjusted cap reflects 24 race calendar + inflation indexation on $135M baseline."
        "2026 uses updated cost category distributions reflecting new technical regulations"
        "wider uncertainty bounds due to low TRL across all categories."
    )
)

cap_map = {
    "2025 Adjusted ($165M)": COST_CAP_2025_ADJUSTED,
    "Baseline ($135M)":      COST_CAP_BASELINE,
    "2026 New Regs ($215M)": COST_CAP_2026_BASELINE,
}
active_cap = cap_map[cap_choice]

# Switch cost category set based on regulatory scenario
# 2026 uses higher baselines and wider bounds reflecting new regs + low TRL
reg_year = "2026" if cap_choice == "2026 New Regs ($215M)" else "2024"
base_categories = CATEGORIES_BY_YEAR[reg_year]

# Correlation toggle
apply_corr = st.sidebar.toggle(
    "Apply Correlation Matrix (Cholesky)",
    value=True,
    help=(
        "When ON: cost categories move together based on shared drivers — "
        "more realistic tail risk. "
        "When OFF: independent sampling (understates breach probability)."
    )
)

# Simulation runs
n_sims = st.sidebar.select_slider(
    "Simulation runs", options=[1000, 5000, 10000, 25000], value=10000
)

st.sidebar.divider()
st.sidebar.markdown("**Cost Category Baselines**")

# Show which reg set is active
if reg_year == "2026":
    st.sidebar.info(
        "⚠️ 2026 distributions active — baselines and uncertainty bounds "
        "reflect new technical regulations and low TRL across all categories."
    )
else:
    st.sidebar.caption("Drag to adjust. Monte Carlo samples within uncertainty bounds.")

# Build custom categories from the correct base set for the selected scenario
custom_categories = []
for cat in base_categories:
    st.sidebar.markdown(f"**{cat.name}**")
    baseline = st.sidebar.slider(
        f"Baseline ($M)",
        float(cat.lower_bound_usd_m),
        float(cat.upper_bound_usd_m),
        float(cat.baseline_usd_m),
        0.5,
        key=f"baseline_{cat.name}_{reg_year}"  # key includes reg year — forces reset on switch
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

config = SimulationConfig(
    n_simulations=n_sims,
    cost_cap_usd_m=active_cap,
    random_seed=42,
    apply_correlations=apply_corr,
)
results = run_simulation(custom_categories, config)
total = results["total_spend"]

total_baseline = sum(c.baseline_usd_m for c in custom_categories)
headroom = active_cap - total_baseline

# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown("# 🏎️ FIA Cost Cap Risk Simulator")
st.markdown(f"Monte Carlo analysis {n_sims:,} simulations")
st.markdown(f"Scenario:{cap_choice}")
st.markdown(f"Correlations: {'ON — Cholesky' if apply_corr else 'OFF — Independent'}")
st.markdown(f"Baseline headroom: ${headroom:.0f}M")

st.divider()

# ─── KPI Row ─────────────────────────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)

breach_pct = results["breach_prob"] * 100
risk_emoji = "🟢" if breach_pct < 10 else "🟡" if breach_pct < 25 else "🔴"

col1.metric(
    "Breach Probability",
    f"{breach_pct:.1f}%",
    f"{risk_emoji} P(Total > ${active_cap:.0f}M)"
)
col2.metric(
    "Minor Breach",
    f"{results['minor_breach_prob']*100:.1f}%",
    "0–5% over cap"
)
col3.metric(
    "Material Breach",
    f"{results['material_breach_prob']*100:.1f}%",
    "≥5% over cap"
)
col4.metric(
    "Median Spend",
    f"${np.percentile(total, 50):.1f}M",
    f"P90: ${np.percentile(total, 90):.1f}M"
)
col5.metric(
    "Expected Excess",
    f"${results['expected_excess']:.2f}M",
    "Amount over cap"
)

# 2026 specific callout — highlight the reduced headroom finding
if reg_year == "2026":
    st.warning(
        f"**2026 Regulations:** Total baseline spend of \${total_baseline:.0f}M leaves only "
        f"\${headroom:.0f}M headroom against the \${active_cap:.0f}M cap — despite the $80M cap increase. "
        f"Wide uncertainty bounds from lower TRL in 2026, drives breach probability to \{breach_pct:.1f}%, "
        f"higher than the 2025 scenario. The \$80M cap increase is largely consumed by new reg development costs."
    )

st.divider()

# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Distribution",
    "🌪️ Sensitivity",
    "🔗 Correlation Impact",
    "🎯 Scenarios",
    "📋 Regulatory Context",
])

# ── Tab 1: Distribution ───────────────────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns([2, 1])

    with col_a:
        counts, edges = np.histogram(total, bins=60)
        mids = (edges[:-1] + edges[1:]) / 2
        colors = ["#E8002D" if m > active_cap else "#00D2BE" for m in mids]

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Bar(
            x=mids, y=counts, marker_color=colors,
            hovertemplate="Spend: $%{x:.1f}M<br>Simulations: %{y}<extra></extra>",
        ))
        fig_hist.add_vline(
            x=active_cap, line_color="#E8002D", line_width=2,
            annotation_text=f"Cap ${active_cap:.0f}M",
            annotation_font_color="#E8002D",
        )
        if active_cap != COST_CAP_BASELINE:
            fig_hist.add_vline(
                x=COST_CAP_BASELINE, line_color="#FF8700", line_width=1,
                line_dash="dot",
                annotation_text=f"Baseline ${COST_CAP_BASELINE:.0f}M",
                annotation_font_color="#FF8700",
            )
        fig_hist.update_layout(
            title=f"Season Spend Distribution — {cap_choice}",
            template="plotly_dark",
            xaxis_title="Total Season Spend ($M)",
            yaxis_title="Simulation Count",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
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
            marker=dict(
                color=["#E8002D" if v > active_cap else "#00D2BE" for v in pct_vals],
                size=10
            ),
        ))
        fig_pct.add_hline(
            y=active_cap, line_color="#E8002D", line_dash="dash",
            annotation_text=f"Cap ${active_cap:.0f}M"
        )
        fig_pct.update_layout(
            title="Percentile Curve",
            template="plotly_dark",
            xaxis_title="Percentile",
            yaxis_title="Spend ($M)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
        )
        st.plotly_chart(fig_pct, use_container_width=True)

    # Percentile band
    p_cols = st.columns(5)
    for i, (p, v) in enumerate(zip(pcts[1:-1], pct_vals[1:-1])):
        p_cols[i].metric(
            f"P{p}", f"${v:.1f}M",
            delta="⚠️ Over cap" if v > active_cap else None,
            delta_color="inverse"
        )

# ── Tab 2: Sensitivity ────────────────────────────────────────────────────────
with tab2:
    sens = results["sensitivity"]
    sorted_pairs = sorted(zip(sens.values(), sens.keys()), reverse=True)
    s_vals, s_names = zip(*sorted_pairs)
    colors_sens = [CATEGORY_COLORS.get(n, "#00D2BE") for n in s_names]

    fig_tornado = go.Figure()
    fig_tornado.add_trace(go.Bar(
        x=s_vals, y=s_names, orientation="h",
        marker_color=colors_sens,
        hovertemplate="%{y}<br>Correlation: %{x:.3f}<extra></extra>",
    ))
    fig_tornado.update_layout(
        title="Tornado Chart — Cost Driver Sensitivity (Pearson r with Total Spend)",
        template="plotly_dark",
        xaxis_title="Correlation Coefficient",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin=dict(l=200),
    )
    st.plotly_chart(fig_tornado, use_container_width=True)

    st.info(
        f"**Top cost driver:** {s_names[0]} (r = {s_vals[0]:.3f})  \n"
        f"Highest correlation with total season spend — greatest influence on breach probability. "
        f"Note: with correlations ON, sensitivity values are higher because correlated categories "
        f"amplify each other's variance."
    )

# ── Tab 3: Correlation Impact ─────────────────────────────────────────────────
with tab3:
    st.markdown("### Correlation Impact Analysis")
    st.markdown(
        "Compares the **independent sampling** model (original) against the "
        "**Cholesky correlated** model (Phase 2). "
        "Independent sampling understates tail risk because it assumes cost categories "
        "move independently — in reality, shared drivers (staff costs, supply chain, "
        "upgrade packages) cause categories to move together."
    )

    with st.spinner("Running correlation comparison..."):
        impact_df = correlation_impact_analysis()

    st.dataframe(impact_df, use_container_width=True, hide_index=True)

    models = impact_df["Model"].tolist()
    p90_vals_corr = impact_df["P90 ($M)"].tolist()
    p95_vals_corr = impact_df["P95 ($M)"].tolist()
    breach_vals_corr = [float(v.strip("%")) for v in impact_df["Breach Prob"].tolist()]

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name="P90 Spend", x=models, y=p90_vals_corr,
            marker_color="#FF8700", opacity=0.85
        ))
        fig_comp.add_trace(go.Bar(
            name="P95 Spend", x=models, y=p95_vals_corr,
            marker_color="#E8002D", opacity=0.85
        ))
        fig_comp.add_hline(
            y=active_cap, line_color="#E8002D", line_dash="dash",
            annotation_text=f"Cap ${active_cap:.0f}M"
        )
        fig_comp.update_layout(
            title="Tail Spend: Independent vs Correlated",
            template="plotly_dark",
            barmode="group",
            yaxis_title="Spend ($M)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    with col_c2:
        fig_breach = go.Figure()
        fig_breach.add_trace(go.Bar(
            x=models, y=breach_vals_corr,
            marker_color=["#00D2BE", "#E8002D"],
            opacity=0.85,
        ))
        fig_breach.update_layout(
            title="Breach Probability: Independent vs Correlated",
            template="plotly_dark",
            yaxis_title="Breach Probability (%)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
        )
        st.plotly_chart(fig_breach, use_container_width=True)

    st.warning(
        "**Key finding:** The correlated model typically shows materially higher breach "
        "probability than the independent model — particularly at the tail (P90, P95). "
        "This is because correlated categories can all overspend simultaneously, "
        "driven by common cost pressures such as inflation, supply chain disruption, "
        "or a major mid season upgrade programme."
    )

# ── Tab 4: Scenarios ─────────────────────────────────────────────────────────
with tab4:
    st.markdown("### Scenario Comparison")
    st.markdown(
        "Conservative, Baseline, and Aggressive spend scenarios compared against "
        "the active cap. The 2026 scenario uses the 2026 cost distributions — "
        "wider bounds reflecting lower TRLs due to new regulations."
    )

    scenarios = {
        "Conservative": (
            [CostCategory(
                c.name,
                max(c.lower_bound_usd_m, c.baseline_usd_m * 0.90),
                c.lower_bound_usd_m,
                c.upper_bound_usd_m * 0.95,
                c.distribution
            ) for c in custom_categories],
            SimulationConfig(
                cost_cap_usd_m=active_cap,
                apply_correlations=apply_corr,
                n_simulations=5000
            )
        ),
        "Baseline": (
            custom_categories,
            SimulationConfig(
                cost_cap_usd_m=active_cap,
                apply_correlations=apply_corr,
                n_simulations=5000
            )
        ),
        "Aggressive": (
            [CostCategory(
                c.name,
                min(c.upper_bound_usd_m, c.baseline_usd_m * 1.08),
                c.lower_bound_usd_m,
                c.upper_bound_usd_m * 1.08,
                c.distribution
            ) for c in custom_categories],
            SimulationConfig(
                cost_cap_usd_m=active_cap,
                apply_correlations=apply_corr,
                n_simulations=5000
            )
        ),
        "2026 Regs ($215M)": (
            DEFAULT_CATEGORIES_2026,
            SimulationConfig(
                cost_cap_usd_m=COST_CAP_2026_BASELINE,
                apply_correlations=apply_corr,
                n_simulations=5000
            )
        ),
    }

    with st.spinner("Running scenario comparison..."):
        sc_df = scenario_comparison(scenarios)

    st.dataframe(sc_df, use_container_width=True, hide_index=True)

    fig_sc = go.Figure()
    fig_sc.add_trace(go.Bar(
        x=sc_df["Scenario"].tolist(),
        y=sc_df["P50 Spend ($M)"].tolist(),
        name="Median Spend ($M)",
        marker_color="#00D2BE",
    ))
    fig_sc.add_trace(go.Bar(
        x=sc_df["Scenario"].tolist(),
        y=sc_df["P90 Spend ($M)"].tolist(),
        name="P90 Spend ($M)",
        marker_color="#FF8700",
    ))
    fig_sc.add_hline(
        y=active_cap, line_color="#E8002D", line_dash="dash",
        annotation_text=f"Active Cap ${active_cap:.0f}M",
        annotation_font_color="#E8002D"
    )
    if active_cap != COST_CAP_2026_BASELINE:
        fig_sc.add_hline(
            y=COST_CAP_2026_BASELINE, line_color="#9B59B6", line_dash="dot",
            annotation_text="2026 Cap $215M",
            annotation_font_color="#9B59B6"
        )
    fig_sc.update_layout(
        title="Scenario Comparison vs FIA Cost Cap",
        template="plotly_dark",
        yaxis_title="Spend ($M)",
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
    )
    st.plotly_chart(fig_sc, use_container_width=True)

# ── Tab 5: Regulatory Context ─────────────────────────────────────────────────
with tab5:
    st.markdown("### FIA Financial Regulations — Context & History")

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.markdown("#### Cap Structure")
        cap_data = {
            "Regulation": [
                "Team Cap: Baseline",
                "Team Cap: 2025 Adjusted",
                "Team Cap: 2026 Baseline",
                "PU Manufacturer Cap (2023–2025)",
                "PU Manufacturer Cap (2026+)"
            ],
            "Value ($M)": [135, 165, 215, 95, 130],
            "Notes": [
                "Article 4 baseline, 2021–2025",
                "24-race calendar + inflation indexation",
                "New technical regulations",
                "Independent cap applies to PU manufacturers only",
                "New PU regulations from 2026"
            ]
        }
        st.dataframe(pd.DataFrame(cap_data), use_container_width=True, hide_index=True)

        st.markdown("#### Key Exclusions from Team Cap")
        st.markdown("""
        The $135M baseline **excludes**:
        - Driver salaries (top 3 earners)
        - Three highest paid non driver staff
        - Power unit development costs (covered by PU cap)
        - Marketing and promotional expenditure
        - HQ building and infrastructure costs
        """)

    with col_r2:
        st.markdown("#### Breach History")
        breach_data = {
            "Year": [2021, 2023, 2023, 2024],
            "Team": ["Red Bull Racing", "Honda Racing Corp", "Alpine Racing SAS", "Aston Martin"],
            "Type": ["Material Breach", "Procedural (ABA)", "Procedural (ABA)", "Minor Procedural"],
            "Penalty": ["10 pts + $7M fine", "$600k fine", "$400k fine", "No financial penalty"],
            "Detail": [
                "~$7M over cap set precedent for points deduction",
                "Incorrect dyno maintenance & inventory reporting",
                "Significant documentation deficiencies",
                "Cooperative response, unforeseen circumstances"
            ]
        }
        st.dataframe(pd.DataFrame(breach_data), use_container_width=True, hide_index=True)
        st.markdown("")
        st.markdown("")
        st.markdown("#### Breach Thresholds")
        st.markdown("""
        - **Minor breach:** <5% over cap → financial penalty + reprimand
        - **Material breach:** ≥5% over cap → can include constructor points deduction
        - **Accepted Breach Agreement (ABA):** procedural/reporting errors resolved cooperatively
        """)

    st.divider()
    st.markdown("#### 2026 Regulation Changes")
    st.info(
        "The 2026 Technical Regulations introduce significant changes: team cap increases from "
        "\$135M to \$215M baseline, PU cap increases from \$95M to \$130M. "
        "Despite the \$80M cap increase, projected development costs under new regulations "
        "consume most of this headroom — this simulator models 2026 spend distributions with "
        "wider uncertainty bounds reflecting low TRL across all categories. "
        "Switch to the 2026 scenario in the sidebar to explore this."
    )

# ─── Summary Table ────────────────────────────────────────────────────────────

st.divider()
st.markdown("### Percentile Summary Table")
st.dataframe(results["summary"], use_container_width=True, hide_index=True)

st.caption(
    f"FIA Financial Regulations 2024/2025/2026 · Active scenario: {cap_choice} · "
    f"Reg year: {reg_year} cost distributions · "
    f"Correlations: {'Cholesky — ON' if apply_corr else 'Independent — OFF'} · "
    f"Distributions: Triangular (default), Normal (fixed costs), PERT (contingency) · "
    f"Figures are illustrative estimates for analytical demonstration"
)
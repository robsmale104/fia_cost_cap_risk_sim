"""
FIA Cost Cap Risk Simulator - Core Monte Carlo Engine
=====================================================
Models season-level spend risk against FIA Financial Regulations.
Identifies cost categories with highest breach probability.

Phase 2 Updates:
- Correct cap figures: $135M baseline adjusted for inflation and race count
  - 2024 actual cap: ~$165M (24 races + inflation indexation)
  - 2026 cap: $215M baseline (new technical regulations)
- Correlation matrix via Cholesky decomposition
  - Categories no longer sampled independently
  - Correlated categories move together, better capturing tail risk
- Regulatory context: breach types, ABA process, historical cases

Key Regulatory Reference:
  FIA Financial Regulations 2024 — Article 4 (Relevant Costs)
  FIA Financial Regulations 2026 — Section D (updated caps)

Breach History:
  2021: Red Bull Racing — material breach (~$7M over), 10 point deduction + $7M fine
  2023: Honda Racing Corp — ABA, $600k fine (procedural, dyno/inventory reporting)
  2023: Alpine Racing SAS — ABA, $400k fine (procedural, documentation deficiencies)
  2024: Aston Martin — minor procedural breach, no financial penalty (cooperative response)

Author: [Your Name]
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

# ─── FIA Regulatory Constants ─────────────────────────────────────────────────

# Baseline caps (USD millions) — before inflation/race adjustments
COST_CAP_BASELINE = 135.0          # Team cap baseline 2021–2025
COST_CAP_2024_ADJUSTED = 165.0     # 2024 actual: 24 races + inflation indexation
COST_CAP_2025_ADJUSTED = 165.0     # Alias — same adjusted cap applies through 2025
COST_CAP_2026_BASELINE = 215.0     # 2026 new technical regulations

PU_CAP_BASELINE = 95.0             # Power Unit manufacturer cap (separate, independent)
PU_CAP_2026_BASELINE = 130.0       # PU cap from 2026

# Penalty thresholds
MINOR_BREACH_THRESHOLD = 0.05      # <5% over cap = minor breach (fine + reprimand)
MATERIAL_BREACH_THRESHOLD = 0.05   # >=5% over cap = material breach (points deduction)

# Default cap to use in simulation — set to 2024 adjusted figure
DEFAULT_CAP = COST_CAP_2024_ADJUSTED


# ─── Cost Category Definition ────────────────────────────────────────────────

@dataclass
class CostCategory:
    """
    Represents a single cost category with uncertainty bounds.

    Distributions:
        triangular  — use when you have a clear min/mode/max estimate
                      (most categories, calibrated from analyst estimates)
        normal      — use for quasi-fixed costs with symmetric uncertainty
                      (PU lease, race logistics — contractually bounded)
        pert        — use for contingency/risk items with right-skewed uncertainty
                      (unknown unknowns — PERT weights the mode more than triangular)
    """
    name: str
    baseline_usd_m: float
    lower_bound_usd_m: float
    upper_bound_usd_m: float
    distribution: str = "triangular"
    description: str = ""
    fia_regulated: bool = True


@dataclass
class SimulationConfig:
    """Configuration for a Monte Carlo run."""
    n_simulations: int = 10_000
    cost_cap_usd_m: float = DEFAULT_CAP
    random_seed: Optional[int] = 42
    apply_correlations: bool = True    # Phase 2: now enabled by default


# ─── Default Cost Categories ─────────────────────────────────────────────────
# Baselines calibrated to publicly available analyst estimates and
# FIA cost cap submission disclosures. Bounds represent ~P10-P90 uncertainty.
# Note: PU costs here represent LEASED unit cost to the constructor —
# the separate $95M PU manufacturer cap is not modelled here as it applies
# to manufacturers (Ferrari, Mercedes, Honda, Renault), not customer teams.

DEFAULT_CATEGORIES = [
    CostCategory(
        name="Aerodynamic Development",
        baseline_usd_m=28.0,
        lower_bound_usd_m=22.0,
        upper_bound_usd_m=38.0,
        description="CFD compute, wind tunnel time, aero parts manufacture. "
                    "High variance — development pace varies significantly by team philosophy.",
    ),
    CostCategory(
        name="Chassis & Bodywork",
        baseline_usd_m=22.0,
        lower_bound_usd_m=18.0,
        upper_bound_usd_m=29.0,
        description="Carbon fibre structures, monocoque, bodywork panels, crash structures.",
    ),
    CostCategory(
        name="Power Unit (Leased)",
        baseline_usd_m=15.0,
        lower_bound_usd_m=14.0,
        upper_bound_usd_m=17.5,
        description="PU lease cost to constructor — semi-fixed, subject to supplier pricing. "
                    "Note: PU manufacturer development costs sit under the separate $95M PU cap.",
        distribution="normal",
    ),
    CostCategory(
        name="Transmission & Suspension",
        baseline_usd_m=14.0,
        lower_bound_usd_m=11.0,
        upper_bound_usd_m=18.0,
        description="Gearbox, driveshafts, suspension geometry components.",
    ),
    CostCategory(
        name="Electronics & Software",
        baseline_usd_m=12.0,
        lower_bound_usd_m=10.0,
        upper_bound_usd_m=15.0,
        description="Control electronics, simulation software licences, data systems.",
    ),
    CostCategory(
        name="Race Operations",
        baseline_usd_m=38.0,
        lower_bound_usd_m=34.0,
        upper_bound_usd_m=44.0,
        description="Travel, freight (24-race calendar), garage, trackside personnel & equipment. "
                    "24-race calendar significantly increases this vs earlier seasons.",
        distribution="normal",
    ),
    CostCategory(
        name="Manufacturing & Tooling",
        baseline_usd_m=18.0,
        lower_bound_usd_m=14.0,
        upper_bound_usd_m=23.0,
        description="Machining, composite manufacture, tooling amortisation, facility costs.",
    ),
    CostCategory(
        name="Contingency & Risk Reserve",
        baseline_usd_m=10.0,
        lower_bound_usd_m=5.0,
        upper_bound_usd_m=18.0,
        description="Unplanned incidents, DNFs, repairs, supply chain disruption, "
                    "regulatory compliance costs. Right-skewed — modelled as PERT.",
        distribution="pert",
    ),
]


# ─── 2026 Cost Categories ────────────────────────────────────────────────────
# 2026 Technical Regulations introduce a new car concept requiring full redesign
# across aero, chassis, and power unit integration. Low TRL across the board
# means significantly wider uncertainty bounds than the mature 2024 programme.
#
# Uplift rationale (~$50M additional spend distributed across five categories):
#   Aerodynamic Development  +$16M — new concept, teams starting from scratch
#   Chassis & Bodywork       +$12M — full redesign, new structural requirements
#   Manufacturing & Tooling  +$10M — new tooling investment for new architecture
#   Power Unit Integration   +$8M  — new hybrid architecture, complex integration
#   Contingency & Risk       +$4M  — low TRL, unknown unknowns across all categories
#
# Uncertainty bounds are significantly wider than 2024 — particularly on
# Aero and Contingency — because no team has cost data for building to these regs.
# PERT distribution on Contingency amplifies the right tail appropriately for
# a low-TRL programme where upside cost risk is substantially higher than downside.

DEFAULT_CATEGORIES_2026 = [
    CostCategory(
        name="Aerodynamic Development",
        baseline_usd_m=44.0,
        lower_bound_usd_m=32.0,
        upper_bound_usd_m=60.0,
        description="New 2026 aero concept — teams rebuilding from scratch. "
                    "Significantly wider bounds reflect low TRL and absence of historical data. "
                    "Front-loaded spend expected as teams converge on new concept.",
    ),
    CostCategory(
        name="Chassis & Bodywork",
        baseline_usd_m=34.0,
        lower_bound_usd_m=26.0,
        upper_bound_usd_m=44.0,
        description="Full chassis redesign for 2026 regulations. "
                    "New structural requirements and crash test standards drive uplift.",
    ),
    CostCategory(
        name="Power Unit (Leased)",
        baseline_usd_m=23.0,
        lower_bound_usd_m=20.0,
        upper_bound_usd_m=27.0,
        description="New 2026 hybrid PU lease cost — increased due to more complex "
                    "architecture and higher integration requirements. "
                    "Semi-fixed, subject to supplier pricing negotiations.",
        distribution="normal",
    ),
    CostCategory(
        name="Transmission & Suspension",
        baseline_usd_m=13.0,
        lower_bound_usd_m=11.0,
        upper_bound_usd_m=17.0,
        description="Suspension geometry evolves under new regs but not a full redesign. "
                    "Modest uplift for revised geometry requirements.",
    ),
    CostCategory(
        name="Electronics & Software",
        baseline_usd_m=13.0,
        lower_bound_usd_m=11.0,
        upper_bound_usd_m=16.0,
        description="New control systems required for 2026 hybrid architecture. "
                    "Software development costs increase with new PU integration.",
    ),
    CostCategory(
        name="Race Operations",
        baseline_usd_m=39.0,
        lower_bound_usd_m=35.0,
        upper_bound_usd_m=45.0,
        description="Race calendar expected to remain at ~24 races. "
                    "Modest uplift for inflation and increased freight complexity.",
        distribution="normal",
    ),
    CostCategory(
        name="Manufacturing & Tooling",
        baseline_usd_m=28.0,
        lower_bound_usd_m=21.0,
        upper_bound_usd_m=37.0,
        description="Significant new tooling investment required for new car architecture. "
                    "Composite tooling, machining fixtures and jigs all need replacement.",
    ),
    CostCategory(
        name="Contingency & Risk Reserve",
        baseline_usd_m=14.0,
        lower_bound_usd_m=6.0,
        upper_bound_usd_m=28.0,
        description="Substantially wider bounds than 2024 — low TRL across all categories "
                    "means unknown unknowns are significantly higher. "
                    "Right-skewed PERT distribution reflects asymmetric upside cost risk "
                    "when building to regulations nobody has built to before.",
        distribution="pert",
    ),
]

# Convenience lookup — allows app to switch category sets by reg year
CATEGORIES_BY_YEAR = {
    "2024": DEFAULT_CATEGORIES,
    "2026": DEFAULT_CATEGORIES_2026,
}


# ─── Correlation Matrix ───────────────────────────────────────────────────────
#
# WHY CORRELATIONS MATTER:
# Without correlations, each category is sampled independently — as if
# overspending on aero has no relationship to overspending on manufacturing.
# In reality, categories share underlying cost drivers:
#   - Staff cost inflation affects everything simultaneously
#   - Supply chain disruption hits manufacturing, chassis, AND aero parts
#   - A major upgrade package drives aero, chassis, AND manufacturing together
#
# Independent sampling UNDERSTATES tail risk — the probability of everything
# going over simultaneously is artificially low.
#
# THE FIX — CHOLESKY DECOMPOSITION:
# 1. Define a correlation matrix R between categories (values 0-1)
# 2. Decompose R using Cholesky: R = L @ L.T
# 3. Generate independent standard normal samples Z
# 4. Apply L to Z: correlated_Z = Z @ L.T
# 5. Transform correlated_Z back to uniform via normal CDF
# 6. Use correlated uniforms to drive each category's distribution
#
# The result: correlated categories tend to be high or low together,
# better reflecting real-world cost behaviour.
#
# CORRELATION VALUES USED:
# 0.0  = no relationship (independent)
# 0.3  = weak positive correlation
# 0.5  = moderate positive correlation
# 0.7  = strong positive correlation
# 1.0  = perfect correlation (same driver)

# Category order must match DEFAULT_CATEGORIES order:
# [Aero Dev, Chassis, PU Leased, Trans/Susp, Electronics, Race Ops, Mfg/Tooling, Contingency]

CORRELATION_MATRIX = np.array([
    # Aero  Chas  PU    T&S   Elec  Race  Mfg   Cont
    [1.00, 0.60, 0.10, 0.40, 0.30, 0.20, 0.65, 0.25],  # Aerodynamic Development
    [0.60, 1.00, 0.10, 0.50, 0.25, 0.15, 0.70, 0.20],  # Chassis & Bodywork
    [0.10, 0.10, 1.00, 0.15, 0.20, 0.10, 0.10, 0.10],  # Power Unit (Leased) — supplier-driven
    [0.40, 0.50, 0.15, 1.00, 0.30, 0.20, 0.55, 0.20],  # Transmission & Suspension
    [0.30, 0.25, 0.20, 0.30, 1.00, 0.25, 0.30, 0.15],  # Electronics & Software
    [0.20, 0.15, 0.10, 0.20, 0.25, 1.00, 0.25, 0.35],  # Race Operations — logistics-driven
    [0.65, 0.70, 0.10, 0.55, 0.30, 0.25, 1.00, 0.25],  # Manufacturing & Tooling
    [0.25, 0.20, 0.10, 0.20, 0.15, 0.35, 0.25, 1.00],  # Contingency & Risk
])

# Key correlation rationale:
# Aero <-> Manufacturing (0.65): aero development drives parts manufacture directly
# Chassis <-> Manufacturing (0.70): composite manufacture is the main chassis cost driver
# Aero <-> Chassis (0.60): major upgrade packages affect both simultaneously
# PU Leased (low correlations): supplier contract — largely independent of in-house spend
# Race Ops <-> Contingency (0.35): logistics disruption drives both


def _validate_correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Ensure the correlation matrix is valid positive semi-definite.
    If not (can happen with manually specified values), apply nearest
    positive definite correction via eigenvalue clipping.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # Clip any small negative eigenvalues to near-zero
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    corrected = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    # Re-normalise diagonal to exactly 1.0
    d = np.sqrt(np.diag(corrected))
    corrected = corrected / np.outer(d, d)
    return corrected


def _apply_cholesky_correlation(
    independent_samples: dict,
    categories: list,
    corr_matrix: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """
    Apply Cholesky decomposition to introduce correlations between
    independently sampled cost categories.

    Method:
    1. Sample each category independently (already done)
    2. Rank-transform samples to uniform [0,1] via empirical CDF
    3. Transform to standard normal via inverse CDF
    4. Apply Cholesky factor to introduce correlation structure
    5. Transform back to uniform, then re-apply original marginal distribution

    This is the Iman-Conover method — preserves marginal distributions
    while introducing the desired correlation structure.
    """
    n = len(next(iter(independent_samples.values())))
    cat_names = [c.name for c in categories if c.fia_regulated]
    k = len(cat_names)

    # Step 1: Stack independent samples into matrix (n x k)
    X = np.column_stack([independent_samples[name] for name in cat_names])

    # Step 2: Validate and decompose correlation matrix
    valid_corr = _validate_correlation_matrix(corr_matrix)
    L = np.linalg.cholesky(valid_corr)

    # Step 3: Generate correlated standard normals
    Z = rng.standard_normal((n, k))
    Z_corr = Z @ L.T  # Apply Cholesky factor

    # Step 4: Get rank order from correlated normals
    ranks = np.argsort(np.argsort(Z_corr, axis=0), axis=0)

    # Step 5: Rearrange original samples to match correlated rank order
    correlated_samples = {}
    for i, name in enumerate(cat_names):
        sorted_original = np.sort(independent_samples[name])
        correlated_samples[name] = sorted_original[ranks[:, i]]

    return correlated_samples


# ─── Distribution Sampling ────────────────────────────────────────────────────

def _sample_category(cat: CostCategory, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n values from a cost category's marginal distribution."""
    if cat.distribution == "triangular":
        return rng.triangular(
            left=cat.lower_bound_usd_m,
            mode=cat.baseline_usd_m,
            right=cat.upper_bound_usd_m,
            size=n,
        )
    elif cat.distribution == "normal":
        sigma = (cat.upper_bound_usd_m - cat.lower_bound_usd_m) / (2 * 1.645)
        return rng.normal(loc=cat.baseline_usd_m, scale=sigma, size=n)
    elif cat.distribution == "pert":
        a = cat.lower_bound_usd_m
        b = cat.upper_bound_usd_m
        m = cat.baseline_usd_m
        mean = (a + 4 * m + b) / 6
        std = (b - a) / 6
        alpha = ((mean - a) / (b - a)) * ((mean - a) * (b - mean) / std**2 - 1)
        beta_param = alpha * (b - mean) / (mean - a)
        alpha = max(alpha, 0.5)
        beta_param = max(beta_param, 0.5)
        return a + (b - a) * rng.beta(alpha, beta_param, size=n)
    else:
        return rng.triangular(
            cat.lower_bound_usd_m, cat.baseline_usd_m, cat.upper_bound_usd_m, size=n
        )


# ─── Main Simulation ──────────────────────────────────────────────────────────

def run_simulation(
    categories: list = None,
    config: SimulationConfig = None,
    corr_matrix: np.ndarray = None,
) -> dict:
    """
    Run Monte Carlo cost cap risk simulation.

    Parameters
    ----------
    categories  : list of CostCategory (defaults to DEFAULT_CATEGORIES)
    config      : SimulationConfig (defaults to 10k sims, 2024 adjusted cap)
    corr_matrix : correlation matrix (defaults to CORRELATION_MATRIX)
                  Only used if config.apply_correlations = True

    Returns
    -------
    dict containing:
        total_spend         : np.ndarray (n_simulations,)
        category_samples    : dict[name -> np.ndarray]
        summary             : pd.DataFrame percentile table
        breach_prob         : float P(total > cap)
        minor_breach_prob   : float P(0 < excess < 5%)
        material_breach_prob: float P(excess >= 5%)
        expected_excess     : float E[max(total - cap, 0)]
        cost_cap            : float
        sensitivity         : dict[name -> pearson_r]
        correlated          : bool — whether correlations were applied
    """
    if categories is None:
        categories = DEFAULT_CATEGORIES
    if config is None:
        config = SimulationConfig()
    if corr_matrix is None:
        corr_matrix = CORRELATION_MATRIX

    rng = np.random.default_rng(config.random_seed)
    n = config.n_simulations
    cap = config.cost_cap_usd_m

    # Step 1: Sample each category independently (marginal distributions)
    cat_samples = {}
    for cat in categories:
        if cat.fia_regulated:
            cat_samples[cat.name] = _sample_category(cat, n, rng)

    # Step 2: Apply correlation structure if enabled
    if config.apply_correlations:
        cat_samples = _apply_cholesky_correlation(
            cat_samples, categories, corr_matrix, rng
        )

    # Step 3: Total season spend
    total = np.sum(list(cat_samples.values()), axis=0)

    # Step 4: Breach analysis
    excess = np.maximum(total - cap, 0)
    breach_mask = total > cap
    minor_mask = breach_mask & (excess < MINOR_BREACH_THRESHOLD * cap)
    material_mask = excess >= MATERIAL_BREACH_THRESHOLD * cap

    # Step 5: Percentile summary table
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    summary_rows = []
    for cat in categories:
        if cat.name in cat_samples:
            s = cat_samples[cat.name]
            row = {"Category": cat.name, "Baseline ($M)": cat.baseline_usd_m}
            for p in percentiles:
                row[f"P{p}"] = round(float(np.percentile(s, p)), 2)
            summary_rows.append(row)

    total_row = {
        "Category": "TOTAL SPEND",
        "Baseline ($M)": sum(c.baseline_usd_m for c in categories if c.fia_regulated),
    }
    for p in percentiles:
        total_row[f"P{p}"] = round(float(np.percentile(total, p)), 2)
    summary_rows.append(total_row)

    # Step 6: Sensitivity — Pearson correlation of each category with total
    sensitivity = {}
    for name, samples in cat_samples.items():
        corr = float(np.corrcoef(samples, total)[0, 1])
        sensitivity[name] = round(corr, 4)

    return {
        "total_spend": total,
        "category_samples": cat_samples,
        "summary": pd.DataFrame(summary_rows),
        "breach_prob": float(breach_mask.mean()),
        "minor_breach_prob": float(minor_mask.mean()),
        "material_breach_prob": float(material_mask.mean()),
        "expected_excess": float(excess.mean()),
        "cost_cap": cap,
        "sensitivity": sensitivity,
        "n_simulations": n,
        "correlated": config.apply_correlations,
    }


# ─── Scenario Comparison ─────────────────────────────────────────────────────

def scenario_comparison(scenarios: dict) -> pd.DataFrame:
    """Compare breach probabilities across named regulatory scenarios."""
    rows = []
    for label, (cats, cfg) in scenarios.items():
        res = run_simulation(cats, cfg)
        rows.append({
            "Scenario": label,
            "Cap ($M)": cfg.cost_cap_usd_m,
            "Total Baseline ($M)": sum(c.baseline_usd_m for c in cats if c.fia_regulated),
            "P50 Spend ($M)": round(float(np.percentile(res["total_spend"], 50)), 2),
            "P90 Spend ($M)": round(float(np.percentile(res["total_spend"], 90)), 2),
            "Breach Probability": f"{res['breach_prob']:.1%}",
            "Expected Excess ($M)": round(res["expected_excess"], 2),
        })
    return pd.DataFrame(rows)


# ─── Correlation Impact Analysis ─────────────────────────────────────────────

def correlation_impact_analysis() -> pd.DataFrame:
    """
    Compare results with and without correlation to demonstrate
    the impact of the Cholesky correlation method on tail risk.
    """
    independent = run_simulation(
        config=SimulationConfig(apply_correlations=False)
    )
    correlated = run_simulation(
        config=SimulationConfig(apply_correlations=True)
    )

    rows = []
    for label, res in [("Independent (original)", independent), ("Correlated (Cholesky)", correlated)]:
        rows.append({
            "Model": label,
            "P50 ($M)": round(float(np.percentile(res["total_spend"], 50)), 2),
            "P90 ($M)": round(float(np.percentile(res["total_spend"], 90)), 2),
            "P95 ($M)": round(float(np.percentile(res["total_spend"], 95)), 2),
            "Breach Prob": f"{res['breach_prob']:.1%}",
            "Material Breach": f"{res['material_breach_prob']:.1%}",
            "Expected Excess ($M)": round(res["expected_excess"], 2),
        })
    return pd.DataFrame(rows)


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("FIA Cost Cap Risk Simulator — Phase 2")
    print("=" * 55)

    # Main simulation — 2024 adjusted cap with correlations
    results = run_simulation()

    print(f"\nRegulatory Context:")
    print(f"  Baseline Cap:          ${COST_CAP_BASELINE:.0f}M")
    print(f"  2024 Adjusted Cap:     ${COST_CAP_2024_ADJUSTED:.0f}M  (24 races + inflation)")
    print(f"  2026 Baseline Cap:     ${COST_CAP_2026_BASELINE:.0f}M  (new technical regs)")
    print(f"  PU Manufacturer Cap:   ${PU_CAP_BASELINE:.0f}M   (separate, independent)")

    print(f"\nSimulation: {results['n_simulations']:,} runs | "
          f"Cap: ${results['cost_cap']:.0f}M | "
          f"Correlations: {'ON' if results['correlated'] else 'OFF'}")

    print(f"\nBreach Analysis:")
    print(f"  Breach Probability:    {results['breach_prob']:.1%}")
    print(f"  Minor Breach (<5%):    {results['minor_breach_prob']:.1%}")
    print(f"  Material Breach (≥5%): {results['material_breach_prob']:.1%}")
    print(f"  Expected Excess:       ${results['expected_excess']:.2f}M")

    print(f"\nSensitivity (top cost drivers):")
    for k, v in sorted(results["sensitivity"].items(), key=lambda x: -abs(x[1])):
        bar = "█" * int(abs(v) * 30)
        print(f"  {k:<35} {v:.3f}  {bar}")

    print(f"\nCorrelation Impact Analysis:")
    print(correlation_impact_analysis().to_string(index=False))

    print(f"\nScenario Comparison (2024 vs 2026 regs + cap):")
    scenarios = {
        "2024 Regs + Cap ($165M)": (DEFAULT_CATEGORIES,      SimulationConfig(cost_cap_usd_m=165.0, apply_correlations=True)),
        "2024 Regs — independent":  (DEFAULT_CATEGORIES,      SimulationConfig(cost_cap_usd_m=165.0, apply_correlations=False)),
        "2026 Regs + Cap ($215M)": (DEFAULT_CATEGORIES_2026, SimulationConfig(cost_cap_usd_m=215.0, apply_correlations=True)),
    }
    print(scenario_comparison(scenarios).to_string(index=False))

    print(f"\n2026 Regulations — Standalone Analysis:")
    results_2026 = run_simulation(
        categories=DEFAULT_CATEGORIES_2026,
        config=SimulationConfig(cost_cap_usd_m=215.0, apply_correlations=True)
    )
    baseline_2026 = sum(c.baseline_usd_m for c in DEFAULT_CATEGORIES_2026)
    print(f"  2026 Total Baseline:   ${baseline_2026:.0f}M")
    print(f"  2026 Cap:              ${COST_CAP_2026_BASELINE:.0f}M")
    print(f"  Headroom (baseline):   ${COST_CAP_2026_BASELINE - baseline_2026:.0f}M")
    print(f"  Breach Probability:    {results_2026['breach_prob']:.1%}")
    print(f"  P50 Spend:             ${np.percentile(results_2026['total_spend'], 50):.1f}M")
    print(f"  P90 Spend:             ${np.percentile(results_2026['total_spend'], 90):.1f}M")

    print(f"\n2024 Percentile Summary:")
    print(results["summary"].to_string(index=False))

    print(f"\n2026 Percentile Summary:")
    print(results_2026["summary"].to_string(index=False))
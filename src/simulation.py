"""
FIA Cost Cap Risk Simulator - Core Monte Carlo Engine
=====================================================
Models season-level spend risk against FIA Financial Regulations.
Identifies cost categories with highest breach probability.

Author: [Your Name]
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

# FIA Cost Cap 2025 (USD millions)
# Ref: FIA Financial Regulations Article 4
COST_CAP_2025 = 135.0

# Adjustments per constructor entry (above 2 constructors)
CONSTRUCTOR_ENTRY_ADJUSTMENT = 4.0

# Penalty thresholds (% over cap)
MINOR_BREACH_THRESHOLD = 0.05  # <5% over = minor breach
MATERIAL_BREACH_THRESHOLD = 0.05  # >=5% over = material breach


@dataclass
class CostCategory:
    """Represents a single cost category with uncertainty bounds."""
    name: str
    baseline_usd_m: float        # Central estimate
    lower_bound_usd_m: float     # Optimistic (e.g. 10th percentile)
    upper_bound_usd_m: float     # Pessimistic (e.g. 90th percentile)
    distribution: str = "triangular"  # triangular | normal | pert
    description: str = ""
    fia_regulated: bool = True   # Whether this falls under cost cap


@dataclass
class SimulationConfig:
    """Configuration for a Monte Carlo run."""
    n_simulations: int = 10_000
    cost_cap_usd_m: float = COST_CAP_2025
    random_seed: Optional[int] = 42
    # Correlation matrix toggle — phase 2 feature
    apply_correlations: bool = False


# --- DEFAULT COST CATEGORIES ---
# Based on publicly available FIA regulation structure
# and published team financial disclosures / analyst estimates.

DEFAULT_CATEGORIES = [
    CostCategory(
        name="Aerodynamic Development",
        baseline_usd_m=22.0,
        lower_bound_usd_m=18.0,
        upper_bound_usd_m=30.0,
        description="CFD, wind tunnel, aero parts manufacture",
    ),
    CostCategory(
        name="Chassis & Bodywork",
        baseline_usd_m=18.0,
        lower_bound_usd_m=15.0,
        upper_bound_usd_m=24.0,
        description="Carbon fibre structures, monocoque, bodywork panels",
    ),
    CostCategory(
        name="Power Unit (Leased)",
        baseline_usd_m=15.0,
        lower_bound_usd_m=14.0,
        upper_bound_usd_m=17.0,
        description="PU lease cost — semi-fixed, subject to supplier pricing",
        distribution="normal",
    ),
    CostCategory(
        name="Transmission & Suspension",
        baseline_usd_m=12.0,
        lower_bound_usd_m=10.0,
        upper_bound_usd_m=16.0,
        description="Gearbox, driveshafts, suspension geometry components",
    ),
    CostCategory(
        name="Electronics & Software",
        baseline_usd_m=10.0,
        lower_bound_usd_m=8.5,
        upper_bound_usd_m=13.0,
        description="Control electronics, simulation software licences",
    ),
    CostCategory(
        name="Race Operations",
        baseline_usd_m=28.0,
        lower_bound_usd_m=25.0,
        upper_bound_usd_m=34.0,
        description="Travel, freight, garage, trackside personnel & equipment",
        distribution="normal",
    ),
    CostCategory(
        name="Manufacturing & Tooling",
        baseline_usd_m=14.0,
        lower_bound_usd_m=11.0,
        upper_bound_usd_m=18.0,
        description="Machining, composite manufacture, tooling amortisation",
    ),
    CostCategory(
        name="Contingency & Risk Reserve",
        baseline_usd_m=8.0,
        lower_bound_usd_m=4.0,
        upper_bound_usd_m=14.0,
        description="Unplanned incidents, DNFs, repairs, supply chain risk",
        distribution="pert",
    ),
]


def _sample_category(cat: CostCategory, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n values from a cost category's distribution."""
    if cat.distribution == "triangular":
        return rng.triangular(
            left=cat.lower_bound_usd_m,
            mode=cat.baseline_usd_m,
            right=cat.upper_bound_usd_m,
            size=n,
        )
    elif cat.distribution == "normal":
        # Use 90th percentile spread to estimate sigma
        sigma = (cat.upper_bound_usd_m - cat.lower_bound_usd_m) / (2 * 1.645)
        return rng.normal(loc=cat.baseline_usd_m, scale=sigma, size=n)
    elif cat.distribution == "pert":
        # PERT via beta distribution
        a = cat.lower_bound_usd_m
        b = cat.upper_bound_usd_m
        m = cat.baseline_usd_m
        mean = (a + 4 * m + b) / 6
        std = (b - a) / 6
        alpha = ((mean - a) / (b - a)) * ((mean - a) * (b - mean) / std**2 - 1)
        beta = alpha * (b - mean) / (mean - a)
        alpha = max(alpha, 0.5)
        beta = max(beta, 0.5)
        return a + (b - a) * rng.beta(alpha, beta, size=n)
    else:
        return rng.triangular(
            cat.lower_bound_usd_m, cat.baseline_usd_m, cat.upper_bound_usd_m, size=n
        )


def run_simulation(
    categories: list[CostCategory] = None,
    config: SimulationConfig = None,
) -> dict:
    """
    Run Monte Carlo simulation and return results dict.

    Returns
    -------
    dict with keys:
        total_spend      : np.ndarray of shape (n_simulations,)
        category_samples : dict[name -> np.ndarray]
        summary          : pd.DataFrame — percentile table
        breach_prob      : float — P(total > cost_cap)
        minor_breach_prob: float — P(0 < excess < 5% cap)
        material_breach_prob: float
        expected_excess  : float — E[max(total - cap, 0)]
        cost_cap         : float
    """
    if categories is None:
        categories = DEFAULT_CATEGORIES
    if config is None:
        config = SimulationConfig()

    rng = np.random.default_rng(config.random_seed)
    n = config.n_simulations
    cap = config.cost_cap_usd_m

    # Sample each category
    cat_samples = {}
    for cat in categories:
        if cat.fia_regulated:
            cat_samples[cat.name] = _sample_category(cat, n, rng)

    # Total season spend
    total = np.sum(list(cat_samples.values()), axis=0)

    # Breach analysis
    excess = np.maximum(total - cap, 0)
    breach_mask = total > cap
    minor_mask = breach_mask & (excess < MINOR_BREACH_THRESHOLD * cap)
    material_mask = excess >= MINOR_BREACH_THRESHOLD * cap

    # Percentile summary
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    summary_rows = []
    for cat in categories:
        if cat.name in cat_samples:
            s = cat_samples[cat.name]
            row = {"Category": cat.name, "Baseline (£M)": cat.baseline_usd_m}
            for p in percentiles:
                row[f"P{p}"] = round(np.percentile(s, p), 2)
            summary_rows.append(row)

    # Add total row
    total_row = {"Category": "TOTAL SPEND", "Baseline (£M)": sum(c.baseline_usd_m for c in categories if c.fia_regulated)}
    for p in percentiles:
        total_row[f"P{p}"] = round(np.percentile(total, p), 2)
    summary_rows.append(total_row)

    summary_df = pd.DataFrame(summary_rows)

    # Sensitivity: Spearman correlation of each category with total
    sensitivity = {}
    for name, samples in cat_samples.items():
        corr = np.corrcoef(samples, total)[0, 1]
        sensitivity[name] = round(corr, 4)

    return {
        "total_spend": total,
        "category_samples": cat_samples,
        "summary": summary_df,
        "breach_prob": float(breach_mask.mean()),
        "minor_breach_prob": float(minor_mask.mean()),
        "material_breach_prob": float(material_mask.mean()),
        "expected_excess": float(excess.mean()),
        "cost_cap": cap,
        "sensitivity": sensitivity,
        "n_simulations": n,
    }


def scenario_comparison(scenarios: dict[str, list[CostCategory]]) -> pd.DataFrame:
    """
    Compare breach probabilities across named scenarios.

    Parameters
    ----------
    scenarios : dict mapping scenario label -> list of CostCategory

    Returns
    -------
    DataFrame with scenario-level breach stats
    """
    rows = []
    for label, cats in scenarios.items():
        res = run_simulation(cats)
        rows.append({
            "Scenario": label,
            "Total Baseline (£M)": sum(c.baseline_usd_m for c in cats if c.fia_regulated),
            "P50 Spend (£M)": round(np.percentile(res["total_spend"], 50), 2),
            "P90 Spend (£M)": round(np.percentile(res["total_spend"], 90), 2),
            "Breach Probability": f"{res['breach_prob']:.1%}",
            "Expected Excess (£M)": round(res["expected_excess"], 2),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Running FIA Cost Cap Monte Carlo Simulation")
    print("=" * 50)
    results = run_simulation()
    print(f"\nCost Cap:           ${results['cost_cap']:.1f}M")
    print(f"Breach Probability: {results['breach_prob']:.1%}")
    print(f"Minor Breach:       {results['minor_breach_prob']:.1%}")
    print(f"Material Breach:    {results['material_breach_prob']:.1%}")
    print(f"Expected Excess:    ${results['expected_excess']:.2f}M")
    print("\nSensitivity (correlation with total spend):")
    for k, v in sorted(results["sensitivity"].items(), key=lambda x: -abs(x[1])):
        print(f"  {k:<35} {v:.3f}")
    print("\nSummary Table:")
    print(results["summary"].to_string(index=False))
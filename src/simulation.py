    """
    FIA Cost Cap Risk Simulation - Core Monte Carlo Engine
    ===============================================
    Models season-level spend risk against FIA Financial Regulations. Identifies cost categories with highest breach probability.
      
    
    """
    
    import numpy as np
    import pandas as pd
    from dataclasses import dataclass, field
    from typing import Optional
    import warnings
    warnings.filterwarnings("ignore")
    
    # FIA cost cap 2025 (USD millions)
    # Ref: FIA Financial Regulations 2025, Article 4
    FIA_COST_CAP_2025 = 135.0
    
    # Adjustmetns per constructor entry (above 2 constructors)
    CONSTRUCTOR_ENTRY_ADJUSTMENT = 4.0
    
    # Penalty hresholds (% over cap)
    MINOR_BREACH_THRESHOLD = 0.05  # 5% over cap
    MATERIAL_BREACH_THRESHOLD = 0.05  # >=5% over = Material Breach
    
    @dataclass
    class SimulationConfig:
        ''' Configuration for Monte Carlo Run'''
        n_simulations: int = 10000
        cost_cap_usd_m = FIA_COST_CAP_2025
        random_seed: Optional[int] = 42
        # Correlation matrix toggle
        apply_correlation: bool = False
        

    # Default cost categories 
    # Based on publicy  available data and FIA regulations
    
    DEFAULT_COST_CATEGORIES = [
        CostCategory(
            name='Aerodynamics Dev',
            baseline_usd_m = 22.0,
            lower_bound_usd_m = 18.0,
            upper_bound_usd_m = 30.0,
            desc= 'CFD, wind tunnel, aero parts manufacturing',
        ),
        CostCategory(
            name='Chassis & Bodywork',
            baseline_usd_m = 18.0,
            lower_bound_usd_m = 15.0,
            upper_bound_usd_m = 24.0,
            desc= 'Carbon Fibre Structures, monocoque, body work pannels',
        ),    
       CostCategory(
               name='Power Unit (Leased)',
                baseline_usd_m = 15.0,
                lower_bound_usd_m = 14.0,
                upper_bound_usd_m = 17.0,
                desc= 'PU lease costs - semi-fixed, subject to supplier priciing and usage',
                distrubution = 'normal'
            ),
            CostCategory(
                name='Transmission & Suspension',
                baseline_usd_m = 12.0,
                lower_bound_usd_m = 10.0,
                upper_bound_usd_m = 16.0,
                desc= 'Gearbox, driveshafts, suspension geometry components',
            ),
            CostCategory(
                name='Electronics & Software',
                baseline_usd_m = 10.0,
                lower_bound_usd_m = 8.5,
                upper_bound_usd_m = 13.0,
                desc= 'ECUs, telemetry, control software development',
            ),
            CostCategory(
                name='Race Operations & Logistics',
                baseline_usd_m = 28.0,
                lower_bound_usd_m = 25.0,
                upper_bound_usd_m = 34.0,
                desc= 'Travel, freight, garage, trackside personnel, hospitality',
                distrubution = 'normal'
            ),
            CostCategory(
                name='Manufacturing & Tooling',
                baseline_usd_m = 14.0,
                lower_bound_usd_m = 11.0,
                upper_bound_usd_m = 18.0,
                desc= 'Machining, composite manufacturing, tooling amortisation'
            ),
            CostCatgeory(
                name='Contingency & Risk Reserve',
                basline_usd_m = 8.0,
                lower_bound_usd_m = 4.0,
                upper_bound_usd_m = 14.0,
                desc= 'Budget reserve for unforeseen costs, design changes, supplier issues',
                distrubution = 'pert'
            ),             
    ]    
        
    def _sample_category(cat: CostCategory, n: int, rng: np.random.Generator) -> np.ndarray:
        ''' Sample n values from a cost category distribution'''
        if cat.distrubution == 'triangular':
            return rng.triangular(
                left = cat.lower_bound_usd_m,
                mode= cat.baseline_usd_m,
                right = cat.upper_bound_usd_m,
                size = n,
            )
        elif cat.distrubution == 'normal':
            # Use 90th percentile as upper bound to avoid extreme outliers
            sigma = (cat.upper_bound_usd_m - cat.lower_bound_usd_m) / (2*1.645)
            return rng.normal(loc= cat.baseline_usd_m, scale=sigma, size=n)
        elif cat.distrubution == 'pert':
            # PERT via beta distribution parameters
            a = cat.lower_bound_usd_m
            b = cat.upper_bound_usd_m
            m = cat.baseline_usd_m
            mean = (a + 4*m + b) / 6
            std = (b - a) / 6
            alpha = ((mean - a) * (2*b - a - mean)) / ((b - a) * std**2) - 1
            beta = alpha * (b - mean) / (mean - a)
            alpha = max(alpha, 0.05)
            beta = max(beta, 0.05)
            return a + (b - a) * rng.beta(alpha, beta, size=n)
        else:
            return rng.trigangular(
                cat.lower_bound_usd_m, cat.baseline_usd_m, cat.upper_bound_usd_m, size=n)
            
    def run_simulation(
        categories: list[CostCategory] = None,
        config: SimulationConfig = None,
        ) -> dict:

        ''' Run Monte Carlo Simulation for FIA Cost Cap Risk Analysis and returns dict.
        Returns: 
        -------
        dict with keys:
            total_spend: np.ndarray of shape (n_sumulations,) with total simulated spend per iteration
            category_samples : dict[name -> np.ndarray]
            summary : pd.DataFrame - percentile table
            breach_prob : float - P(total > cost_cap)
            minor_breach_prob : float - P(0 < excess < 5% cap)
            material_breach_prob : float
            expected_excess : float - E[max(total - cost_cap, 0 )]
            cost_cap : float - cost cap used in simulation
            
        '''
        if categories is None:
            categories = DEFAULT_COST_CATEGORIES
        if config is None:
            config = SimulationConfig()
        
        rng = np.random.default_rng(config.random_seed)
        n = config.n_simulations
        cap = config.cost_cap_usd_m
        
        # Sample each category
        category_samples = {}
        for cat in categories:
            if cat.fia_regulated:
                cat_samples[cat.name] = _sample_category(cat, n, rng)
        
        # total season spend
        total = np.sum(list(cat_samples.values()), axis=0)
        
        # Breach analysis
        excess = np.maximum(total - cap, 0)
        breach_mask = total > cap 
        minor_mask = breach_mask & (excess < MINOR_BREACH_THRESHOLD * cap)
        material_mask = excess >= MATERIAL_BREACH_THRESHOLD * cap
        
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        summary_rows = []
        for cat in categories:
            if cat.name in cat_samples:
                s = cat_samples[cat.name]
                row = {
                    'Category': cat.name,
                    'Bseline ($M)': cat.baseline_usd_m,
                }
            for p in percentiles:
                row[f'{p}th Pctl'] = round(np.percentile(s, p),2) 
            summary_rows.append(row)
            
        # Add total row
        total_row = { 
                    'Category': 'Total Spend','Baseline ($M)': sum(c.baseline_usd_m for c in categories if c.fia_regulated)}
        for p in percentiles:
            total_row[f'P{p}'] = round(np.percentile(total, p),2)
        summary_rows.append(total_row)
        
        summary_df = pd.DataFrame(summary_rows)
        
        # Sensitivity analysis - correlation matrix (if enabled)
        
        return {
            'total_spend': total,
            'category_samples': cat_samples,
            'summary': summary_df,
            'breach_prob': float(breach_mask.mean()),
            'minor_breach_prob': float(minor_mask.mean()),
            'material_breach_prob': float(material_mask.mean()),
            'expected_excess': float(excess.mean()),
            'cost_cap': cap,
            'sensitivity': sensitivity,
            'n_simulations': n,
        }
        
    def scenario_comparison(scenarios: dict[str,list[CostCategory]]) -> pd.DataFrame:
        ''' 
        Compare breach probabilities across named scenarios.
        Parameters:
        -----------
        scenarios: dict mapping scenario labal -> list of CostCatgegory
        
        Returns:
        --------
        DataFrame with scenario-level breach stats        
        '''
        
        rows = []
        for label, cats in scenarios.items():
            res = run_simulation(cats)
            rows.append({
                'Scenario': label,
                'Total Baseline ($M)': sum(c.baseline_usd_m for c in cats if c.fia_regulated),
                'P50 Spend ($M)': round(np.percentile(res['total_spend'], 50),2),
                'P90 Spend ($M)': round(np.percentile(res['total_spend'], 90),2),
                'Breach Prob': f'{res["breach_prob"]:.2%}',
                'Expected Excess ($M)': round(res['expected_excess'],2),
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
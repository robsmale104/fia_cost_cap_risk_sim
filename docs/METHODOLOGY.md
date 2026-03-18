Methodology Notes
Distribution Assumptions
Triangular Distribution
Used for most cost categories. Requires three parameters: minimum, mode (most likely), maximum.

Appropriate when engineering judgement can bound the cost range and identify a central estimate
Directly analogous to three-point estimating in programme cost engineering (ACEIT, PRICE H)
Implemented via numpy.random.Generator.triangular

Normal Distribution
Used for quasi-fixed costs (Power Unit lease, Race Operations logistics).

Sigma estimated from P10–P90 spread: σ = (P90 - P10) / (2 × 1.645)
Appropriate when costs are driven by contractual/market rates with symmetric uncertainty

PERT Distribution
Used for Contingency & Risk Reserve.

PERT is a special case of the Beta distribution, parameterised by (min, mode, max)
Mean = (a + 4m + b) / 6 — weights the mode more heavily than triangular
Produces a smoother, more realistic right tail for "unknown unknown" cost items
Standard in programme risk analysis (MoD CADMID, NASA NPR 7120.5)

Sensitivity Analysis
Pearson correlation coefficient r(category_i, total_spend) across all simulations.
This is a simplified form of importance measure. Full variance based sensitivity (Sobol indices)
is planned as a Phase 2 extension, this would decompose total variance by first-order and
interaction effects across categories.
FIA Regulation Notes

Cost cap applies to "Relevant Costs" as defined in Article 4.1
Excluded costs include: driver salaries (top 3), PU development, marketing, HQ building costs
Minor breach: ≤5% over cap, financial penalty + reprimand
Material breach: >5% over cap, can include points deduction (precedent: Red Bull 2021)
Adjustments available for constructors entering >2 cars, and for inflation indexing

Planned Phase 2 Extensions

Correlation matrix: model inter-category correlations using Cholesky decomposition
Competitor modelling: estimate grid-wide spend distribution from public disclosures
FIA penalty model: translate breach probability into expected championship points impact
ML integration: gradient boosted model for budget allocation optimisation
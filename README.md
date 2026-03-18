# 🏎️ FIA Cost Cap Risk Simulator

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B)](https://fiacostcaprisksim-acpqsuw4qz5gvmfafulcgz.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)

> Monte Carlo simulation modelling season-level spend risk against the FIA Financial Regulations.  
> Identifies which cost categories carry the highest cost cap breach probability.  
> Built to demonstrate cost engineering and quantitative risk analysis methods applied to Formula One.

**[→ Open Live Dashboard](https://fiacostcaprisksim-acpqsuw4qz5gvmfafulcgz.streamlit.app/)**

---

## Overview

Since the introduction of the FIA cost cap in 2021, budget management has become one of the most strategically critical disciplines in Formula One. Teams that breach the cap face sporting and financial penalties, a minor breach (<5% over) triggers a reprimand and fine; a material breach (>5%) can result in constructor championship point deductions.

This project applies Monte Carlo simulation to model the probabilistic risk of a team breaching the cost cap across a full season, given uncertainty in each cost category. It is built to demonstrate core cost engineering and risk analysis skills relevant to high performance engineering environments.

---

## Key Features

- **Monte Carlo engine** — 10,000 simulation runs sampling from Triangular, Normal, and PERT distributions per cost category
- **Cholesky correlation matrix** — categories sampled with realistic correlations rather than independently, correctly capturing tail risk
- **Breach probability analysis** — overall, minor breach (<5%), and material breach (>5%) probabilities
- **Correlation impact analysis** — live comparison showing why independent sampling understates breach risk
- **Sensitivity / tornado analysis** — Pearson correlation identifying top cost drivers
- **Regulatory scenario switching** — 2025 adjusted cap ($165M), baseline ($135M), and 2026 new regulations ($215M)
- **2026 cost distributions** — separate category set with wider bounds reflecting lower TRL across new regulation categories
- **Regulatory context tab** — breach history, cap structure, FIA exclusions, 2026 changes

---

## Regulatory Context

| Regulation | Value | Notes |
|---|---|---|
| Team Cap — Baseline | $135M | Article 4, 2021–2025 |
| Team Cap — 2025 Adjusted | $165M | 24-race calendar + inflation indexation |
| Team Cap — 2026 Baseline | $215M | New technical regulations |
| PU Manufacturer Cap (2023–2025) | $95M | Independent cap — manufacturers only |
| PU Manufacturer Cap (2026+) | $130M | New PU regulations |

### Breach History

| Year | Team | Type | Penalty |
|---|---|---|---|
| 2021 | Red Bull Racing | Material Breach | 10 pts + $7M fine |
| 2023 | Honda Racing Corp | Procedural (ABA) | $600k fine |
| 2023 | Alpine Racing SAS | Procedural (ABA) | $400k fine |
| 2024 | Aston Martin | Minor Procedural | No financial penalty |

---

## Cost Categories Modelled

### 2025 Regulations — $157M total baseline, $8M headroom

| Category | Baseline ($M) | Distribution |
|---|---|---|
| Race Operations | 38.0 | Normal |
| Aerodynamic Development | 28.0 | Triangular |
| Chassis & Bodywork | 22.0 | Triangular |
| Manufacturing & Tooling | 18.0 | Triangular |
| Power Unit (Leased) | 15.0 | Normal |
| Transmission & Suspension | 14.0 | Triangular |
| Electronics & Software | 12.0 | Triangular |
| Contingency & Risk | 10.0 | PERT |

### 2026 Regulations — $208M total baseline, $7M headroom vs $215M cap

Despite the $80M cap increase, projected development costs consume most of the headroom. The $50M uplift is distributed across five categories reflecting the new technical regulations:

| Category | Uplift | Rationale |
|---|---|---|
| Aerodynamic Development | +$16M | New concept — teams rebuilding from scratch |
| Chassis & Bodywork | +$12M | Full redesign for new regulations |
| Manufacturing & Tooling | +$10M | New tooling investment |
| Power Unit (Leased) | +$8M | New hybrid architecture |
| Contingency & Risk | +$4M | Lower TRL across all categories |

---

## Phase 2 — Cholesky Correlation Matrix

The key analytical advancement in Phase 2 is introducing a Cholesky decomposition correlation matrix between cost categories.

Without correlations, each category is sampled independently, understating the probability of everything going wrong simultaneously. In reality, shared drivers (staff cost inflation, supply chain disruption, major upgrade packages) cause categories to move together.

### The finding

| Model | P90 Spend | Breach Probability |
|---|---|---|
| Independent (original) | $168.7M | 24.9% |
| Correlated (Cholesky) | $174.3M | 34.1% |

A 9 percentage point increase in breach probability from introducing realistic correlations — the number a cost cap compliance manager should plan against.

---

## Quickstart

```bash
git clone https://github.com/robsmaale104/f1-cost-cap-simulator.git
cd f1-cost-cap-simulator
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Structure

```
f1-cost-cap-simulator/
├── src/
│   ├── simulation.py       # Monte Carlo engine
│   └── dashboard.jsx       # React browser demo
├── docs/
│   └── methodology.md      # Distribution assumptions and methodology
├── app.py                  # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Planned Extensions

- [ ] Sobol variance based sensitivity indices
- [ ] Competitor spend modelling from public FIA submissions
- [ ] FIA penalty scenario analysis championship points impact
- [ ] ML layer — gradient boosted model for optimal budget allocation
- [ ] FastF1 data calibration against real FIA submission data

---

## Skills Demonstrated

| Skill | Implementation |
|---|---|
| Monte Carlo simulation | Triangular, Normal, PERT sampling |
| Cholesky correlation | Iman-Conover method |
| Cost driver analysis | Pearson sensitivity / tornado chart |
| Scenario planning | Conservative / Baseline / Aggressive / 2026 |
| Regulatory analysis | FIA Financial Regulations breach history |
| Python | Pandas, NumPy, Plotly, Streamlit |

---

## Companion Project

Sits alongside the [F1 Season Resource Allocation Wargame](https://github.com/[username]/f1-wargame-simulator) campaign-level simulation of development token allocation with adversarial competitor modelling. Together they cover the two core F1 analytical problems: staying under the cap and allocating resources optimally within it.

---

*FIA Financial Regulations 2025/2026 · Figures are illustrative estimates · Built by Rob Smale*
# BDA Group Project — NYPD Stop, Question & Frisk 2024

## Project Overview
Bayesian logistic regression analysis of NYPD SQF 2024 data.
**Research Question**: Do suspect characteristics, stop type, and perceived crime increase the probability of arrest? Is there systematic variation across precincts and boroughs?

## File Structure
```
bda_sqf_project/
├── code/
│   ├── 01_preprocess.py       # Data cleaning, EDA plots
│   ├── 02_jags_models.py      # 3 JAGS models + diagnostics + DIC + LOO-CV
│   └── 03_prior_checks.py     # Prior predictive checks + sensitivity analysis
├── outputs/
│   ├── plots/                 # All figures (11 PNG files)
│   ├── sqf_clean.csv          # Cleaned dataset (N=21,643)
│   ├── data_info.json         # Dataset summary statistics
│   ├── convergence_model2.csv # R-hat + ESS diagnostics
│   ├── model[1-3]_posterior_summary.csv  # Posterior means, SDs, 95% CrI
│   ├── extra_params.json      # sigma_u, sigma_v, rho estimates
│   ├── loo_results.json       # LOO-CV ELPD for all 3 models
│   └── prior_sensitivity.csv  # Sensitivity analysis results
└── README.md
```

## How to Run
```bash
pip install pandas numpy matplotlib openpyxl scipy
# Place sqf-2024.xlsx in the working directory
python 01_preprocess.py     # → sqf_clean.csv, data_info.json, plots/eda_overview.png
python 02_jags_models.py    # → all model outputs, plots (requires JAGS + pyjags for full MCMC)
python 03_prior_checks.py   # → plots/prior_predictive_check.png, prior_sensitivity.png
```
For full MCMC (recommended):
1. Install JAGS: https://sourceforge.net/projects/mcmc-jags/
2. pip install pyjags

## Key Results
- **N = 21,643** stops after preprocessing; **78 precincts**; overall arrest rate = **30.1%**
- **Model 2** (hierarchical with precinct random intercepts) is preferred by both DIC and LOO-CV
- Strongest predictors of arrest: firearm found (OR≈26), searched (OR≈13), weapon found (OR≈2.8)
- Significant precinct-level variation in arrest probability (σ_u ≈ 0.05)
- Race effects: White suspects have lower arrest odds than Black suspects (reference)
- Seasonal effects: Spring/Summer associated with modestly lower arrest rates vs Autumn

## Models
- **M1**: Logistic regression, fixed effects only  
  `logit(p_i) = α + X_i β`
- **M2**: Hierarchical, precinct random intercepts  
  `logit(p_i) = α + X_i β + u_j[i]`, `u_j ~ N(0, σ_u²)`
- **M3**: Hierarchical, random intercept + random weapon slope  
  `logit(p_i) = α + X_i β + u_j[i] + v_j[i] · weapon_i`

## Priors
All β ~ N(0, 2.5²) — weakly informative (Gelman et al. 2008 recommendation)
σ_u ~ Half-Normal(0,1) — regularises extreme precinct effects

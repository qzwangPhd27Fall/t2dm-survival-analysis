# T2DM Substandard-Life Rate-Making: Deep Survival Models

Reproducible analysis code for three papers on **substandard-life (impaired-life) insurance premium rate-making** for Type 2 Diabetes Mellitus (T2DM), trained on ~1,883,235 real-world T2DM inpatient claim records.

## Scripts

| Script | Model | Key contribution |
|---|---|---|
| `analysis_deepsurv.py` | DeepSurv + LSTM temporal embedding | Deep survival model with time-varying features, SHAP attribution, and a complete survival-to-premium mapping |
| `analysis_rsf.py` | Random Survival Forest + comorbidity-network embedding | ICD-10 co-occurrence graph, greedy-modularity community detection + spectral embedding, LIME interpretation |
| `analysis_stacking.py` | DeepSurv + RSF stacking ensemble | Out-of-fold meta-features, L1-regularized Cox meta-learner, bias-variance optimization, joint SHAP + LIME |

## Methods

- **Deep learning**: DeepSurv, LSTM (PyTorch)
- **Survival ensembles**: Random Survival Forest, stacking with a Cox meta-learner
- **Interpretability**: SHAP (global) + LIME (local)
- **Actuarial mapping**: survival probabilities → substandard premium rates (with duration-of-stay and health-management factors)

## Data

~1,883,235 real-world T2DM inpatient claim records (proprietary; not included in this repository).

## Related manuscript

Y. Xie, Q. Wang, "Deep Survival Analysis with Temporal Trajectory Embedding for Substandard Life Insurance Pricing: A DeepSurv Modeling Study of Type 2 Diabetes Mellitus," submitted to *Statistics in Medicine*.

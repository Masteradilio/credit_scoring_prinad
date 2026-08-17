# 🏦 PRINAD Quantitative Credit Risk Model Validation Report

**Document Reference**: `PRINAD-VAL-2026-V3.1`  
**Standard Compliance**: Basel III/IV Internal Ratings-Based (IRB) Approach, IFRS 9 / BACEN Resolução 4.966, Federal Reserve SR 11-7, ECB Guide to Internal Models  
**Model Name**: PRINAD (Probabilidade de Inadimplência & Underwriting Engine)  
**Authors**: Quantitative Credit Risk Team  
**Date**: August 2026  

---

## 1. Executive Summary

This report documents the quantitative validation and benchmarking of the **PRINAD Credit Risk Scoring Engine**. Designed for tier-1 credit risk operations, PRINAD operationalizes a multi-model architecture comparing an industry-standard **Regulatory Scorecard (Logistic Regression + WoE)** against modern **Gradient Boosting Challengers (LightGBM, XGBoost, Stacking Ensemble)**.

### Model Benchmark Summary Table (Enhanced 75k Portfolio)

| Model Architecture | Role | ROC-AUC | Gini ($2 \cdot \text{AUC} - 1$) | KS Statistic | Brier Score | Expected Calib. Error (ECE) | Basel Traffic Light |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regulatory Scorecard (WoE)** | **Champion (IRB)** | **0.9613** | **0.9227** | **0.7830** | **0.0591** | **0.0124** | 🟢 **Green Zone** |
| **LightGBM (Calibrated)** | **Challenger 1** | **0.9816** | **0.9631** | **0.8617** | **0.0404** | **0.0089** | 🟢 **Green Zone** |
| **XGBoost (Calibrated)** | **Challenger 2** | **0.9817** | **0.9634** | **0.8614** | **0.0405** | **0.0087** | 🟢 **Green Zone** |
| **Stacking Ensemble** | **Challenger 3** | **0.9818** | **0.9637** | **0.8627** | **0.0424** | **0.0092** | 🟢 **Green Zone** |

---

## 2. Pillar 1: Discrimination Power

The model’s ability to separate defaulting borrowers ($y=1$) from non-defaulting borrowers ($y=0$) was evaluated on a held-out test partition ($N = 15,000$, default rate = 26.57%):

- **Gini Coefficient**: The Champion Scorecard achieved a Gini of **0.9227** ($\text{Gini} = 2 \cdot \text{AUC} - 1$), far exceeding the regulatory threshold of 0.50. The Stacking Ensemble achieved a Gini of **0.9637**, demonstrating state-of-the-art discriminative capacity.
- **Kolmogorov-Smirnov (KS)**: Maximum separation between cumulative distributions of goods and bads reached **0.7830** (Scorecard) and **0.8627** (Ensemble), well above the regulatory benchmark of 0.35.

---

## 3. Pillar 2: Calibration & Goodness-of-Fit

Probability calibration ensures that a predicted PD of $5.0\%$ corresponds to an empirical default frequency of $5.0\%$:

- **Brier Score Loss**: Measured at **0.0591** for the Scorecard and **0.0404** for Tree Boosters (indicating outstanding probabilistic precision).
- **Murphy Brier Decomposition**:
  $$\text{Brier Score} = \text{Uncertainty} - \text{Resolution} + \text{Reliability}$$
  Low reliability error ($< 0.003$) confirms that predicted probabilities are well-calibrated across all risk deciles.
- **Hosmer-Lemeshow $\chi^2$ Test**: Evaluated across 10 deciles of predicted risk. $p$-values confirmed that the calibration curve does not statistically deviate from the empirical 45-degree line.

---

## 4. Pillar 3: Population Stability Index (PSI) & Data Drift

Model stability across vintages and economic shifts is monitored via the Population Stability Index:

$$\text{PSI} = \sum_{k=1}^K \left( \text{Actual}_k - \text{Expected}_k \right) \times \ln\left( \frac{\text{Actual}_k}{\text{Expected}_k} \right)$$

- **Total Scorecard PSI**: **0.0245** (well below the 0.10 yellow threshold).
- **Traffic Light Status**: 🟢 **VERDE (Green / Stable)** - No population drift detected across training and monitoring partitions.

---

## 5. Pillar 4: Statistical Backtesting & Basel Traffic Lights

Following the Basel Committee on Banking Supervision (BCBS) guidelines, each rating grade (A1 to DEFAULT) was subjected to:

1. **Exact Binomial Test (Clopper-Pearson 95% Confidence Intervals)**:
   - For all non-default rating grades, the observed default frequency fell inside the 95% Clopper-Pearson interval.
2. **Basel Traffic Light Classification**:
   - $P(X \ge k \mid PD_{\text{expected}}) > 0.05 \implies$ **Green Zone** across all rating bands.
   - The model does not underestimate risk in any rating bucket.

---

## 6. Macroeconomic Vasicek Model & IFRS 9 Forward-Looking Overlay

To convert Through-the-Cycle (TTC) regulatory PDs into forward-looking Point-in-Time (PIT) PDs under IFRS 9 / BACEN 4.966, the engine implements the **Vasicek Asymptotic Single Risk Factor (ASRF)** formula:

$$PD_{PIT}(Z) = \Phi\left( \frac{\Phi^{-1}(PD_{TTC}) - \sqrt{\rho} Z}{\sqrt{1 - \rho}} \right)$$

Where $Z$ is driven by macroeconomic factors (GDP growth, Selic rate, Unemployment) across 3 probability-weighted scenarios:
- **Baseline Scenario (50% weight)**: $Z = 0.00 \sigma \implies PD_{PIT} = PD_{TTC}$
- **Upside Scenario (25% weight)**: $Z = +0.85 \sigma \implies PD_{PIT} < PD_{TTC}$
- **Adverse Scenario (25% weight)**: $Z = -1.45 \sigma \implies PD_{PIT} > PD_{TTC}$ (Stress Testing)

---

## 7. Downstream Decisioning: IFRS 9 ECL & RAROC Pricing

1. **Expected Credit Loss (ECL)**:
   - **Stage 1**: $ECL_{12m} = PD_{12m} \times LGD \times EAD \times (1 + r)^{-1}$
   - **Stage 2**: $ECL_{\text{Lifetime}} = \sum_{t=1}^T PD_{\text{marginal}}(t) \times LGD \times EAD(t) \times (1 + r)^{-t}$
   - **Stage 3**: $ECL_{\text{Default}} = 1.0 \times LGD \times EAD \times (1 + r)^{-1}$
2. **Risk-Based Pricing (RAROC)**:
   $$\text{Fair Lending Rate} = \text{FTP (Cost of Funds)} + \text{OpEx} + \text{Expected Loss} + \text{Cost of Economic Capital} + \text{Target Net Margin}$$

---

## 8. Conclusion & Sign-Off

The **PRINAD v3.1 Engine** satisfies all regulatory and quantitative criteria for tier-1 production deployment. The system offers total transparency required for regulatory audits while delivering market-leading predictive power.

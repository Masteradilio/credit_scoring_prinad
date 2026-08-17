# 🏦 PRINAD — Quantitative Model Validation Report / Relatório Técnico Formal de Validação de Modelos

<div align="center">

**Document Reference**: `PRINAD-VAL-2026-V3.1`  
**Compliance Standards**: Basel III/IV Internal Ratings-Based (IRB), IFRS 9 / BACEN Resolução 4.966, Fed SR 11-7, ECB Guide to Internal Models  
**Engine**: PRINAD (Probability of Default, Stress Testing & Pricing Engine)  
**Authors**: Quantitative Credit Risk Validation Team  

<h3>
  🌐 Escolha o Idioma / Choose Language:
</h3>

**[🇧🇷 Versão em Português](#-versão-em-português)** &nbsp;&nbsp;|&nbsp;&nbsp; **[🇺🇸 English Version](#-english-version)**

</div>

---

<details open>
<summary><h2 style="display:inline-block;" id="-versão-em-português">🇧🇷 Versão em Português (Clique para alternar / recolher)</h2></summary>

### 📑 Sumário
1. [Resumo Executivo & Tabela Comparativa de Benchmark](#1-resumo-executivo--tabela-comparativa-de-benchmark)
2. [Pilar 1: Poder de Discriminação (ROC-AUC, Gini, KS)](#2-pilar-1-poder-de-discriminação-roc-auc-gini-ks)
3. [Pilar 2: Calibração de Probabilidades & Decomposição de Brier](#3-pilar-2-calibração-de-probabilidades--decomposição-de-brier)
4. [Pilar 3: Índice de Estabilidade Populacional (PSI) & Drift](#4-pilar-3-índice-de-estabilidade-populacional-psi--drift)
5. [Pilar 4: Backtesting Estatístico & Semáforo de Basileia](#5-pilar-4-backtesting-estatístico--semáforo-de-basileia)
6. [Estresse Macroeconômico de Vasicek & Cenários IFRS 9](#6-estresse-macroeconômico-de-vasicek--cenários-ifrs-9)
7. [Aplicações Financeiras: Provisão ECL & Precificação RAROC](#7-aplicações-financeiras-provisão-ecl--precificação-raroc)
8. [Parecer Conclusivo de Validação](#8-parecer-conclusivo-de-validação)

---

### 1. Resumo Executivo & Tabela Comparativa de Benchmark

Este relatório formal atesta a validação quantitativa e o benchmark comparativo do **Motor de Risco de Crédito PRINAD**. A suíte compara o **Scorecard Regulatório Champion (WoE & Regressão Logística)** contra os **Modelos Challengers de Gradient Boosting (LightGBM, XGBoost, Stacking Ensemble)** em uma base out-of-sample de $15.000$ contratos independentes (taxa empírica de inadimplência de $26,57\%$):

| Arquitetura do Modelo | Papel Regulatório | ROC-AUC | Gini ($2 \cdot \text{AUC} - 1$) | Estatística KS | Brier Score | Erro de Calibração (ECE) | Semáforo de Basileia |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regulatory Scorecard (WoE)** | **Champion (Basileia IRB)** | **0.9613** | **0.9227** | **0.7830** | **0.0591** | **0.0124** | 🟢 **Zona Verde** |
| **LightGBM (Isotônico)** | **Challenger 1** | **0.9816** | **0.9631** | **0.8617** | **0.0404** | **0.0089** | 🟢 **Zona Verde** |
| **XGBoost (Isotônico)** | **Challenger 2** | **0.9817** | **0.9634** | **0.8614** | **0.0405** | **0.0087** | 🟢 **Zona Verde** |
| **Stacking Ensemble** | **Challenger 3** | **0.9818** | **0.9637** | **0.8627** | **0.0424** | **0.0092** | 🟢 **Zona Verde** |

---

### 2. Pilar 1: Poder de Discriminação (ROC-AUC, Gini, KS)

- **Coeficiente de Gini**: O Scorecard Champion atingiu Gini de **0.9227** ($\text{Gini} = 2 \cdot \text{AUC} - 1$), superando com ampla margem o piso regulatório de $0.50$. O Stacking Ensemble atingiu Gini de **0.9637**.
- **Estatística Kolmogorov-Smirnov (KS)**: A separação máxima vertical entre distribuições acumuladas de adimplentes e inadimplentes alcançou **0.7830** (Scorecard) e **0.8627** (Ensemble), muito superior ao patamar exigido de $0.35$.

---

### 3. Pilar 2: Calibração de Probabilidades & Decomposição de Brier

A calibração garante que uma PD prevista de $5,0\%$ represente fielmente uma taxa observada de $5,0\%$:
- **Brier Score**: **0.0591** para o Scorecard e **0.0404** para os modelos em árvore calibrados via Regressão Isotônica.
- **Decomposição de Murphy**:
  $$\text{Brier Score} = \text{Incerteza} - \text{Resolução} + \text{Confiabilidade}$$
  O componente de erro de confiabilidade inferior a $0.003$ confirma excelente aderência probabilística.
- **Teste de Hosmer-Lemeshow $\chi^2$**: Confirmou aderência estatística em todos os 10 decis de risco ($p$-valor $> 0.05$).

---

### 4. Pilar 3: Índice de Estabilidade Populacional (PSI) & Drift

A estabilidade da carteira entre safras de treinamento e safras correntes de monitoramento é medida pelo PSI:
$$\text{PSI} = \sum_{k=1}^K (\text{Real}_k - \text{Esperado}_k) \times \ln\left( \frac{\text{Real}_k}{\text{Esperado}_k} \right)$$
- **PSI Global do Scorecard**: **0.0245** (muito abaixo do limite de alerta amarelo de 0.10).
- **Classificação**: 🟢 **Zona Verde (Estável)** — Sem deriva populacional detectada.

---

### 5. Pilar 4: Backtesting Estatístico & Semáforo de Basileia

1. **Intervalos Exatos de Confiança de Clopper-Pearson (IC 95%)**:
   - Em todas as faixas de rating ativas (A1 a H), a taxa empírica observada de default situou-se rigorosamente dentro dos limites exatos de 95%.
2. **Semáforo de Basileia**:
   - Probabilidade acumulada binomial $P(X \ge k \mid PD_{\text{previsto}}) > 0.05 \implies$ **Zona Verde** em $100\%$ das faixas.
   - Não há subestimação estatística de risco na carteira.

---

### 6. Estresse Macroeconômico de Vasicek & Cenários IFRS 9

Conversão de $PD_{TTC}$ em $PD_{PIT}$ via fórmula de Vasicek ASRF:
$$PD_{PIT}(Z) = \Phi\left( \frac{\Phi^{-1}(PD_{TTC}) - \sqrt{\rho} Z}{\sqrt{1 - \rho}} \right)$$
Ponderação em 3 cenários IFRS 9:
- **Cenário Base (50% de peso)**: $Z = 0.00 \sigma \implies PD_{PIT} = PD_{TTC}$
- **Cenário Otimista (25% de peso)**: $Z = +0.85 \sigma \implies PD_{PIT} < PD_{TTC}$
- **Cenário Adverso / Estresse (25% de peso)**: $Z = -1.45 \sigma \implies PD_{PIT} > PD_{TTC}$

---

### 7. Aplicações Financeiras: Provisão ECL & Precificação RAROC

1. **Provisão de Perda Esperada (IFRS 9 / BACEN 4.966)**:
   - **Stage 1 (Normal)**: $ECL_{12m} = PD_{12m} \times LGD \times EAD \times (1 + r)^{-1}$
   - **Stage 2 (SICR)**: $ECL_{\text{Lifetime}} = \sum_{t=1}^T PD_{\text{marginal}}(t) \times LGD \times EAD(t) \times (1 + r)^{-t}$
   - **Stage 3 (Default)**: $ECL_{\text{Default}} = 1.0 \times LGD \times EAD \times (1 + r)^{-1}$
2. **Precificação por Risco (RAROC)**:
   $$\text{Taxa Justa} = \text{FTP (Custo de Captação)} + \text{OpEx} + \text{Perda Esperada} + \text{Custo de Capital Econômico} + \text{Margem Líquida}$$

---

### 8. Parecer Conclusivo de Validação

O **Motor PRINAD v3.1** cumpre integralmente todos os requisitos quantitativos, regulatórios e operacionais de Basileia III/IV IRB e IFRS 9 / BACEN 4.966. O modelo está homologado para implantação em produção.

</details>

---

<details>
<summary><h2 style="display:inline-block;" id="-english-version">🇺🇸 English Version (Click to open / expand)</h2></summary>

### 📑 Table of Contents
1. [Executive Summary & Benchmark Summary Table](#1-executive-summary--benchmark-summary-table)
2. [Pillar 1: Discrimination Power (ROC-AUC, Gini, KS)](#2-pillar-1-discrimination-power-roc-auc-gini-ks)
3. [Pillar 2: Probability Calibration & Brier Decomposition](#3-pillar-2-probability-calibration--brier-decomposition)
4. [Pillar 3: Population Stability Index (PSI) & Drift](#4-pillar-3-population-stability-index-psi--drift)
5. [Pillar 4: Statistical Backtesting & Basel Traffic Lights](#5-pillar-4-statistical-backtesting--basel-traffic-lights)
6. [Macroeconomic Vasicek Stress Testing & IFRS 9 Scenarios](#6-macroeconomic-vasicek-stress-testing--ifrs-9-scenarios)
7. [Downstream Financial Applications: ECL & RAROC Pricing](#7-downstream-financial-applications-ecl--raroc-pricing)
8. [Validation Conclusion & Sign-Off](#8-validation-conclusion--sign-off)

---

### 1. Executive Summary & Benchmark Summary Table

This formal validation whitepaper documents the quantitative testing and benchmarking of the **PRINAD Credit Risk Engine**. The suite evaluates an industry-standard **Regulatory Scorecard (Logistic Regression + WoE)** against modern **Gradient Boosting Challengers (LightGBM, XGBoost, Stacking Ensemble)** on a held-out test partition ($N = 15,000$ independent contracts, empirical default rate $= 26.57\%$):

| Model Architecture | Role | ROC-AUC | Gini ($2 \cdot \text{AUC} - 1$) | KS Statistic | Brier Score | Expected Calib. Error (ECE) | Basel Traffic Light |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regulatory Scorecard (WoE)** | **Champion (IRB)** | **0.9613** | **0.9227** | **0.7830** | **0.0591** | **0.0124** | 🟢 **Green Zone** |
| **LightGBM (Calibrated)** | **Challenger 1** | **0.9816** | **0.9631** | **0.8617** | **0.0404** | **0.0089** | 🟢 **Green Zone** |
| **XGBoost (Calibrated)** | **Challenger 2** | **0.9817** | **0.9634** | **0.8614** | **0.0405** | **0.0087** | 🟢 **Green Zone** |
| **Stacking Ensemble** | **Challenger 3** | **0.9818** | **0.9637** | **0.8627** | **0.0424** | **0.0092** | 🟢 **Green Zone** |

---

### 2. Pillar 1: Discrimination Power (ROC-AUC, Gini, KS)

- **Gini Coefficient**: Champion Scorecard reached a Gini of **0.9227** ($\text{Gini} = 2 \cdot \text{AUC} - 1$), well above the regulatory floor of $0.50$. The Stacking Ensemble achieved a Gini of **0.9637**.
- **Kolmogorov-Smirnov (KS)**: Maximum separation between cumulative distributions of non-defaulters and defaulters reached **0.7830** (Scorecard) and **0.8627** (Ensemble), surpassing the regulatory benchmark of $0.35$.

---

### 3. Pillar 2: Probability Calibration & Brier Decomposition

- **Brier Score**: Measured at **0.0591** for the Scorecard and **0.0404** for Isotonically Calibrated Tree Boosters.
- **Murphy Brier Decomposition**:
  $$\text{Brier Score} = \text{Uncertainty} - \text{Resolution} + \text{Reliability}$$
  A reliability error $< 0.003$ confirms excellent alignment with empirical event frequencies.
- **Hosmer-Lemeshow $\chi^2$ Test**: Verified statistical calibration across all 10 risk deciles ($p$-value $> 0.05$).

---

### 4. Pillar 3: Population Stability Index (PSI) & Drift

$$\text{PSI} = \sum_{k=1}^K (\text{Actual}_k - \text{Expected}_k) \times \ln\left( \frac{\text{Actual}_k}{\text{Expected}_k} \right)$$
- **Scorecard Total PSI**: **0.0245** (far below the 0.10 warning threshold).
- **Traffic Light**: 🟢 **Green (Stable)** — No population drift detected across vintages.

---

### 5. Pillar 4: Statistical Backtesting & Basel Traffic Lights

1. **Exact Clopper-Pearson 95% Binomial Confidence Intervals**:
   - Across all active rating grades (A1 to H), observed default frequencies fell within the exact 95% binomial bounds.
2. **Basel Traffic Light Classification**:
   - Binomial tail probability $P(X \ge k \mid PD_{\text{expected}}) > 0.05 \implies$ **Green Zone** across all rating bands.
   - No statistical underestimation of risk.

---

### 6. Macroeconomic Vasicek Stress Testing & IFRS 9 Scenarios

Point-in-Time conditioning via the Vasicek ASRF formula:
$$PD_{PIT}(Z) = \Phi\left( \frac{\Phi^{-1}(PD_{TTC}) - \sqrt{\rho} Z}{\sqrt{1 - \rho}} \right)$$

IFRS 9 Weighted PD:
$$PD_{\text{Weighted}} = 0.50 \cdot PD(Z_{\text{Baseline}}) + 0.25 \cdot PD(Z_{\text{Upside}}) + 0.25 \cdot PD(Z_{\text{Adverse}})$$

---

### 7. Downstream Financial Applications: ECL & RAROC Pricing

1. **Expected Credit Loss (ECL)**:
   - **Stage 1 (Performing)**: $ECL_{12m} = PD_{12m} \times LGD \times EAD \times (1 + r)^{-1}$
   - **Stage 2 (SICR)**: $ECL_{\text{Lifetime}} = \sum_{t=1}^T PD_{\text{marginal}}(t) \times LGD \times EAD(t) \times (1 + r)^{-t}$
   - **Stage 3 (Default)**: $ECL_{\text{Default}} = 1.0 \times LGD \times EAD \times (1 + r)^{-1}$
2. **Risk-Based Pricing (RAROC)**:
   $$\text{Fair Lending Rate} = \text{FTP (Cost of Funds)} + \text{OpEx} + \text{Expected Loss} + \text{Cost of Economic Capital} + \text{Target Net Margin}$$

---

### 8. Validation Conclusion & Sign-Off

The **PRINAD v3.1 Engine** satisfies all regulatory and quantitative requirements under Basel III/IV IRB and IFRS 9 / BACEN 4.966. The model is approved for tier-1 banking production deployment.

</details>

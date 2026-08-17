"""
PRINAD - Lifetime PD & Term Structure Module
============================================
Implements multi-year marginal, cumulative, and survival PD curves for IFRS 9 / BACEN 4.966 Stage 2/3:
- Markov Credit Rating Transition Matrix Multi-Period Projection
- Hazard Rate & Cumulative Survival Formulation
- Lifetime PD Term Structure Generation (Years 1 to 10)
- Discount Factor Curve integration for Lifetime ECL Calculation

Author: PRINAD Quantitative Risk Team
Standard: IFRS 9 / BACEN 4.966 Stage 2 & 3 Compliance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)


class LifetimePDEngine:
    """
    Computes Lifetime Marginal and Cumulative PD Curves across 1 to 10 Year Horizons.
    """
    
    # 11 Rating Categories (PRINAD Master Scale)
    RATINGS = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'D', 'DEFAULT']
    
    # 1-Year Baseline Transition Matrix (Standard IRB Calibrated Matrix)
    # Rows: Initial Rating, Columns: Rating at End of Year (DEFAULT is Absorbing State)
    BASE_TRANSITION_MATRIX = np.array([
        # A1     A2     A3     B1     B2     B3     C1     C2     C3     D      DEFAULT
        [0.860, 0.100, 0.025, 0.010, 0.003, 0.001, 0.000, 0.000, 0.000, 0.000, 0.001],  # A1 (~0.1% PD)
        [0.080, 0.780, 0.090, 0.030, 0.012, 0.005, 0.002, 0.000, 0.000, 0.000, 0.001],  # A2 (~0.1% PD)
        [0.020, 0.080, 0.740, 0.100, 0.035, 0.015, 0.005, 0.002, 0.001, 0.000, 0.002],  # A3 (~0.2% PD)
        [0.005, 0.025, 0.090, 0.700, 0.110, 0.040, 0.018, 0.007, 0.002, 0.001, 0.002],  # B1 (~0.2% PD)
        [0.001, 0.008, 0.030, 0.090, 0.670, 0.120, 0.045, 0.020, 0.009, 0.002, 0.005],  # B2 (~0.5% PD)
        [0.000, 0.002, 0.010, 0.035, 0.095, 0.640, 0.120, 0.050, 0.025, 0.008, 0.015],  # B3 (~1.5% PD)
        [0.000, 0.000, 0.003, 0.010, 0.030, 0.090, 0.600, 0.140, 0.065, 0.022, 0.040],  # C1 (~4.0% PD)
        [0.000, 0.000, 0.001, 0.003, 0.010, 0.030, 0.080, 0.570, 0.160, 0.056, 0.090],  # C2 (~9.0% PD)
        [0.000, 0.000, 0.000, 0.001, 0.003, 0.010, 0.025, 0.080, 0.540, 0.161, 0.180],  # C3 (~18.0% PD)
        [0.000, 0.000, 0.000, 0.000, 0.001, 0.002, 0.008, 0.020, 0.080, 0.439, 0.450],  # D  (~45.0% PD)
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000],  # DEFAULT (100% Absorbing)
    ])
    
    def __init__(self):
        # Normalize rows to guarantee exact sum = 1.0
        self.transition_matrix = self.BASE_TRANSITION_MATRIX / self.BASE_TRANSITION_MATRIX.sum(axis=1, keepdims=True)
        
    def generate_rating_term_structure(self, max_years: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Compute multi-year transition matrices M^1, M^2, ..., M^max_years.
        """
        matrices = {}
        curr_m = np.eye(len(self.RATINGS))
        
        for y in range(1, max_years + 1):
            curr_m = np.matmul(curr_m, self.transition_matrix)
            df_m = pd.DataFrame(curr_m, index=self.RATINGS, columns=self.RATINGS)
            matrices[f'Year_{y}'] = df_m
            
        return matrices

    def get_lifetime_curve_from_rating(
        self,
        rating: str,
        max_years: int = 5,
        discount_rate: float = 0.10
    ) -> pd.DataFrame:
        """
        Generate marginal and cumulative Lifetime PD schedule for a given initial rating grade.
        """
        if rating not in self.RATINGS:
            rating = 'B2'
            
        rating_idx = self.RATINGS.index(rating)
        default_col_idx = self.RATINGS.index('DEFAULT')
        
        curr_m = np.eye(len(self.RATINGS))
        
        records = []
        prev_cum_pd = 0.0
        
        for t in range(1, max_years + 1):
            curr_m = np.matmul(curr_m, self.transition_matrix)
            cum_pd = float(curr_m[rating_idx, default_col_idx])
            marginal_pd = max(0.0, cum_pd - prev_cum_pd)
            survival_prob = 1.0 - cum_pd
            discount_factor = 1.0 / ((1.0 + discount_rate) ** t)
            
            records.append({
                'Year': t,
                'Cumulative_PD': round(cum_pd, 6),
                'Cumulative_PD_Pct': round(cum_pd * 100, 3),
                'Marginal_PD': round(marginal_pd, 6),
                'Marginal_PD_Pct': round(marginal_pd * 100, 3),
                'Survival_Rate': round(survival_prob, 6),
                'Discount_Factor': round(discount_factor, 4)
            })
            prev_cum_pd = cum_pd
            
        return pd.DataFrame(records)

    def generate_custom_lifetime_curve(
        self,
        pd_12m: float,
        maturity_years: int = 5,
        decay_factor: float = 0.85,
        discount_rate: float = 0.10
    ) -> pd.DataFrame:
        """
        Generate continuous term-structure curve for any exact PD_12m value.
        
        Uses decreasing hazard decay curve: h(t) = h(1) * decay_factor^(t-1)
        """
        pd_12m = max(0.0005, min(0.999, pd_12m))
        h1 = float(-np.log(max(1.0 - pd_12m, 1e-6)))
        
        records = []
        cum_survival = 1.0
        
        for t in range(1, maturity_years + 1):
            ht = h1 * (decay_factor ** (t - 1))
            marginal_survival = float(np.exp(-ht))
            
            prev_cum_surv = cum_survival
            cum_survival = cum_survival * marginal_survival
            
            cum_pd = 1.0 - cum_survival
            marginal_pd = max(0.0, prev_cum_surv - cum_survival)
            discount_factor = 1.0 / ((1.0 + discount_rate) ** t)
            
            records.append({
                'Year': t,
                'Hazard_Rate': round(ht, 6),
                'Cumulative_PD': round(cum_pd, 6),
                'Cumulative_PD_Pct': round(cum_pd * 100, 3),
                'Marginal_PD': round(marginal_pd, 6),
                'Marginal_PD_Pct': round(marginal_pd * 100, 3),
                'Survival_Rate': round(cum_survival, 6),
                'Discount_Factor': round(discount_factor, 4)
            })
            
        return pd.DataFrame(records)

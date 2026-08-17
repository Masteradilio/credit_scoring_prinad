"""
PRINAD - Credit Decisioning, ECL & Risk-Based Pricing Engine
============================================================
Implements financial and business applications for Credit Risk:
1. IFRS 9 / BACEN 4.966 Expected Credit Loss (ECL) Calculation across Stages 1, 2, and 3
2. Risk-Based Pricing & RAROC (Risk-Adjusted Return on Capital) Engine
3. Basel Economic Capital & Unexpected Loss (UL) Calculation
4. Economic Cut-off Optimization (Net Profit Maximization Curve)

Author: PRINAD Quantitative Risk Team
Standard: IFRS 9 / BACEN 4.966 & Basel III/IV Capital Framework
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ECLResult:
    """Detailed result of IFRS 9 / BACEN 4.966 Expected Credit Loss calculation."""
    stage: int                          # Stage 1, 2, or 3
    stage_name: str                     # "Stage 1 (Normal)", "Stage 2 (SICR)", "Stage 3 (Default)"
    ead: float                          # Exposure at Default ($)
    lgd: float                          # Loss Given Default (0 to 1)
    pd_applied: float                   # PD used for the calculation
    horizon: str                        # "12 Meses" or "Lifetime"
    ecl_amount: float                   # Provision Amount ($)
    ecl_percentage: float               # Provision as % of EAD
    sicr_trigger: str                   # Reason for stage classification


@dataclass
class PricingResult:
    """Risk-Based Pricing breakdown."""
    cost_of_funds_pct: float            # Funding cost / FTP (%)
    opex_cost_pct: float                # Operational expense (%)
    expected_loss_pct: float            # EL = PD * LGD (%)
    economic_capital_pct: float         # Basel Unexpected Loss Capital Allocation (%)
    capital_charge_pct: float           # Cost of Capital (Hurdle Rate * Capital) (%)
    target_net_margin_pct: float        # Profit margin desired (%)
    fair_lending_rate_annual: float     # Recommended minimum annual interest rate (%)
    raroc_percentage: float             # Risk-Adjusted Return on Capital (%)


class DecisionAndPricingEngine:
    """
    Financial and Econometric Decision Engine for Credit Risk.
    """
    
    def __init__(
        self,
        hurdle_rate: float = 0.15,      # 15% Target Return on Equity (ROE)
        base_ftp_rate: float = 0.105,   # 10.5% Base Cost of Funds (Selic/Treasury)
        opex_rate: float = 0.025,       # 2.5% Annual Operating Expenses
        default_lgd: float = 0.45       # 45% Standard Unsecured LGD (Basel Standard)
    ):
        self.hurdle_rate = hurdle_rate
        self.base_ftp_rate = base_ftp_rate
        self.opex_rate = opex_rate
        self.default_lgd = default_lgd

    # =========================================================================
    # 1. IFRS 9 / BACEN 4.966 EXPECTED CREDIT LOSS (ECL)
    # =========================================================================
    
    def calculate_ecl(
        self,
        ead: float,
        pd_12m: float,
        pd_lifetime: float,
        days_past_due: int = 0,
        pd_origination: Optional[float] = None,
        lgd: Optional[float] = None,
        discount_rate: float = 0.10
    ) -> ECLResult:
        """
        Calculate Stage-based Expected Credit Loss (ECL).
        
        Staging Rules:
        - Stage 3: DPD >= 90 days or Objective Impairment (PD >= 0.85)
        - Stage 2: DPD >= 30 days OR Significant Increase in Credit Risk (SICR: PD_current / PD_orig >= 2.5)
        - Stage 1: Standard performing (DPD < 30 and no SICR)
        """
        lgd_val = lgd if lgd is not None else self.default_lgd
        df_1y = 1.0 / (1.0 + discount_rate)
        
        # 1. Determine Stage
        is_stage_3 = (days_past_due >= 90) or (pd_12m >= 0.85)
        
        # SICR test: 2.5x increase in PD since origination or 30+ DPD
        sicr_pd_ratio = (pd_12m / max(pd_origination, 1e-4)) if pd_origination else 1.0
        is_stage_2 = (not is_stage_3) and ((days_past_due >= 30) or (sicr_pd_ratio >= 2.5) or (pd_12m >= 0.40))
        
        if is_stage_3:
            stage = 3
            stage_name = "Stage 3 (Credit-Impaired / Inadimplente)"
            horizon = "Lifetime"
            pd_applied = 1.0
            sicr_trigger = f"DPD={days_past_due}>=90 ou PD={pd_12m:.1%}>=85%"
            ecl = ead * lgd_val * df_1y
        elif is_stage_2:
            stage = 2
            stage_name = "Stage 2 (Underperforming / Risco Aumentado)"
            horizon = "Lifetime"
            pd_applied = pd_lifetime
            sicr_trigger = f"SICR detectado (DPD={days_past_due}>=30 ou PD={pd_12m:.1%})"
            ecl = ead * pd_lifetime * lgd_val * df_1y
        else:
            stage = 1
            stage_name = "Stage 1 (Performing / Normal)"
            horizon = "12 Meses"
            pd_applied = pd_12m
            sicr_trigger = "Crédito adimplente com risco controlado"
            ecl = ead * pd_12m * lgd_val * df_1y
            
        return ECLResult(
            stage=stage,
            stage_name=stage_name,
            ead=round(ead, 2),
            lgd=round(lgd_val, 4),
            pd_applied=round(pd_applied, 6),
            horizon=horizon,
            ecl_amount=round(ecl, 2),
            ecl_percentage=round((ecl / max(ead, 1e-6)) * 100, 3),
            sicr_trigger=sicr_trigger
        )

    # =========================================================================
    # 2. RISK-BASED PRICING & RAROC (PRECIFICAÇÃO POR RISCO)
    # =========================================================================
    
    def price_credit(
        self,
        pd_12m: float,
        lgd: Optional[float] = None,
        target_margin: float = 0.03,  # 3.0% Desired net economic profit
        asset_class: str = "retail_other"
    ) -> PricingResult:
        """
        Calculate risk-adjusted fair lending interest rate and RAROC.
        """
        lgd_val = lgd if lgd is not None else self.default_lgd
        pd_val = max(0.0005, min(0.999, pd_12m))
        
        # 1. Expected Loss (EL)
        el_rate = pd_val * lgd_val
        
        # 2. Basel Unexpected Loss / Economic Capital (99.9% VaR)
        # Asset correlation rho
        if asset_class == "retail_revolving":
            rho = 0.04
        elif asset_class == "retail_mortgage":
            rho = 0.15
        else:
            k = (1.0 - np.exp(-35.0 * pd_val)) / (1.0 - np.exp(-35.0))
            rho = 0.03 * k + 0.16 * (1.0 - k)
            
        # 99.9% Vasicek Worst-Case Default Rate (WCDR)
        inv_pd = norm.ppf(pd_val)
        inv_999 = norm.ppf(0.999)
        wcdr = norm.cdf((inv_pd + np.sqrt(rho) * inv_999) / np.sqrt(1.0 - rho))
        
        # Unexpected Loss (Capital required per $1 exposure)
        economic_capital = max(0.01, (wcdr - pd_val) * lgd_val)
        
        # Cost of Capital (Capital Charge)
        capital_charge = economic_capital * self.hurdle_rate
        
        # Fair Lending Rate (Annual Nominal Rate)
        fair_rate = self.base_ftp_rate + self.opex_rate + el_rate + capital_charge + target_margin
        
        # RAROC = (Revenues - Costs - EL) / Economic Capital
        net_risk_adjusted_income = fair_rate - self.base_ftp_rate - self.opex_rate - el_rate
        raroc = net_risk_adjusted_income / max(economic_capital, 1e-4)
        
        return PricingResult(
            cost_of_funds_pct=round(self.base_ftp_rate * 100, 2),
            opex_cost_pct=round(self.opex_rate * 100, 2),
            expected_loss_pct=round(el_rate * 100, 2),
            economic_capital_pct=round(economic_capital * 100, 2),
            capital_charge_pct=round(capital_charge * 100, 2),
            target_net_margin_pct=round(target_margin * 100, 2),
            fair_lending_rate_annual=round(fair_rate * 100, 2),
            raroc_percentage=round(raroc * 100, 2)
        )

    # =========================================================================
    # 3. ECONOMIC CUT-OFF OPTIMIZATION (CURVA DE LUCRO LÍQUIDO)
    # =========================================================================
    
    def optimize_cutoff(
        self,
        predicted_pds: np.ndarray,
        average_loan_amount: float = 10000.0,
        average_interest_rate: float = 0.28,  # 28% annual loan rate
        lgd: float = 0.50,
        n_thresholds: int = 50
    ) -> Dict[str, Any]:
        """
        Simulate net profitability across different PD acceptance thresholds.
        Identifies the optimal threshold maximizing Total Net Portfolio Profit ($).
        """
        thresholds = np.linspace(0.01, 0.60, n_thresholds)
        pds = np.array(predicted_pds, dtype=float)
        
        simulation_data = []
        best_profit = -np.inf
        optimal_cutoff = 0.15
        
        for cut in thresholds:
            accepted_mask = (pds <= cut)
            n_accepted = int(accepted_mask.sum())
            acceptance_rate = float(n_accepted / len(pds))
            
            if n_accepted == 0:
                continue
                
            accepted_pds = pds[accepted_mask]
            avg_pd_accepted = float(accepted_pds.mean())
            
            # Revenue = Volume * Interest Rate * (1 - PD)
            total_volume = n_accepted * average_loan_amount
            gross_interest = total_volume * average_interest_rate
            funding_cost = total_volume * self.base_ftp_rate
            opex_cost = total_volume * self.opex_rate
            expected_defaults_loss = total_volume * avg_pd_accepted * lgd
            
            net_profit = gross_interest - funding_cost - opex_cost - expected_defaults_loss
            profit_per_loan = net_profit / n_accepted
            
            if net_profit > best_profit:
                best_profit = net_profit
                optimal_cutoff = cut
                
            simulation_data.append({
                'pd_cutoff_pct': round(cut * 100, 1),
                'acceptance_rate_pct': round(acceptance_rate * 100, 1),
                'accepted_loans_count': n_accepted,
                'portfolio_volume': round(total_volume, 2),
                'portfolio_avg_pd_pct': round(avg_pd_accepted * 100, 2),
                'net_profit_total': round(net_profit, 2),
                'net_profit_per_loan': round(profit_per_loan, 2)
            })
            
        return {
            'optimal_pd_cutoff_pct': round(optimal_cutoff * 100, 2),
            'max_net_profit': round(best_profit, 2),
            'simulation_curve': simulation_data
        }

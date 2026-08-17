"""
PRINAD - Vasicek Macroeconomic & PIT-TTC Transition Module
==========================================================
Implements the Asymptotic Single Risk Factor (ASRF) Vasicek Framework:
- Point-in-Time (PIT) to Through-the-Cycle (TTC) transformation
- Forward-Looking Macroeconomic Factor Conditioning (Z-factor)
- Basel Regulatory Asset Correlation (rho) functions
- IFRS 9 Multi-Scenario Probability-Weighted Forward-Looking PD (Baseline, Upside, Adverse)

Author: PRINAD Quantitative Risk Team
Standard: Basel III/IV IRB & IFRS 9 / BACEN 4.966
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MacroScenario:
    """Definition of a forward-looking macroeconomic scenario."""
    name: str
    weight: float                     # Probability weight of the scenario (sum to 1.0)
    gdp_growth: float                 # Annual GDP growth rate (%)
    selic_rate: float                 # Benchmark interest rate (%)
    unemployment_rate: float          # Unemployment rate (%)
    inflation_ipca: float             # Inflation rate (%)
    z_factor: Optional[float] = None  # Computed systematic shock factor


class VasicekMacroEngine:
    """
    Quantitative Engine for Vasicek ASRF Transformation & Macroeconomic Stress Testing.
    
    Formula:
        PD_PIT(Z) = Phi( (Phi^-1(PD_TTC) - sqrt(rho) * Z) / sqrt(1 - rho) )
        
    Where:
        - Phi: Standard normal cumulative distribution function
        - Phi^-1: Inverse standard normal CDF (probit link)
        - rho: Asset correlation (sensitivity to systematic factor)
        - Z: Systematic macroeconomic factor (Z > 0 expansion, Z < 0 recession)
    """
    
    # Historical Macroeconomic Baselines (Brazil / LatAm calibration)
    MACRO_MEANS = {
        'gdp_growth': 2.0,           # Baseline 2.0% annual growth
        'selic_rate': 10.5,          # Baseline 10.5% Selic
        'unemployment_rate': 7.8,    # Baseline 7.8% unemployment
        'inflation_ipca': 4.0        # Baseline 4.0% IPCA
    }
    
    MACRO_STDS = {
        'gdp_growth': 2.2,
        'selic_rate': 3.5,
        'unemployment_rate': 2.0,
        'inflation_ipca': 2.5
    }
    
    # Elasticity coefficients (how macro shocks affect Z)
    MACRO_WEIGHTS = {
        'gdp_growth': 0.45,          # Positive GDP -> Positive Z (Lower Default)
        'selic_rate': -0.30,         # High Selic -> Negative Z (Higher Default)
        'unemployment_rate': -0.25   # High Unemployment -> Negative Z (Higher Default)
    }

    def __init__(self, default_asset_class: str = "retail_other"):
        self.asset_class = default_asset_class
        
        # Default IFRS 9 forward-looking 3 scenarios
        self.default_scenarios = [
            MacroScenario(
                name="Cenário Base (Expectativa de Mercado)",
                weight=0.50,
                gdp_growth=2.2,
                selic_rate=10.0,
                unemployment_rate=7.5,
                inflation_ipca=3.8
            ),
            MacroScenario(
                name="Cenário Otimista (Upside)",
                weight=0.25,
                gdp_growth=3.8,
                selic_rate=8.5,
                unemployment_rate=6.2,
                inflation_ipca=3.2
            ),
            MacroScenario(
                name="Cenário Adverso (Downturn / Estresse)",
                weight=0.25,
                gdp_growth=-1.5,
                selic_rate=14.5,
                unemployment_rate=11.0,
                inflation_ipca=7.5
            )
        ]
        
    @staticmethod
    def calculate_basel_rho(pd_val: float, asset_class: str = "retail_other") -> float:
        """
        Calculate regulatory asset correlation (rho) per Basel III/IV IRB formulas.
        
        Args:
            pd_val: Probability of default (0 to 1)
            asset_class: 'corporate', 'retail_revolving', 'retail_mortgage', 'retail_other'
        """
        pd_clamped = max(0.0003, min(0.999, pd_val))
        
        if asset_class == "retail_revolving":  # Credit card / Cheque especial
            return 0.04
        elif asset_class == "retail_mortgage":  # Imobiliário
            return 0.15
        elif asset_class == "corporate":  # Empresas
            k = (1.0 - np.exp(-50.0 * pd_clamped)) / (1.0 - np.exp(-50.0))
            return 0.12 * k + 0.24 * (1.0 - k)
        else:  # 'retail_other' (Consignado, Pessoal, Veículos)
            k = (1.0 - np.exp(-35.0 * pd_clamped)) / (1.0 - np.exp(-35.0))
            return 0.03 * k + 0.16 * (1.0 - k)

    def calculate_z_factor(self, gdp: float, selic: float, unemployment: float) -> float:
        """
        Calculate systematic macroeconomic factor Z (in standard standard normal units).
        
        Z > 0: Favorable economic cycle -> Lower default rate
        Z < 0: Unfavorable economic stress -> Higher default rate
        """
        norm_gdp = (gdp - self.MACRO_MEANS['gdp_growth']) / self.MACRO_STDS['gdp_growth']
        norm_selic = (selic - self.MACRO_MEANS['selic_rate']) / self.MACRO_STDS['selic_rate']
        norm_unemp = (unemployment - self.MACRO_MEANS['unemployment_rate']) / self.MACRO_STDS['unemployment_rate']
        
        z = (
            self.MACRO_WEIGHTS['gdp_growth'] * norm_gdp +
            self.MACRO_WEIGHTS['selic_rate'] * norm_selic +
            self.MACRO_WEIGHTS['unemployment_rate'] * norm_unemp
        )
        return float(np.clip(z, -3.5, 3.5))

    def ttc_to_pit(
        self,
        pd_ttc: Union[float, np.ndarray],
        z_factor: float,
        asset_class: Optional[str] = None
    ) -> Union[float, np.ndarray]:
        """
        Convert Through-the-Cycle (TTC) PD to Point-in-Time (PIT) PD conditional on Z.
        """
        ac = asset_class or self.asset_class
        is_scalar = np.isscalar(pd_ttc)
        pds = np.array([pd_ttc] if is_scalar else pd_ttc, dtype=float)
        pds = np.clip(pds, 1e-4, 0.9999)
        
        pit_pds = np.zeros_like(pds)
        for i, pd_val in enumerate(pds):
            rho = self.calculate_basel_rho(pd_val, ac)
            inv_ttc = norm.ppf(pd_val)
            numerator = inv_ttc - np.sqrt(rho) * z_factor
            denominator = np.sqrt(1.0 - rho)
            pit_pds[i] = norm.cdf(numerator / denominator)
            
        pit_pds = np.clip(pit_pds, 0.0005, 0.9999)
        return float(pit_pds[0]) if is_scalar else pit_pds

    def pit_to_ttc(
        self,
        pd_pit: Union[float, np.ndarray],
        z_factor: float,
        asset_class: Optional[str] = None
    ) -> Union[float, np.ndarray]:
        """
        Convert Point-in-Time (PIT) PD back to Through-the-Cycle (TTC) long-term baseline.
        """
        ac = asset_class or self.asset_class
        is_scalar = np.isscalar(pd_pit)
        pds = np.array([pd_pit] if is_scalar else pd_pit, dtype=float)
        pds = np.clip(pds, 1e-4, 0.9999)
        
        ttc_pds = np.zeros_like(pds)
        for i, pd_val in enumerate(pds):
            rho = self.calculate_basel_rho(pd_val, ac)
            inv_pit = norm.ppf(pd_val)
            numerator = inv_pit * np.sqrt(1.0 - rho) + np.sqrt(rho) * z_factor
            ttc_pds[i] = norm.cdf(numerator)
            
        ttc_pds = np.clip(ttc_pds, 0.0005, 0.9999)
        return float(ttc_pds[0]) if is_scalar else ttc_pds

    def evaluate_ifrs9_scenarios(
        self,
        pd_baseline: float,
        is_input_ttc: bool = False,
        scenarios: Optional[List[MacroScenario]] = None
    ) -> Dict[str, Any]:
        """
        Perform IFRS 9 forward-looking multi-scenario probability-weighted PD calculation.
        """
        active_scenarios = scenarios or self.default_scenarios
        
        # Ensure we have TTC PD as the unconditional anchor
        if is_input_ttc:
            pd_ttc = pd_baseline
        else:
            # Assume baseline input is PIT at current neutral condition (Z=0)
            pd_ttc = pd_baseline
            
        scenario_results = []
        weighted_pd = 0.0
        
        for sc in active_scenarios:
            z = self.calculate_z_factor(sc.gdp_growth, sc.selic_rate, sc.unemployment_rate)
            sc.z_factor = round(z, 3)
            
            sc_pd = self.ttc_to_pit(pd_ttc, z)
            scenario_results.append({
                'scenario_name': sc.name,
                'probability_weight': sc.weight,
                'macro_inputs': {
                    'gdp_growth': sc.gdp_growth,
                    'selic_rate': sc.selic_rate,
                    'unemployment': sc.unemployment_rate,
                    'ipca': sc.inflation_ipca
                },
                'z_factor': round(z, 3),
                'conditional_pd': round(sc_pd, 6),
                'conditional_pd_pct': round(sc_pd * 100, 3)
            })
            weighted_pd += sc.weight * sc_pd
            
        return {
            'pd_ttc_anchor': round(pd_ttc, 6),
            'pd_ttc_anchor_pct': round(pd_ttc * 100, 3),
            'pd_forward_looking_weighted': round(weighted_pd, 6),
            'pd_forward_looking_weighted_pct': round(weighted_pd * 100, 3),
            'scenarios': scenario_results
        }

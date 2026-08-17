"""
Unit Tests for Vasicek Macro Engine, Lifetime PD and Decision/Pricing Engine
"""
import pytest
import numpy as np
import sys
from pathlib import Path

MODELOS_DIR = Path(__file__).resolve().parent.parent / "modelos"
if str(MODELOS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELOS_DIR))

from vasicek_macro import VasicekMacroEngine
from lifetime_pd import LifetimePDEngine
from decision_pricing_engine import DecisionAndPricingEngine


def test_vasicek_ttc_to_pit_recession():
    engine = VasicekMacroEngine()
    pd_ttc = 0.05
    
    # Severe economic stress (negative Z)
    z_recession = -2.0
    pd_pit = engine.ttc_to_pit(pd_ttc, z_recession)
    
    # In recession, PIT PD must be higher than TTC PD
    assert pd_pit > pd_ttc
    assert 0.05 < pd_pit < 1.0


def test_vasicek_ttc_to_pit_expansion():
    engine = VasicekMacroEngine()
    pd_ttc = 0.05
    
    # Economic boom (positive Z)
    z_boom = 2.0
    pd_pit = engine.ttc_to_pit(pd_ttc, z_boom)
    
    # In boom, PIT PD must be lower than TTC PD
    assert pd_pit < pd_ttc
    assert 0.0005 <= pd_pit < 0.05


def test_ifrs9_multi_scenarios():
    engine = VasicekMacroEngine()
    res = engine.evaluate_ifrs9_scenarios(pd_baseline=0.04)
    
    assert 'pd_forward_looking_weighted' in res
    assert 'scenarios' in res
    assert len(res['scenarios']) == 3
    # Check that weights sum to 1.0
    total_w = sum(s['probability_weight'] for s in res['scenarios'])
    assert abs(total_w - 1.0) < 1e-5


def test_lifetime_pd_monotonicity():
    engine = LifetimePDEngine()
    df_curve = engine.get_lifetime_curve_from_rating('B2', max_years=5)
    
    assert len(df_curve) == 5
    # Cumulative PD must be strictly non-decreasing over time
    cum_pds = df_curve['Cumulative_PD'].values
    assert np.all(np.diff(cum_pds) >= 0)
    assert cum_pds[-1] <= 1.0


def test_ecl_staging_logic():
    engine = DecisionAndPricingEngine()
    
    # Stage 1: clean borrower
    ecl_1 = engine.calculate_ecl(ead=10000.0, pd_12m=0.02, pd_lifetime=0.06, days_past_due=0)
    assert ecl_1.stage == 1
    assert ecl_1.ecl_amount < 200.0
    
    # Stage 2: 35 days past due (SICR)
    ecl_2 = engine.calculate_ecl(ead=10000.0, pd_12m=0.08, pd_lifetime=0.25, days_past_due=35)
    assert ecl_2.stage == 2
    assert ecl_2.horizon == "Lifetime"
    assert ecl_2.ecl_amount > ecl_1.ecl_amount
    
    # Stage 3: 95 days past due (Default)
    ecl_3 = engine.calculate_ecl(ead=10000.0, pd_12m=0.90, pd_lifetime=0.95, days_past_due=95)
    assert ecl_3.stage == 3
    assert ecl_3.pd_applied == 1.0


def test_risk_based_pricing():
    engine = DecisionAndPricingEngine()
    pricing = engine.price_credit(pd_12m=0.03, lgd=0.45)
    
    assert pricing.fair_lending_rate_annual > pricing.cost_of_funds_pct
    assert pricing.raroc_percentage > 0.0

"""
PRINAD - Master Classifier & Decision Engine v3.0
=================================================
Core Credit Risk Classification & Quantitative Underwriting Engine:
- Multi-Model Support: Champion Scorecard, LightGBM, XGBoost, Stacking Ensemble
- Vasicek Macroeconomic Overlay (PIT <-> TTC conversion & IFRS 9 Scenarios)
- Lifetime PD Term Structure & Survival Curves
- IFRS 9 / BACEN 4.966 Staging & ECL Provisioning
- Risk-Based Loan Pricing (RAROC / Hurdle Rate)
- Glass-Box Scorecard Points & Feature Attribution

Author: PRINAD Quantitative Risk Team
Standard: Basel III/IV IRB & BACEN 4.966 / IFRS 9
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
import joblib
from datetime import datetime
from dataclasses import dataclass, asdict, field
import sys

# Local directory imports
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scorecard_woe import RegulatoryScorecard
from vasicek_macro import VasicekMacroEngine
from lifetime_pd import LifetimePDEngine
from decision_pricing_engine import DecisionAndPricingEngine
from feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"


@dataclass
class ClassificationResult:
    """Master Quantitative Credit Risk Result."""
    cpf: str
    prinad_score: int                   # Credit Score (300 to 850)
    pd_12m_pit: float                   # Point-in-Time 12m PD (0 to 1)
    pd_12m_pit_pct: float               # Point-in-Time 12m PD (%)
    pd_ttc: float                       # Through-the-Cycle PD (Basel Anchor)
    pd_lifetime: float                  # Cumulative Lifetime PD (IFRS 9)
    rating: str                         # A1 to DEFAULT
    rating_descricao: str               # "Risco Mínimo", "Risco Moderado", etc.
    cor: str                            # 'verde', 'amarelo', 'laranja', 'vermelho', 'preto'
    estagio_pe: int                     # Stage 1, 2, or 3 (IFRS 9 / BACEN 4966)
    estagio_descricao: str              # "Stage 1 (Normal)", "Stage 2 (SICR)", etc.
    ecl_provision_amount: float         # IFRS 9 Provision ($)
    fair_interest_rate_pct: float       # Risk-Based Lending Rate (% p.a.)
    raroc_pct: float                    # Risk-Adjusted Return on Capital (%)
    acao_sugerida: str                  # Credit policy action
    model_architecture: str             # "Champion_Scorecard", "LightGBM", etc.
    scorecard_points_breakdown: List[Dict[str, Any]] = field(default_factory=list)
    macro_scenarios: Dict[str, Any] = field(default_factory=dict)
    lifetime_curve: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RatingMasterScale:
    """PRINAD Master Rating Scale (11-Grade IRB Calibration)."""
    
    RATING_BANDS = [
        # (Rating, Min PD, Max PD, Label, Color, Action)
        ('A1', 0.000, 0.049, 'Risco Mínimo', 'verde', 'Aprovação automática, taxas prime'),
        ('A2', 0.050, 0.149, 'Risco Muito Baixo', 'verde', 'Aprovação automática com limites ampliados'),
        ('A3', 0.150, 0.249, 'Risco Baixo', 'verde', 'Aprovação simplificada'),
        ('B1', 0.250, 0.349, 'Risco Baixo-Moderado', 'amarelo', 'Análise padrão'),
        ('B2', 0.350, 0.449, 'Risco Moderado', 'amarelo', 'Análise cadastral detalhada'),
        ('B3', 0.450, 0.549, 'Risco Moderado-Alto', 'laranja', 'Exige comprovação adicional de renda'),
        ('C1', 0.550, 0.649, 'Risco Alto', 'vermelho', 'Exige garantias ou avalista'),
        ('C2', 0.650, 0.749, 'Risco Muito Alto', 'vermelho', 'Condições especiais / Taxa de estresse'),
        ('C3', 0.750, 0.849, 'Risco Crítico', 'vermelho', 'Recusa ou garantia real de 150%'),
        ('D',  0.850, 0.949, 'Pré-Default / Iminente', 'preto', 'Recusa, monitoramento preventivo'),
        ('DEFAULT', 0.950, 1.010, 'Default / Inadimplência', 'preto', 'Recusa e cobrança / recuperação')
    ]
    
    @classmethod
    def get_rating(cls, pd_val: float) -> Dict[str, str]:
        pd_val = max(0.0, min(1.0, float(pd_val)))
        for rating, lower, upper, desc, color, action in cls.RATING_BANDS:
            if lower <= pd_val < upper:
                return {
                    'rating': rating,
                    'descricao': desc,
                    'cor': color,
                    'acao_sugerida': action,
                    'faixa': f"{lower*100:.1f}% - {upper*100:.1f}%"
                }
        return {
            'rating': 'DEFAULT',
            'descricao': 'Default',
            'cor': 'preto',
            'acao_sugerida': 'Recusa e cobrança',
            'faixa': '95% - 100%'
        }


class PRINADClassifier:
    """
    Production-Grade Unified PRINAD Credit Risk Classifier v3.0.
    """
    
    def __init__(self, model_type: str = "scorecard"):
        self.model_type = model_type
        self.scorecard_model: Optional[RegulatoryScorecard] = None
        self.lightgbm_model = None
        self.xgboost_model = None
        self.ensemble_model = None
        self.preprocessor = None
        self.feature_engineer = FeatureEngineer()
        
        self.vasicek_engine = VasicekMacroEngine()
        self.lifetime_engine = LifetimePDEngine()
        self.pricing_engine = DecisionAndPricingEngine()
        
        self._load_artifacts()

    def _load_artifacts(self):
        """Load trained model binaries from `artifacts/`."""
        try:
            sc_path = ARTIFACTS_DIR / "scorecard_model.joblib"
            if sc_path.exists():
                self.scorecard_model = joblib.load(sc_path)
                
            lgb_path = ARTIFACTS_DIR / "lightgbm_model.joblib"
            if lgb_path.exists():
                self.lightgbm_model = joblib.load(lgb_path)
                
            xgb_path = ARTIFACTS_DIR / "xgboost_model.joblib"
            if xgb_path.exists():
                self.xgboost_model = joblib.load(xgb_path)
                
            ens_path = ARTIFACTS_DIR / "ensemble_model.joblib"
            if ens_path.exists():
                self.ensemble_model = joblib.load(ens_path)
                
            prep_path = ARTIFACTS_DIR / "preprocessor.joblib"
            if prep_path.exists():
                self.preprocessor = joblib.load(prep_path)
                
            logger.info("PRINAD Classifier artifacts loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load some model artifacts: {e}")

    def is_ready(self) -> bool:
        """Check if models are loaded and ready for inference."""
        return (self.scorecard_model is not None) or (self.lightgbm_model is not None)

    def classify_borrower(
        self,
        borrower_data: Dict[str, Any],
        model_type: Optional[str] = None,
        loan_amount: float = 10000.0,
        asset_class: str = "retail_other"
    ) -> ClassificationResult:
        """
        Full quantitative evaluation of a single loan applicant.
        """
        chosen_model = model_type or self.model_type
        cpf = str(borrower_data.get('CPF', borrower_data.get('cpf', '00000000000')))
        
        # Merge dictionary data if nested
        cad = borrower_data.get('dados_cadastrais', {})
        comp = borrower_data.get('dados_comportamentais', {})
        flat_data = {**borrower_data, **cad, **comp}
        
        # Convert to DataFrame
        df_input = pd.DataFrame([flat_data])
        df_fe = self.feature_engineer.transform(df_input)
        
        # 1. Predict 12-Month Point-in-Time PD
        scorecard_breakdown = []
        credit_score = 600
        
        if chosen_model == "scorecard" and self.scorecard_model:
            pd_12m = float(self.scorecard_model.predict_proba(df_fe)[0])
            credit_score = int(self.scorecard_model.predict_score(df_fe)[0])
            explanation = self.scorecard_model.explain_borrower(df_fe.iloc[0])
            scorecard_breakdown = explanation.get('points_breakdown', [])
            arch_name = "Champion_Regulatory_Scorecard"
        elif chosen_model in ["lightgbm", "xgboost", "ensemble"] and self.preprocessor:
            X_proc = self.preprocessor.transform(df_fe)
            if chosen_model == "lightgbm" and self.lightgbm_model:
                pd_12m = float(self.lightgbm_model.predict_proba(X_proc)[0, 1])
                arch_name = "Challenger_LightGBM"
            elif chosen_model == "xgboost" and self.xgboost_model:
                pd_12m = float(self.xgboost_model.predict_proba(X_proc)[0, 1])
                arch_name = "Challenger_XGBoost"
            elif self.ensemble_model:
                pd_12m = float(self.ensemble_model.predict_proba(X_proc)[0, 1])
                arch_name = "Challenger_Stacking_Ensemble"
            else:
                pd_12m = float(self.scorecard_model.predict_proba(df_fe)[0]) if self.scorecard_model else 0.10
                arch_name = "Scorecard_Fallback"
            # Map PD to Score points
            credit_score = int(np.clip(600 - (20.0 / np.log(2.0)) * np.log(pd_12m / max(1.0 - pd_12m, 1e-6)), 300, 850))
        else:
            # Fallback
            pd_12m = 0.10
            credit_score = 580
            arch_name = "Fallback"
            
        pd_12m = float(np.clip(pd_12m, 0.0005, 0.999))
        
        # 2. Rating & Description
        rating_info = RatingMasterScale.get_rating(pd_12m)
        rating_grade = rating_info['rating']
        
        # 3. Vasicek Macroeconomic Overlay (TTC & IFRS 9 Scenarios)
        macro_eval = self.vasicek_engine.evaluate_ifrs9_scenarios(pd_12m, is_input_ttc=False)
        pd_ttc = float(macro_eval['pd_ttc_anchor'])
        
        macro_clean = {
            'pd_ttc_anchor': float(round(macro_eval.get('pd_ttc_anchor', 0.0), 6)),
            'pd_forward_looking_weighted': float(round(macro_eval.get('pd_forward_looking_weighted', 0.0), 6)),
            'scenarios': [
                {
                    'scenario_name': str(s['scenario_name']),
                    'probability_weight': float(s['probability_weight']),
                    'z_factor': float(s['z_factor']),
                    'conditional_pd': float(s['conditional_pd']),
                    'conditional_pd_pct': float(s['conditional_pd_pct'])
                } for s in macro_eval.get('scenarios', [])
            ]
        }
        
        # 4. Lifetime PD Term Structure (IFRS 9 Curve)
        lt_curve_df = self.lifetime_engine.get_lifetime_curve_from_rating(rating_grade, max_years=5)
        pd_lifetime = float(lt_curve_df['Cumulative_PD'].iloc[-1])
        lt_records = [
            {
                k: int(v) if isinstance(v, (np.integer, np.int64, np.int32))
                else float(v) if isinstance(v, (np.floating, np.float64, np.float32))
                else v
                for k, v in r.items()
            }
            for r in lt_curve_df.to_dict(orient='records')
        ]
        
        # 5. IFRS 9 / BACEN 4.966 ECL Calculation
        dpd = int(flat_data.get('scr_dias_atraso', flat_data.get('max_dias_atraso_12m', 0)) or 0)
        ecl_result = self.pricing_engine.calculate_ecl(
            ead=float(loan_amount),
            pd_12m=pd_12m,
            pd_lifetime=pd_lifetime,
            days_past_due=dpd
        )
        
        # 6. Risk-Based Loan Pricing & RAROC
        pricing_result = self.pricing_engine.price_credit(
            pd_12m=pd_12m,
            asset_class=asset_class
        )
        
        return ClassificationResult(
            cpf=str(cpf),
            prinad_score=int(credit_score),
            pd_12m_pit=float(round(pd_12m, 6)),
            pd_12m_pit_pct=float(round(pd_12m * 100, 2)),
            pd_ttc=float(round(pd_ttc, 6)),
            pd_lifetime=float(round(pd_lifetime, 6)),
            rating=str(rating_grade),
            rating_descricao=str(rating_info['descricao']),
            cor=str(rating_info['cor']),
            estagio_pe=int(ecl_result.stage),
            estagio_descricao=str(ecl_result.stage_name),
            ecl_provision_amount=float(round(ecl_result.ecl_amount, 2)),
            fair_interest_rate_pct=float(pricing_result.fair_lending_rate_annual),
            raroc_pct=float(pricing_result.raroc_percentage),
            acao_sugerida=str(rating_info['acao_sugerida']),
            model_architecture=str(arch_name),
            scorecard_points_breakdown=scorecard_breakdown,
            macro_scenarios=macro_clean,
            lifetime_curve=lt_records
        )

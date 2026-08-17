"""
PRINAD - Quantitative Feature Engineering Module v3.1
=====================================================
Transforms raw cadastral, financial, behavioral, and SCR bureau data into
highly predictive, monotonic, and interpretable credit risk features.

Features Engineered:
1. Financial Leverage & Capacity Ratios (Debt-to-Income, Utilization stress, Free cash flow)
2. Behavioral Delinquency Dynamics (Weighted Arrears Score, Velocity, Severity)
3. Central Bank SCR Bureau Systemic Distress (Overdue-to-Income, Write-off flags)
4. Sociodemographic Stability Metrics (Tenure, Age stability, Residence stability)
5. Non-linear Interaction Terms (Financial Stress Index, Debt-Tenure Interaction)

Author: PRINAD Quantitative Risk Team
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compatible Feature Engineering Transformer for Credit Risk.
    """
    
    OCCUPATION_RISK_MAP = {
        'SERVIDOR PUBLICO': -0.45,
        'APOSENTADO': -0.25,
        'ASSALARIADO': -0.10,
        'EMPRESARIO': 0.18,
        'AUTONOMO': 0.35
    }
    
    EDUCATION_SCORE_MAP = {
        'ANALFABETO': 0,
        'FUNDAM': 1,
        'MEDIO': 2,
        'SUPERIOR': 3,
        'POS': 4
    }
    
    RESIDENCE_RISK_MAP = {
        'PROPRIA': -0.25,
        'FINANCIADA': -0.05,
        'ALUGADA': 0.18,
        'CEDIDA': 0.28
    }
    
    V_COL_DAYS = {
        'v205': 30, 'v210': 60, 'v220': 90, 'v230': 120, 'v240': 150,
        'v245': 180, 'v250': 210, 'v255': 240, 'v260': 270, 'v270': 300,
        'v280': 330, 'v290': 360
    }
    
    def __init__(self):
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None) -> 'FeatureEngineer':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all credit risk transformations."""
        df = X.copy()
        
        # 1. Financial & Leverage Ratios
        renda = np.maximum(df['RENDA_BRUTA'].fillna(1500).values if 'RENDA_BRUTA' in df.columns else np.full(len(df), 1500.0), 1.0)
        df['log_renda_bruta'] = np.log1p(renda)
        
        if 'QT_DEPENDENTES' in df.columns:
            deps = df['QT_DEPENDENTES'].fillna(0).values + 1.0
            df['renda_per_capita'] = renda / deps
            df['log_renda_per_capita'] = np.log1p(df['renda_per_capita'])
            
        if 'RENDA_LIQUIDA' in df.columns:
            r_liq = np.where(df['RENDA_LIQUIDA'].notna(), df['RENDA_LIQUIDA'].values, renda * 0.8)
            df['ratio_liquida_bruta'] = np.clip(r_liq / renda, 0.5, 1.0)
            
        if 'limite_total' in df.columns:
            lim_tot = df['limite_total'].fillna(0).values
            df['limite_to_income_ratio'] = np.clip(lim_tot / renda, 0.0, 20.0)
            
        # 2. Credit Utilization & Debt Stress
        if 'limite_total' in df.columns and 'limite_utilizado' in df.columns:
            lim = np.maximum(df['limite_total'].fillna(1000).values, 1.0)
            used = df['limite_utilizado'].fillna(0).values
            df['taxa_utilizacao_calc'] = np.clip(used / lim, 0.0, 1.5)
            df['is_high_utilization'] = (df['taxa_utilizacao_calc'] > 0.80).astype(int)
            df['is_critical_utilization'] = (df['taxa_utilizacao_calc'] > 0.95).astype(int)
            
        if 'COMP_RENDA' in df.columns:
            comp = df['COMP_RENDA'].fillna(0.3).values
            df['is_severe_debt_burden'] = (comp > 0.50).astype(int)
            df['is_critical_debt_burden'] = (comp > 0.70).astype(int)
            
            # Interaction: Combined Financial Stress Index
            if 'taxa_utilizacao' in df.columns:
                taxa = df['taxa_utilizacao'].fillna(0.3).values
                df['financial_stress_index'] = (comp * 1.5 + taxa * 1.2)
                
        # 3. Behavioral Delinquency Score (Weighted Historical Delays)
        v_cols_present = [v for v in self.V_COL_DAYS.keys() if v in df.columns]
        if v_cols_present:
            score_atraso = np.zeros(len(df))
            total_exposicao = np.zeros(len(df))
            max_dias = np.zeros(len(df))
            recent_score = np.zeros(len(df))
            
            for v, days in self.V_COL_DAYS.items():
                if v in df.columns:
                    val = df[v].fillna(0).values
                    has_val = (val > 0).astype(float)
                    score_atraso += has_val * (days / 30.0)
                    total_exposicao += val
                    max_dias = np.where(val > 0, np.maximum(max_dias, days), max_dias)
                    if days <= 90:
                        recent_score += has_val * 2.0
                    else:
                        recent_score += has_val * 1.0
                        
            df['score_delinquencia_interna'] = score_atraso
            df['total_exposicao_atraso'] = total_exposicao
            df['log_total_exposicao_atraso'] = np.log1p(total_exposicao)
            df['max_dias_atraso_interno'] = max_dias
            df['delinquency_recency_score'] = recent_score
            df['has_internal_delinquency'] = (max_dias > 0).astype(int)
            df['has_severe_internal_delinquency'] = (max_dias >= 90).astype(int)
            
        # 4. SCR Bureau & Systemic Risk Features
        if 'scr_score_risco' in df.columns:
            df['scr_score_num'] = df['scr_score_risco'].fillna(2).astype(float)
            
        if 'scr_dias_atraso' in df.columns:
            scr_dias = df['scr_dias_atraso'].fillna(0).values
            df['has_scr_arrears'] = (scr_dias > 0).astype(int)
            df['has_scr_severe_arrears'] = (scr_dias >= 60).astype(int)
            
        if 'scr_valor_vencido' in df.columns:
            venc = df['scr_valor_vencido'].fillna(0).values
            df['scr_vencido_to_income'] = np.clip(venc / renda, 0.0, 10.0)
            df['log_scr_valor_vencido'] = np.log1p(venc)
            
        if 'scr_tem_prejuizo' in df.columns:
            df['scr_tem_prejuizo_flag'] = df['scr_tem_prejuizo'].fillna(0).astype(int)
            
        # Combined Systemic Distress Indicator
        if 'scr_dias_atraso' in df.columns and 'scr_tem_prejuizo' in df.columns:
            df['has_systemic_distress'] = (
                (df['scr_dias_atraso'].fillna(0) >= 60) | 
                (df['scr_tem_prejuizo'].fillna(0) == 1)
            ).astype(int)
            
        # 5. Categorical Risk Scoring & Demographics
        if 'OCUPACAO' in df.columns:
            df['score_ocupacao'] = df['OCUPACAO'].map(self.OCCUPATION_RISK_MAP).fillna(0.0)
            
        if 'ESCOLARIDADE' in df.columns:
            df['score_escolaridade'] = df['ESCOLARIDADE'].map(self.EDUCATION_SCORE_MAP).fillna(2)
            
        if 'TIPO_RESIDENCIA' in df.columns:
            df['score_residencia'] = df['TIPO_RESIDENCIA'].map(self.RESIDENCE_RISK_MAP).fillna(0.0)
            
        if 'TEMPO_RELAC' in df.columns:
            tempo = df['TEMPO_RELAC'].fillna(12).values
            df['log_tempo_relac'] = np.log1p(np.maximum(tempo, 0))
            df['is_new_client'] = (tempo < 6).astype(int)
            df['is_mature_client'] = (tempo >= 48).astype(int)
            
        if 'IDADE_CLIENTE' in df.columns:
            idade = df['IDADE_CLIENTE'].fillna(35).values
            df['idade_squared'] = (idade / 10.0) ** 2
            df['is_working_age'] = ((idade >= 22) & (idade <= 65)).astype(int)
            
        self.feature_names_ = list(df.columns)
        return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to run feature engineering."""
    fe = FeatureEngineer()
    return fe.fit_transform(df)

"""
PRINAD - Feature Engineering Module
Creates derived features for credit risk modeling.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering class for PRINAD model.
    Creates derived features from raw cadastral and behavioral data.
    """
    
    # Behavioral columns mapping (days delinquent)
    V_COLS_MAPPING = {
        'v205': 30,   # 30 days
        'v210': 60,   # 60 days
        'v220': 90,   # 90 days
        'v230': 120,  # 120 days
        'v240': 150,  # 150 days
        'v245': 180,  # 180 days
        'v250': 210,  # 210 days
        'v255': 240,  # 240 days
        'v260': 270,  # 270 days
        'v270': 300,  # 300 days
        'v280': 330,  # 330 days
        'v290': 360   # 360 days
    }
    
    # Occupation risk scores (higher = more risk)
    OCCUPATION_RISK = {
        'AUTONOMO': 0.3,
        'EMPRESARIO': 0.2,
        'APOSENTADO': -0.1,
        'PENSIONISTA': -0.1,
        'ASSALARIADO': -0.2,
        'SERVIDOR PUBLICO': -0.3,
        'FUNCIONARIO PUBLICO': -0.3,
    }
    
    # Education scores (higher = less risk)
    EDUCATION_SCORE = {
        'ANALFABETO': 0,
        'FUNDAM': 1,
        'FUNDAMENTAL': 1,
        'MEDIO': 2,
        'SUP.': 3,
        'SUPERIOR': 3,
        'POS': 4,
        'POS-GRADUACAO': 4,
    }
    
    def __init__(self):
        self.feature_names_: List[str] = []
    
    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """Fit the feature engineer (learn feature names)."""
        _ = self.transform(df)
        return self
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the data."""
        return self.transform(df)
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw data into engineered features.
        
        Args:
            df: Raw DataFrame with cadastral and behavioral data
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Create all derived features
        df = self._create_financial_features(df)
        df = self._create_behavioral_features(df)
        df = self._create_categorical_scores(df)
        df = self._create_age_features(df)
        df = self._create_relationship_features(df)
        
        # Convert categorical columns to string to avoid fillna issues
        cat_cols = df.select_dtypes(include=['category']).columns
        for col in cat_cols:
            df[col] = df[col].astype(str)
        
        # Store feature names
        self.feature_names_ = list(df.columns)
        
        return df
    
    def _create_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create financial ratio features."""
        
        # Commitment ratio (renda comprometida)
        if 'COMP_RENDA' in df.columns:
            # Handle None values before pd.cut
            comp_renda_safe = df['COMP_RENDA'].fillna(0.5)  # Default to medium
            df['comp_renda_faixa'] = pd.cut(
                comp_renda_safe,
                bins=[-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf],
                labels=['muito_baixo', 'baixo', 'medio', 'alto', 'muito_alto']
            )
        
        # Income per dependent
        if 'RENDA_LIQUIDA' in df.columns and 'QT_DEPENDENTES' in df.columns:
            df['renda_por_dependente'] = df['RENDA_LIQUIDA'] / (df['QT_DEPENDENTES'] + 1)
            
            # Log transform for normalization
            df['log_renda_liquida'] = np.log1p(df['RENDA_LIQUIDA'].clip(lower=0))
            df['log_renda_por_dependente'] = np.log1p(df['renda_por_dependente'].clip(lower=0))
        
        # Net/Gross income ratio
        if 'RENDA_LIQUIDA' in df.columns and 'RENDA_BRUTA' in df.columns:
            df['ratio_liquida_bruta'] = np.where(
                df['RENDA_BRUTA'] > 0,
                df['RENDA_LIQUIDA'] / df['RENDA_BRUTA'],
                0.8  # Default
            )
        
        return df
    
    def _create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create behavioral score features from v-columns."""
        
        v_cols = [col for col in self.V_COLS_MAPPING.keys() if col in df.columns]
        
        if not v_cols:
            return df
        
        # Maximum delinquency indicator
        df['max_atraso_dias'] = 0
        for col, days in sorted(self.V_COLS_MAPPING.items(), key=lambda x: x[1], reverse=True):
            if col in df.columns:
                df.loc[df[col] > 0, 'max_atraso_dias'] = days
        
        # Delinquency score (weighted sum)
        df['score_atraso'] = 0
        for col, days in self.V_COLS_MAPPING.items():
            if col in df.columns:
                weight = days / 360  # Normalize weight
                df['score_atraso'] += df[col] * weight
        
        # Short-term delinquency (<=120 days)
        short_term_cols = ['v205', 'v210', 'v220', 'v230']
        df['tem_atraso_curto'] = df[[c for c in short_term_cols if c in df.columns]].sum(axis=1) > 0
        
        # Long-term delinquency (>120 days)
        long_term_cols = ['v240', 'v245', 'v250', 'v255', 'v260', 'v270', 'v280', 'v290']
        df['tem_atraso_longo'] = df[[c for c in long_term_cols if c in df.columns]].sum(axis=1) > 0
        
        # Severe delinquency (>180 days)
        severe_cols = ['v245', 'v250', 'v255', 'v260', 'v270', 'v280', 'v290']
        df['tem_inadimplencia'] = df[[c for c in severe_cols if c in df.columns]].sum(axis=1) > 0
        
        # Total delinquency exposure
        df['total_exposicao_atraso'] = df[v_cols].sum(axis=1)
        df['log_exposicao_atraso'] = np.log1p(df['total_exposicao_atraso'].clip(lower=0))
        
        # Severity categories
        df['severidade_atraso'] = 'sem_atraso'
        df.loc[df['tem_atraso_curto'], 'severidade_atraso'] = 'curto_prazo'
        df.loc[df['tem_atraso_longo'], 'severidade_atraso'] = 'longo_prazo'
        df.loc[df['tem_inadimplencia'], 'severidade_atraso'] = 'inadimplente'
        
        return df
    
    def _create_categorical_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create risk scores from categorical variables."""
        
        # Occupation risk score
        if 'OCUPACAO' in df.columns:
            df['score_ocupacao'] = df['OCUPACAO'].map(self.OCCUPATION_RISK).fillna(0)
        
        # Education score
        if 'ESCOLARIDADE' in df.columns:
            df['score_escolaridade'] = df['ESCOLARIDADE'].map(self.EDUCATION_SCORE).fillna(1)
        
        # Vehicle indicator
        if 'POSSUI_VEICULO' in df.columns:
            df['tem_veiculo'] = df['POSSUI_VEICULO'].str.upper().isin(['SIM', 'S', '1', 'TRUE']).astype(int)
        
        # Residence type score
        if 'TIPO_RESIDENCIA' in df.columns:
            residence_score = {
                'PROPRIA': 0,
                'PROPRIA QUITADA': 0,
                'PROPRIA FINANCIADA': 0.1,
                'ALUGADA': 0.2,
                'CEDIDA': 0.3,
                'OUTROS': 0.3
            }
            df['score_residencia'] = df['TIPO_RESIDENCIA'].map(residence_score).fillna(0.2)
        
        # Marital status score
        if 'ESTADO_CIVIL' in df.columns:
            marital_score = {
                'CASADO': -0.1,
                'CASADO(A)': -0.1,
                'SOLTEIRO': 0.1,
                'SOLTEIRO(A)': 0.1,
                'DIVORCIADO': 0.05,
                'DIVORCIADO(A)': 0.05,
                'VIUVO': 0,
                'VIUVO(A)': 0,
            }
            df['score_estado_civil'] = df['ESTADO_CIVIL'].map(marital_score).fillna(0)
        
        # Portability indicator
        if 'PORTABILIDADE' in df.columns:
            df['is_portado'] = df['PORTABILIDADE'].str.contains('PORTADO', case=False, na=False).astype(int)
        
        return df
    
    def _create_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create age-related features."""
        
        if 'IDADE_CLIENTE' not in df.columns:
            return df
        
        # Handle None values before pd.cut
        idade_safe = df['IDADE_CLIENTE'].fillna(35)  # Default to median age
        
        # Age bins
        df['faixa_etaria'] = pd.cut(
            idade_safe,
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=['jovem', 'adulto_jovem', 'adulto', 'maduro', 'senior', 'idoso']
        )
        
        # Quadratic age (captures non-linear relationship with risk)
        df['idade_squared'] = idade_safe ** 2
        
        # Working age indicator
        df['em_idade_ativa'] = ((idade_safe >= 18) & (idade_safe <= 65)).astype(int)
        
        return df
    
    def _create_relationship_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create relationship tenure features."""
        
        if 'TEMPO_RELAC' not in df.columns:
            return df
        
        # Handle None values before pd.cut
        tempo_safe = df['TEMPO_RELAC'].fillna(0)  # Default to new client
        
        # Relationship bins
        df['faixa_relacionamento'] = pd.cut(
            tempo_safe,
            bins=[-np.inf, 6, 12, 24, 60, 120, np.inf],
            labels=['novo', 'recente', 'estabelecido', 'consolidado', 'longo_prazo', 'fidelizado']
        )
        
        # Log transform
        df['log_tempo_relac'] = np.log1p(tempo_safe.clip(lower=0))
        
        # New client indicator (< 6 months)
        df['cliente_novo'] = (tempo_safe < 6).astype(int)
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Return list of engineered feature names."""
        return self.feature_names_.copy()


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to create all engineered features.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    fe = FeatureEngineer()
    return fe.fit_transform(df)


if __name__ == "__main__":
    # Test feature engineering
    import sys
    sys.path.append(str(Path(__file__).parent))
    from data_pipeline import load_and_prepare_full_dataset
    
    df_full, X, y = load_and_prepare_full_dataset()
    
    print(f"\nOriginal features: {len(X.columns)}")
    
    fe = FeatureEngineer()
    X_engineered = fe.fit_transform(X)
    
    print(f"Engineered features: {len(X_engineered.columns)}")
    print(f"\nNew features: {set(X_engineered.columns) - set(X.columns)}")

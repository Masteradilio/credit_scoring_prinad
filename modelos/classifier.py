"""
PRINAD - Classifier Module v2.0
Complete classification pipeline for credit risk scoring.
Legacy demonstrative implementation; regulatory conformity is not evaluated.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging
import joblib
from datetime import datetime
from dataclasses import dataclass, asdict
import sys

# Add paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
SRC_DIR = Path(__file__).parent

# Ensure src is in path for local imports
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Ensure project root is in path for shared imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local imports  
from historical_penalty import HistoricalPenaltyCalculator, HistoricalAnalysis

# BACEN 4966 imports from shared utils
try:
    from shared.utils import (
        get_rating_from_prinad as get_rating_bacen,
        calcular_pd_por_rating,
        PD_POR_RATING
    )
    BACEN_4966_AVAILABLE = True
except ImportError:
    BACEN_4966_AVAILABLE = False

logger = logging.getLogger(__name__)

# Paths
MODELO_DIR = BASE_DIR / "artefatos"


@dataclass
class ClassificationResult:
    """Result of a demonstrative PRINAD classification using synthetic inputs."""
    cpf: str
    prinad: float
    rating: str
    rating_descricao: str
    cor: str
    pd_base: float
    pd_12m: float              # BACEN 4966: 12-month PD
    pd_lifetime: float         # BACEN 4966: Lifetime PD
    penalidade_historica: float
    peso_atual: float
    peso_historico: float
    acao_sugerida: str
    explicacao_shap: List[Dict[str, Any]]  # SHAP explanation for ML model (50%)
    timestamp: str
    model_version: str
    estagio_pe: int = 1        # Estágio de Perda Esperada / IFRS 9 Stage (1, 2, or 3)
    explicacao_completa: Optional[Dict[str, Any]] = None  # Full explanation including penalties
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RatingMapper:
    """Maps PRINAD scores to ratings with custom bands."""
    
    RATING_CONFIG = [
        # (rating, lower, upper, description, color, action)
        ('A1', 0, 5, 'Risco Mínimo', 'verde', 'Aprovação automática, melhores taxas'),
        ('A2', 5, 15, 'Risco Muito Baixo', 'verde', 'Aprovação automática'),
        ('A3', 15, 25, 'Risco Baixo', 'verde', 'Aprovação com análise simplificada'),
        ('B1', 25, 35, 'Risco Baixo-Moderado', 'amarelo', 'Análise padrão'),
        ('B2', 35, 45, 'Risco Moderado', 'amarelo', 'Análise detalhada'),
        ('B3', 45, 55, 'Risco Moderado-Alto', 'laranja', 'Análise rigorosa, possíveis garantias'),
        ('C1', 55, 65, 'Risco Alto', 'vermelho', 'Exige garantias ou fiador'),
        ('C2', 65, 75, 'Risco Muito Alto', 'vermelho', 'Negação ou condições especiais'),
        ('C3', 75, 85, 'Risco Crítico', 'vermelho', 'Negação, exige garantias sólidas'),
        ('D', 85, 95, 'Pré-Default', 'preto', 'Negação, monitoramento intensivo'),
        ('DEFAULT', 95, 100.01, 'Default', 'preto', 'Negação, encaminhar para cobrança'),
    ]
    
    @classmethod
    def get_rating(cls, prinad: float) -> Dict[str, str]:
        """Get rating information for a PRINAD score."""
        prinad = max(0, min(100, prinad))
        
        for rating, lower, upper, desc, color, action in cls.RATING_CONFIG:
            if lower <= prinad < upper:
                return {
                    'rating': rating,
                    'descricao': desc,
                    'cor': color,
                    'acao_sugerida': action,
                    'faixa': f'{lower}% - {upper}%'
                }
        
        # Default to worst rating
        return {
            'rating': 'D',
            'descricao': 'Default/Iminente',
            'cor': 'preto',
            'acao_sugerida': 'Negação, encaminhar para cobrança',
            'faixa': '90% - 100%'
        }


class PRINADClassifier:
    """
    Complete PRINAD classification pipeline v2.0.
    Combines ML model prediction with historical penalty.
    Legacy demonstrative classifier; regulatory conformity is not evaluated.
    """
    
    MODEL_VERSION = "2.0.0"
    PESO_ATUAL = 0.5
    PESO_HISTORICO = 0.5
    
    def __init__(self, modelo_dir: Optional[Path] = None):
        """
        Initialize the classifier.
        
        Args:
            modelo_dir: Path to model artifacts directory
        """
        self.modelo_dir = modelo_dir or MODELO_DIR
        self.model = None
        self.preprocessor = None
        self.feature_engineer = None
        self.shap_explainer = None
        self.historical_calculator = HistoricalPenaltyCalculator(
            forgiveness_months=6,
            max_penalty_internal=0.75,
            max_penalty_external=0.75
        )
        
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model artifacts from disk."""
        # Try to load artifacts
        try:
            # Model
            if (MODELO_DIR / "ensemble_model.joblib").exists():
                model_data = joblib.load(MODELO_DIR / "ensemble_model.joblib")
                self.model = model_data.get('model')
            else:
                logger.warning(f"Model artifact not found at {MODELO_DIR}")
                
            # Preprocessor
            if (MODELO_DIR / "preprocessor.joblib").exists():
                preproc_data = joblib.load(MODELO_DIR / "preprocessor.joblib")
                self.preprocessor = preproc_data.get('preprocessor')
                self.feature_engineer = preproc_data.get('feature_engineer')
                self.feature_names = preproc_data.get('feature_names')
            else:
                logger.warning(f"Preprocessor artifact not found at {MODELO_DIR}")
                
            # SHAP
            if (MODELO_DIR / "shap_explainer.joblib").exists():
                self.shap_explainer = joblib.load(MODELO_DIR / "shap_explainer.joblib")
            else:
                logger.warning(f"SHAP artifact not found at {MODELO_DIR}")

        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
            # Don't raise, allow heuristic fallback
            
    def is_ready(self) -> bool:
        """Check if model is ready for inference."""
        # For testing purposes, if artifacts are missing, we might want to simulate readiness 
        # or fail gracefully. For now, let's be strict but allow fallback if designed.
        return self.model is not None and self.preprocessor is not None
    
    def classify(self, client_data: Dict[str, Any], include_shap: bool = False) -> ClassificationResult:
        """
        Classify a single synthetic client for demonstration.
        
        Args:
            client_data: Dictionary with client data
                Required keys: dados_cadastrais, dados_comportamentais
                Optional: cpf
            include_shap: If True, generates SHAP explanation (adds latency)
                
        Returns:
            ClassificationResult with full classification details including
            pd_12m, pd_lifetime, and estagio_pe (IFRS 9)
        """
        cpf = client_data.get('cpf', 'unknown')
        
        # Extract data components
        dados_cadastrais = client_data.get('dados_cadastrais', {})
        dados_comportamentais = client_data.get('dados_comportamentais', {})
        
        # Merge into single dict for feature engineering
        merged_data = {**dados_cadastrais, **dados_comportamentais}
        
        # Convert to DataFrame for prediction
        df = pd.DataFrame([merged_data])
        
        # Get PD Base from model or heuristic
        pd_base = self._predict_pd_base(df, dados_comportamentais)
        
        # Calculate historical penalty
        prinad_final, historical_analysis = self.historical_calculator.apply_penalty(
            pd_base, dados_comportamentais
        )
        
        # Calculate BACEN 4966 calibrated PD (12m and lifetime)
        if BACEN_4966_AVAILABLE:
            rating_bacen = get_rating_bacen(prinad_final)
            pd_result = calcular_pd_por_rating(prinad_final, rating_bacen)
            pd_12m = pd_result['pd_12m']
            pd_lifetime = pd_result['pd_lifetime']
        else:
            # Simple approximation if BACEN functions not available
            # PD 12m = PRINAD converted to probability (capped at 100%)
            pd_12m = min(prinad_final / 100.0, 1.0)
            
            # PD Lifetime using survival probability formula: 1 - (1 - PD_12m)^n
            # Assuming average maturity of 5 years
            maturity_years = 5
            pd_lifetime = 1 - ((1 - pd_12m) ** maturity_years)
            pd_lifetime = min(pd_lifetime, 1.0)  # Cap at 100%
        
        # Determine IFRS 9 Stage / Estágio PE
        dias_atraso = int(dados_comportamentais.get('scr_dias_atraso', 0) or 0)
        if dias_atraso >= 90 or prinad_final >= 85:
            estagio_pe = 3
            estagio_justificativa = f"Stage 3 (Default): dias_atraso={dias_atraso} ou PRINAD={prinad_final:.1f}%>=85%"
        elif dias_atraso >= 30 or prinad_final >= 50:
            estagio_pe = 2
            estagio_justificativa = f"Stage 2 (Risco Aumentado): dias_atraso={dias_atraso}>=30 ou PRINAD={prinad_final:.1f}%>=50%"
        else:
            estagio_pe = 1
            estagio_justificativa = f"Stage 1 (Normal): dias_atraso={dias_atraso}<30 e PRINAD={prinad_final:.1f}%<50%"
        
        # Get rating
        rating_info = RatingMapper.get_rating(prinad_final)
        
        # Get SHAP explanation (enabled only when include_shap=True)
        shap_explanation = self._get_shap_explanation(df, enabled=include_shap)
        
        # Build complete explanation
        explicacao_completa = {
            # Composição do PRINAD
            'composicao_prinad': {
                'pd_modelo_ml': {
                    'valor': round(pd_base, 2),
                    'peso': f'{self.PESO_ATUAL*100:.0f}%',
                    'contribuicao': round(pd_base * self.PESO_ATUAL, 2),
                    'descricao': 'Probabilidade base calculada pelo modelo ensemble (XGBoost)',
                    'explicacao_shap': shap_explanation,
                    'nota_shap': 'Top features que mais influenciaram a predição do modelo ML (valores positivos aumentam o risco, negativos diminuem)'
                },
                'penalidade_interna': {
                    'valor': round(historical_analysis.penalidade_interna * 100, 2),
                    'peso': '25%',
                    'contribuicao': round(historical_analysis.penalidade_interna * 100 * 0.25, 2),
                    'meses_desde_atraso': historical_analysis.meses_desde_ultimo_atraso_interno,
                    'descricao': 'Penalidade baseada no histórico interno de atrasos (v-columns)',
                    'detalhes': historical_analysis.detalhes.get('interno', {})
                },
                'penalidade_externa': {
                    'valor': round(historical_analysis.penalidade_externa * 100, 2),
                    'peso': '25%',
                    'contribuicao': round(historical_analysis.penalidade_externa * 100 * 0.25, 2),
                    'meses_desde_atraso': historical_analysis.meses_desde_ultimo_atraso_externo,
                    'descricao': 'Penalidade baseada no histórico SCR/BACEN',
                    'detalhes': historical_analysis.detalhes.get('externo', {})
                },
                'prinad_final': round(prinad_final, 2),
                'formula': 'PRINAD = (PD_ML × 50%) + (Pen.Interna × 25%) + (Pen.Externa × 25%)'
            },
            # Justificativa do Rating
            'rating': {
                'valor': rating_info['rating'],
                'faixa_prinad': f"{rating_info.get('faixa_min', 0):.0f}% - {rating_info.get('faixa_max', 100):.0f}%",
                'descricao': rating_info['descricao'],
                'justificativa': f"PRINAD {prinad_final:.1f}% está na faixa do rating {rating_info['rating']}"
            },
            # Justificativa do PD
            'pd_calibrado': {
                'pd_12m': {
                    'valor': round(pd_12m, 6),
                    'percentual': f'{pd_12m*100:.4f}%',
                    'uso': 'Cálculo de ECL para Stage 1',
                    'justificativa': f"PD 12m calibrado para rating {rating_info['rating']} conforme tabela BACEN 4966"
                },
                'pd_lifetime': {
                    'valor': round(pd_lifetime, 6),
                    'percentual': f'{pd_lifetime*100:.4f}%',
                    'uso': 'Cálculo de ECL para Stage 2/3',
                    'justificativa': f"PD Lifetime = PD_12m × multiplicador do rating {rating_info['rating']}"
                }
            },
            # Justificativa do Estágio PE
            'estagio_pe': {
                'valor': estagio_pe,
                'nome': ['', 'Normal', 'Risco Aumentado', 'Default'][estagio_pe],
                'ecl_horizonte': '12 meses' if estagio_pe == 1 else 'Lifetime',
                'pd_aplicavel': 'pd_12m' if estagio_pe == 1 else 'pd_lifetime',
                'justificativa': estagio_justificativa,
                'criterios': {
                    'dias_atraso': dias_atraso,
                    'prinad': round(prinad_final, 2),
                    'limiar_stage_2': 'dias_atraso>=30 OU prinad>=50%',
                    'limiar_stage_3': 'dias_atraso>=90 OU prinad>=85%'
                }
            },
            # Elegibilidade para perdão
            'elegibilidade_perdao': {
                'elegivel': historical_analysis.elegivel_perdao,
                'nivel_delinquencia': historical_analysis.nivel_delinquencia.value,
                'requisitos': 'Ambos (interno E externo) limpos por 6+ meses consecutivos'
            }
        }
        
        # Build result
        result = ClassificationResult(
            cpf=cpf,
            prinad=round(prinad_final, 2),
            rating=rating_info['rating'],
            rating_descricao=rating_info['descricao'],
            cor=rating_info['cor'],
            pd_base=round(pd_base, 2),
            pd_12m=round(pd_12m, 6),
            pd_lifetime=round(pd_lifetime, 6),
            penalidade_historica=round(historical_analysis.penalidade_total, 2),
            peso_atual=self.PESO_ATUAL,
            peso_historico=self.PESO_HISTORICO,
            acao_sugerida=rating_info['acao_sugerida'],
            explicacao_shap=shap_explanation,
            timestamp=datetime.now().isoformat(),
            model_version=self.MODEL_VERSION,
            estagio_pe=estagio_pe,
            explicacao_completa=explicacao_completa
        )
        
        return result
    
    def _predict_pd_base(self, df: pd.DataFrame, dados_comportamentais: Dict[str, Any] = None) -> float:
        """
        Get base PD from ML model or heuristic fallback.
        
        Args:
            df: DataFrame with client data
            dados_comportamentais: Behavioral data for heuristic calculation
            
        Returns:
            Base PD as percentage (0-100)
        """
        # Try ML model first
        if self.model is not None and self.preprocessor is not None:
            try:
                # Ensure all required base columns exist with defaults
                df_prepared = self._ensure_base_columns(df)
                
                # Apply feature engineering
                if self.feature_engineer:
                    df_engineered = self.feature_engineer.transform(df_prepared)
                else:
                    df_engineered = df_prepared
                
                # Fill any remaining NaN values
                df_engineered = df_engineered.fillna(0)
                
                # Preprocess
                X_processed = self.preprocessor.transform(df_engineered)
                
                # Predict probability
                proba = self.model.predict_proba(X_processed)[0, 1]
                
                # Convert to percentage
                pd_base = proba * 100
                
                # Apply minimum floor of 0.5% - no client has zero risk
                pd_base = max(0.5, pd_base)
                
                return pd_base
                
            except Exception as e:
                logger.warning(f"ML model prediction failed, using heuristic: {e}")
        
        # Fallback to heuristic if ML model not available
        return self._calculate_pd_heuristic(df, dados_comportamentais)
    
    def _calculate_pd_heuristic(self, df: pd.DataFrame, dados_comportamentais: Dict[str, Any] = None) -> float:
        """
        Calculate PD using heuristic when ML model not available.
        
        Based on:
        - SCR score (30% weight)
        - Days in arrears (30% weight)
        - Income commitment (20% weight)
        - Relationship time (10% weight)
        - Age (10% weight)
        """
        pd_base = 5.0  # Start with low risk
        
        if dados_comportamentais is None:
            dados_comportamentais = {}
        
        # Extract values from df or dados_comportamentais
        row = df.iloc[0] if len(df) > 0 else {}
        
        # 1. SCR Score contribution (30%)
        scr_score = int(row.get('scr_score_risco', dados_comportamentais.get('scr_score_risco', 0)) or 0)
        # SCR score 0-10 where higher = worse
        scr_contrib = scr_score * 3.0  # 0-30%
        
        # 2. Days in arrears contribution (30%)
        dias_atraso = max(
            int(row.get('scr_dias_atraso', 0) or 0),
            int(dados_comportamentais.get('scr_dias_atraso', 0) or 0)
        )
        if dias_atraso >= 90:
            arrears_contrib = 30.0
        elif dias_atraso >= 60:
            arrears_contrib = 20.0
        elif dias_atraso >= 30:
            arrears_contrib = 10.0
        elif dias_atraso > 0:
            arrears_contrib = 5.0
        else:
            arrears_contrib = 0.0
        
        # 3. Income commitment (20%)
        comp_renda = float(row.get('COMP_RENDA', row.get('comprometimento_renda', 0.3)) or 0.3)
        if comp_renda > 0.7:
            commitment_contrib = 20.0
        elif comp_renda > 0.5:
            commitment_contrib = 10.0
        elif comp_renda > 0.3:
            commitment_contrib = 5.0
        else:
            commitment_contrib = 0.0
        
        # 4. Relationship time (10%) - longer = better
        tempo_relac = float(row.get('TEMPO_RELAC', 12) or 12)
        if tempo_relac >= 48:
            relac_contrib = -5.0  # Bonus for long relationship
        elif tempo_relac >= 24:
            relac_contrib = 0.0
        elif tempo_relac >= 12:
            relac_contrib = 5.0
        else:
            relac_contrib = 10.0
        
        # 5. Age (10%)
        idade = int(row.get('IDADE_CLIENTE', 35) or 35)
        if idade < 25 or idade > 70:
            age_contrib = 10.0
        elif idade < 30 or idade > 60:
            age_contrib = 5.0
        else:
            age_contrib = 0.0
        
        # 6. Prejudice (adds 20% if present)
        tem_prejuizo = int(dados_comportamentais.get('scr_tem_prejuizo', 0) or 0)
        prejuizo_contrib = 20.0 if tem_prejuizo else 0.0
        
        # Combine
        pd_base = pd_base + scr_contrib + arrears_contrib + commitment_contrib + relac_contrib + age_contrib + prejuizo_contrib
        
        # Clamp to 0.5-100
        pd_base = max(0.5, min(100.0, pd_base))
        
        return pd_base
    
    def _ensure_base_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required base columns exist with sensible defaults."""
        
        df = df.copy()
        
        # Cadastral columns with defaults
        cadastral_defaults = {
            'IDADE_CLIENTE': 35,
            'RENDA_BRUTA': 3000.0,
            'RENDA_LIQUIDA': 2500.0,
            'OCUPACAO': 'ASSALARIADO',
            'ESCOLARIDADE': 'MEDIO',
            'ESTADO_CIVIL': 'SOLTEIRO',
            'QT_DEPENDENTES': 0,
            'TEMPO_RELAC': 12.0,
            'TIPO_RESIDENCIA': 'ALUGADA',
            'POSSUI_VEICULO': 'NAO',
            'PORTABILIDADE': 'NAO PORTADO',
            'COMP_RENDA': 0.3,
            'QT_PRODUTOS': 1
        }
        
        # V-columns (behavioral) - all zero means no delinquency
        v_cols = ['v205', 'v210', 'v220', 'v230', 'v240', 'v245', 
                  'v250', 'v255', 'v260', 'v270', 'v280', 'v290']
        
        # SCR columns with safe defaults (good standing)
        scr_defaults = {
            'scr_classificacao_risco': 'A',
            'scr_dias_atraso': 0,
            'scr_lim_credito': 5000.0,
            'scr_lim_utilizado': 1000.0,
            'scr_valor_vencer': 1000.0,
            'scr_valor_vencido': 0.0,
            'scr_valor_prejuizo': 0.0,
            'scr_qtd_instituicoes': 1,
            'scr_qtd_operacoes': 2,
            'scr_modalidade_principal': 'CARTAO',
            'scr_taxa_utilizacao': 0.2,
            'scr_score_risco': 1,
            'scr_tem_prejuizo': 0,
            'scr_faixa_atraso': '0',
            'scr_exposicao_total': 2000.0,
            'scr_ratio_vencido': 0.0
        }
        
        # Apply cadastral defaults
        for col, default in cadastral_defaults.items():
            if col not in df.columns:
                df[col] = default
            else:
                df[col] = df[col].fillna(default)
        
        # Apply v-column defaults (0 = no delinquency)
        for col in v_cols:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(0.0)
        
        # Apply SCR defaults
        for col, default in scr_defaults.items():
            if col not in df.columns:
                df[col] = default
            else:
                df[col] = df[col].fillna(default)
        
        return df
    
    def _get_shap_explanation(self, df: pd.DataFrame, max_features: int = 5, enabled: bool = False) -> List[Dict[str, Any]]:
        """
        Generate SHAP explanation for the prediction.
        
        Args:
            df: DataFrame with client data
            max_features: Number of top features to return
            enabled: If False, returns empty list for performance (SHAP adds ~5s latency)
        """
        
        # SHAP is disabled by default for performance
        if not enabled:
            return []
        
        # Check if SHAP explainer is available
        if self.shap_explainer is None:
            logger.warning("SHAP explainer not loaded")
            return []
        
        # Check if feature_names is available
        if self.feature_names is None:
            logger.warning("Feature names not loaded, cannot generate SHAP")
            return [{'feature': 'missing_feature_names', 'contribuicao': 0, 'direcao': 'erro'}]
        
        try:
            # For SHAP, we skip feature engineering since the dataset should already
            # have all required features pre-computed (from data_consolidator_prinad.py)
            df_prepared = df.copy()
            
            # Create a DataFrame with all required features (fill missing with 0)
            X_data = {}
            for feature in self.feature_names:
                if feature in df_prepared.columns:
                    X_data[feature] = df_prepared[feature].values
                else:
                    X_data[feature] = [0] * len(df_prepared)
            
            X_features = pd.DataFrame(X_data)[self.feature_names].fillna(0)
            
            # Process through preprocessor
            X_processed = self.preprocessor.transform(X_features)
            
            # Get SHAP values
            shap_values = self.shap_explainer(X_processed)
            
            # Get feature names from preprocessor
            try:
                transformed_feature_names = self.preprocessor.get_feature_names_out().tolist()
            except:
                transformed_feature_names = [f'feature_{i}' for i in range(X_processed.shape[1])]
            
            # Extract values - handle different SHAP return formats
            if hasattr(shap_values, 'values'):
                values = shap_values.values
            else:
                values = shap_values
            
            # Flatten if needed (take first row for single prediction)
            if len(values.shape) > 1:
                values = values[0]
            
            # Ensure it's a 1D array
            values = np.array(values).flatten()
            
            # Sort by importance
            importance = list(zip(transformed_feature_names, values))
            importance.sort(key=lambda x: abs(float(x[1])), reverse=True)
            
            # Build explanation
            explanation = []
            for feat, val in importance[:max_features]:
                # Clean feature name (remove prefixes like 'num__', 'cat__')
                clean_name = feat
                for prefix in ['num__', 'cat__', 'bool__']:
                    if clean_name.startswith(prefix):
                        clean_name = clean_name[len(prefix):]
                        break
                
                explanation.append({
                    'feature': clean_name,
                    'contribuicao': round(float(val) * 100, 2),  # Convert to percentage points
                    'direcao': 'aumenta_risco' if val > 0 else 'reduz_risco'
                })
            
            logger.info(f"SHAP explanation generated with {len(explanation)} features")
            return explanation
            
        except Exception as e:
            logger.warning(f"Could not generate SHAP explanation: {e}")
            # Return a placeholder explanation indicating the failure
            return [{
                'feature': 'shap_error',
                'contribuicao': 0,
                'direcao': 'erro',
                'mensagem': f'Erro ao calcular SHAP: {str(e)[:100]}'
            }]
    
    def classify_batch(self, clients: List[Dict[str, Any]]) -> List[ClassificationResult]:
        """
        Classify multiple clients.
        
        Args:
            clients: List of client data dictionaries
            
        Returns:
            List of ClassificationResults
        """
        results = []
        for client_data in clients:
            try:
                result = self.classify(client_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Error classifying client: {e}")
                # Create error result
                results.append(ClassificationResult(
                    cpf=client_data.get('cpf', 'unknown'),
                    prinad=50.0,
                    rating='B3',
                    rating_descricao='Erro na classificação',
                    cor='laranja',
                    pd_base=50.0,
                    pd_12m=0.10,       # 10% annual PD
                    pd_lifetime=0.30,  # 30% lifetime PD
                    penalidade_historica=0.0,
                    peso_atual=0.5,
                    peso_historico=0.5,
                    acao_sugerida='Reprocessar manualmente',
                    explicacao_shap=[],
                    timestamp=datetime.now().isoformat(),
                    model_version=self.MODEL_VERSION,
                    stage=2  # Conservative middle stage
                ))
        
        return results


# Global classifier instance
_classifier: Optional[PRINADClassifier] = None


def get_classifier() -> PRINADClassifier:
    """Get or create global classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = PRINADClassifier()
    return _classifier


def classify_client(cpf: str, dados_cadastrais: Dict[str, Any], 
                    dados_comportamentais: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to classify a client.
    
    Args:
        cpf: Client CPF
        dados_cadastrais: Cadastral data dictionary
        dados_comportamentais: Behavioral data dictionary
        
    Returns:
        Classification result as dictionary
    """
    classifier = get_classifier()
    result = classifier.classify({
        'cpf': cpf,
        'dados_cadastrais': dados_cadastrais,
        'dados_comportamentais': dados_comportamentais
    })
    return result.to_dict()


if __name__ == "__main__":
    # Test classification
    logging.basicConfig(level=logging.INFO)
    
    # Test data
    test_client = {
        'cpf': '12345678901',
        'dados_cadastrais': {
            'IDADE_CLIENTE': 35,
            'RENDA_BRUTA': 5000.0,
            'RENDA_LIQUIDA': 4200.0,
            'OCUPACAO': 'ASSALARIADO',
            'ESCOLARIDADE': 'SUPERIOR',
            'ESTADO_CIVIL': 'CASADO',
            'QT_DEPENDENTES': 2,
            'TEMPO_RELAC': 48.0,
            'TIPO_RESIDENCIA': 'PROPRIA',
            'POSSUI_VEICULO': 'SIM',
            'PORTABILIDADE': 'NAO PORTADO',
            'COMP_RENDA': 0.35
        },
        'dados_comportamentais': {
            'v205': 0.0,
            'v210': 0.0,
            'v220': 0.0,
            'v230': 0.0,
            'v240': 0.0,
            'v245': 0.0,
            'v250': 0.0,
            'v255': 0.0,
            'v260': 0.0,
            'v270': 0.0,
            'v280': 0.0,
            'v290': 0.0
        }
    }
    
    print("\n" + "="*60)
    print("PRINAD CLASSIFICATION TEST")
    print("="*60)
    
    try:
        classifier = PRINADClassifier()
        
        if classifier.is_ready():
            result = classifier.classify(test_client)
            
            print(f"\nCPF: {result.cpf}")
            print(f"PRINAD: {result.prinad}%")
            print(f"Rating: {result.rating} - {result.rating_descricao}")
            print(f"Cor: {result.cor}")
            print(f"PD Base: {result.pd_base}%")
            print(f"Penalidade Histórica: {result.penalidade_historica}")
            print(f"Ação Sugerida: {result.acao_sugerida}")
            print(f"\nExplicação SHAP:")
            for exp in result.explicacao_shap:
                print(f"  - {exp['feature']}: {exp['contribuicao']:+.2f}% ({exp['direcao']})")
        else:
            print("Classifier not ready. Train the model first.")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Run train_model.py first to create model artifacts.")

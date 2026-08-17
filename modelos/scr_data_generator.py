"""
Mock SCR Data Generator
Generates realistic SCR (Sistema de Informações de Crédito) data for each client.

The data is generated consistently (not randomly) based on:
1. Client's internal default status (target variable)
2. Client's payment behavior patterns
3. Correlation with existing features

This ensures the mock data has predictive value for model training.
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SCRDataGenerator:
    """
    Generates mock SCR data based on client characteristics and default status.
    
    The generation follows realistic patterns:
    - Clients who default internally (target=1) have higher probability of SCR issues
    - However, some clients may have external issues without internal issues (and vice-versa)
    - The correlation is not perfect to avoid data leakage
    """
    
    # Rating distribution probabilities by default status
    # For defaulters: higher probability of bad ratings
    RATING_PROBS_DEFAULT = {
        'AA': 0.02, 'A': 0.05, 'B': 0.08, 'C': 0.15,
        'D': 0.20, 'E': 0.20, 'F': 0.15, 'G': 0.10, 'H': 0.05
    }
    
    # For non-defaulters: higher probability of good ratings
    RATING_PROBS_NON_DEFAULT = {
        'AA': 0.25, 'A': 0.30, 'B': 0.20, 'C': 0.12,
        'D': 0.06, 'E': 0.03, 'F': 0.02, 'G': 0.01, 'H': 0.01
    }
    
    def __init__(self, seed: int = 42):
        """Initialize with fixed seed for reproducibility."""
        np.random.seed(seed)
        self.ratings = ['AA', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        
    def _get_rating(self, is_defaulter: bool, risk_factor: float) -> str:
        """
        Get SCR rating based on default status and risk factor.
        
        Args:
            is_defaulter: Whether client defaulted internally
            risk_factor: Additional risk from other features (0-1)
        """
        probs = self.RATING_PROBS_DEFAULT if is_defaulter else self.RATING_PROBS_NON_DEFAULT
        
        # Adjust probabilities based on risk factor
        if risk_factor > 0.5:
            # Shift probability towards worse ratings
            adjusted_probs = {}
            for i, rating in enumerate(self.ratings):
                base_prob = probs[rating]
                if i < 3:  # Good ratings
                    adjusted_probs[rating] = base_prob * (1 - risk_factor * 0.5)
                else:  # Bad ratings
                    adjusted_probs[rating] = base_prob * (1 + risk_factor * 0.3)
            # Normalize
            total = sum(adjusted_probs.values())
            probs = {k: v/total for k, v in adjusted_probs.items()}
        
        return np.random.choice(self.ratings, p=list(probs.values()))
    
    def _get_dias_atraso(self, rating: str, is_defaulter: bool) -> int:
        """
        Get days overdue based on rating and default status.
        Days overdue correlates with rating.
        """
        rating_to_base_days = {
            'AA': 0, 'A': 0, 'B': 0, 'C': 15,
            'D': 35, 'E': 70, 'F': 100, 'G': 150, 'H': 200
        }
        
        base_days = rating_to_base_days.get(rating, 0)
        
        if base_days == 0:
            # Good rating - 90% chance of 0 days, 10% chance of small delay
            if np.random.random() < 0.1:
                return np.random.randint(1, 15)
            return 0
        else:
            # Add some variance around base
            variance = int(base_days * 0.3)
            return max(0, base_days + np.random.randint(-variance, variance + 1))
    
    def _get_valores(self, is_defaulter: bool, rating: str, 
                     renda: float = None, idade: int = None) -> dict:
        """
        Generate monetary values based on profile and status.
        
        Returns dict with:
        - valorVencer: Amount to mature
        - valorVencido: Overdue amount
        - valorPrejuizo: Written-off amount
        - limCredito: Credit limit
        - limCreditoUtilizado: Used credit
        """
        # Base credit limit based on income (or default if not available)
        if renda and renda > 0:
            base_limit = float(renda) * np.random.uniform(2, 8)
        else:
            base_limit = np.random.uniform(5000, 50000)
        
        # Adjust by age (older = higher limits generally)
        if idade:
            age_factor = min(1.5, max(0.5, idade / 40))
            base_limit *= age_factor
        
        # Round to nice numbers
        lim_credito = round(base_limit / 1000) * 1000
        
        # Credit utilization varies by rating
        utilization_by_rating = {
            'AA': (0.1, 0.4), 'A': (0.2, 0.5), 'B': (0.3, 0.6),
            'C': (0.4, 0.7), 'D': (0.5, 0.8), 'E': (0.6, 0.9),
            'F': (0.7, 1.0), 'G': (0.8, 1.1), 'H': (0.9, 1.2)
        }
        low, high = utilization_by_rating.get(rating, (0.3, 0.6))
        utilization = np.random.uniform(low, high)
        
        lim_utilizado = min(lim_credito, lim_credito * utilization)
        
        # Amount to mature (positive for everyone with credit)
        valor_vencer = lim_utilizado if utilization < 1 else lim_credito * np.random.uniform(0.5, 0.9)
        
        # Overdue amount (only for bad ratings or defaulters)
        valor_vencido = 0
        if rating in ['D', 'E', 'F', 'G', 'H'] or (is_defaulter and np.random.random() < 0.7):
            overdue_pct = {
                'D': 0.1, 'E': 0.2, 'F': 0.35, 'G': 0.5, 'H': 0.7
            }.get(rating, 0.15 if is_defaulter else 0.05)
            valor_vencido = lim_utilizado * overdue_pct * np.random.uniform(0.5, 1.5)
        
        # Written-off amount (only for very bad ratings)
        valor_prejuizo = 0
        if rating in ['G', 'H'] or (is_defaulter and np.random.random() < 0.3):
            if rating == 'H':
                valor_prejuizo = np.random.uniform(1000, 20000)
            elif rating == 'G':
                valor_prejuizo = np.random.uniform(500, 10000) if np.random.random() < 0.4 else 0
            else:
                valor_prejuizo = np.random.uniform(500, 5000) if np.random.random() < 0.2 else 0
        
        return {
            'scr_lim_credito': round(lim_credito, 2),
            'scr_lim_utilizado': round(lim_utilizado, 2),
            'scr_valor_vencer': round(valor_vencer, 2),
            'scr_valor_vencido': round(valor_vencido, 2),
            'scr_valor_prejuizo': round(valor_prejuizo, 2)
        }
    
    def _get_operacoes(self, is_defaulter: bool, rating: str) -> dict:
        """
        Generate operation counts.
        """
        # Number of institutions (more for active credit users)
        if rating in ['AA', 'A', 'B']:
            qtd_inst = np.random.choice([1, 2, 3, 4], p=[0.4, 0.3, 0.2, 0.1])
        elif rating in ['C', 'D']:
            qtd_inst = np.random.choice([1, 2, 3, 4, 5], p=[0.3, 0.3, 0.2, 0.15, 0.05])
        else:
            qtd_inst = np.random.choice([2, 3, 4, 5, 6], p=[0.2, 0.25, 0.25, 0.2, 0.1])
        
        # Operations per institution
        qtd_ops = qtd_inst * np.random.randint(1, 4)
        
        # Modality (simplified - using main categories)
        modalidades = ['PESSOAL', 'CARTAO', 'VEICULOS', 'IMOBILIARIO', 'CONSIGNADO']
        if is_defaulter:
            modalidade = np.random.choice(modalidades, p=[0.35, 0.35, 0.15, 0.05, 0.10])
        else:
            modalidade = np.random.choice(modalidades, p=[0.20, 0.25, 0.20, 0.15, 0.20])
        
        return {
            'scr_qtd_instituicoes': qtd_inst,
            'scr_qtd_operacoes': qtd_ops,
            'scr_modalidade_principal': modalidade
        }
    
    def generate_for_client(self, cpf: str, is_defaulter: bool, 
                           renda: float = None, idade: int = None,
                           atraso_interno: int = None) -> dict:
        """
        Generate complete SCR record for a single client.
        
        Args:
            cpf: Client CPF
            is_defaulter: Target variable (1 = default, 0 = good)
            renda: Client income (optional, for realistic limit generation)
            idade: Client age (optional)
            atraso_interno: Days overdue internally (if any, increases SCR risk)
        """
        # Calculate risk factor based on available info
        risk_factor = 0.0
        if atraso_interno and atraso_interno > 0:
            risk_factor += min(0.5, atraso_interno / 60)
        if is_defaulter:
            risk_factor += 0.3
        risk_factor = min(1.0, risk_factor)
        
        # Generate rating
        rating = self._get_rating(is_defaulter, risk_factor)
        
        # Generate days overdue
        dias_atraso = self._get_dias_atraso(rating, is_defaulter)
        
        # Generate monetary values
        valores = self._get_valores(is_defaulter, rating, renda, idade)
        
        # Generate operation info
        operacoes = self._get_operacoes(is_defaulter, rating)
        
        # Build complete record
        record = {
            'CPF': cpf,
            'scr_classificacao_risco': rating,
            'scr_dias_atraso': dias_atraso,
            **valores,
            **operacoes
        }
        
        # Calculate derived features
        record['scr_taxa_utilizacao'] = (
            record['scr_lim_utilizado'] / record['scr_lim_credito']
            if record['scr_lim_credito'] > 0 else 0
        )
        
        total_divida = record['scr_valor_vencer'] + record['scr_valor_vencido']
        record['scr_ratio_vencido'] = (
            record['scr_valor_vencido'] / total_divida
            if total_divida > 0 else 0
        )
        
        record['scr_tem_prejuizo'] = 1 if record['scr_valor_prejuizo'] > 0 else 0
        
        # Numeric risk score
        rating_to_score = {
            'AA': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 
            'E': 5, 'F': 6, 'G': 7, 'H': 8
        }
        record['scr_score_risco'] = rating_to_score[rating]
        
        # Days overdue bucket
        if dias_atraso == 0:
            faixa = '0'
        elif dias_atraso <= 14:
            faixa = '1-14'
        elif dias_atraso <= 30:
            faixa = '15-30'
        elif dias_atraso <= 60:
            faixa = '31-60'
        elif dias_atraso <= 90:
            faixa = '61-90'
        else:
            faixa = '90+'
        record['scr_faixa_atraso'] = faixa
        
        # Total exposure
        record['scr_exposicao_total'] = record['scr_valor_vencer'] + record['scr_valor_vencido']
        
        return record
    
    def generate_for_dataset(self, df: pd.DataFrame, 
                            cpf_col: str = 'CPF',
                            target_col: str = 'TARGET',
                            renda_col: str = 'RENDA_LIQUIDA',
                            idade_col: str = 'IDADE_CLIENTE',
                            atraso_col: str = None) -> pd.DataFrame:
        """
        Generate SCR data for entire dataset.
        
        Args:
            df: DataFrame with client data
            cpf_col: Column name for CPF
            target_col: Column name for target variable
            renda_col: Column name for income (optional)
            idade_col: Column name for age (optional)
            atraso_col: Column name for internal days overdue (optional)
        
        Returns:
            DataFrame with SCR data for each client
        """
        logger.info(f"Generating SCR data for {len(df)} clients...")
        
        records = []
        for idx, row in df.iterrows():
            cpf = row.get(cpf_col, str(idx))
            is_defaulter = row.get(target_col, 0) == 1
            renda = row.get(renda_col) if renda_col in df.columns else None
            idade = row.get(idade_col) if idade_col in df.columns else None
            atraso = row.get(atraso_col) if atraso_col and atraso_col in df.columns else None
            
            record = self.generate_for_client(
                cpf=cpf,
                is_defaulter=is_defaulter,
                renda=renda,
                idade=idade,
                atraso_interno=atraso
            )
            records.append(record)
        
        scr_df = pd.DataFrame(records)
        
        # Log statistics
        logger.info(f"SCR data generated:")
        logger.info(f"  - Records: {len(scr_df)}")
        logger.info(f"  - Avg rating score: {scr_df['scr_score_risco'].mean():.2f}")
        logger.info(f"  - Clients with overdue: {(scr_df['scr_valor_vencido'] > 0).sum()} ({(scr_df['scr_valor_vencido'] > 0).mean()*100:.1f}%)")
        logger.info(f"  - Clients with prejuizo: {(scr_df['scr_tem_prejuizo'] == 1).sum()} ({(scr_df['scr_tem_prejuizo'] == 1).mean()*100:.1f}%)")
        
        return scr_df


def main():
    """Generate SCR mock data and merge with existing dataset."""
    from data_pipeline import load_and_prepare_full_dataset
    
    logger.info("Loading existing dataset...")
    df_full, X, y = load_and_prepare_full_dataset()
    
    # Create combined dataframe with index as identifier
    df_combined = X.copy().reset_index(drop=True)
    df_combined['TARGET'] = y.values
    df_combined['client_idx'] = df_combined.index
    
    # Use index as CPF if CPF column doesn't exist
    cpf_col = 'CPF' if 'CPF' in df_combined.columns else 'client_idx'
    
    # Generate SCR data
    generator = SCRDataGenerator(seed=42)
    scr_df = generator.generate_for_dataset(
        df_combined,
        cpf_col=cpf_col,
        target_col='TARGET',
        renda_col='RENDA_LIQUIDA',
        idade_col='IDADE_CLIENTE'
    )
    
    # Save SCR data
    output_path = Path('dados/scr_mock_data.csv')
    output_path.parent.mkdir(exist_ok=True)
    scr_df.to_csv(output_path, index=False)
    logger.info(f"SCR mock data saved to {output_path}")
    
    # Show sample
    print("\n" + "="*70)
    print("SAMPLE SCR DATA (first 10 rows)")
    print("="*70)
    cols_to_show = ['CPF', 'scr_classificacao_risco', 'scr_dias_atraso', 
                    'scr_valor_vencido', 'scr_valor_prejuizo', 'scr_score_risco']
    print(scr_df[cols_to_show].head(10).to_string())
    
    # Show distribution by target
    print("\n" + "="*70)
    print("RATING DISTRIBUTION BY TARGET")
    print("="*70)
    
    # Add target to scr_df using index
    scr_df['TARGET'] = df_combined['TARGET'].values
    
    print("\nNon-defaulters (TARGET=0):")
    print(scr_df[scr_df['TARGET'] == 0]['scr_classificacao_risco'].value_counts(normalize=True).sort_index())
    print("\nDefaulters (TARGET=1):")
    print(scr_df[scr_df['TARGET'] == 1]['scr_classificacao_risco'].value_counts(normalize=True).sort_index())
    
    # Remove target from scr_df before returning
    scr_df = scr_df.drop(columns=['TARGET'])
    
    return scr_df


if __name__ == "__main__":
    main()

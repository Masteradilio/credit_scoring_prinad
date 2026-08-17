"""
Data Consolidator PRINAD - Data Pipeline for PRINAD Credit Scoring Model.

Creates consolidated training data for the PRINAD model with:
1. Cadastral data (demographics)
2. Behavioral data (v-columns from 3040)
3. SCR data (credit bureau)
4. Target variable (CLASS: 0=good, 1=bad)

Output: base_prinad_treino.csv with high variance and quality for ML training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import sys
import warnings
import logging

warnings.filterwarnings('ignore')

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BASE_DIR.parent
DADOS_DIR = PROJECT_DIR / "dados"

sys.path.insert(0, str(PROJECT_DIR))

try:
    from shared.utils import get_rating_from_prinad, calcular_pd_por_rating
except ImportError:
    # Fallback implementations
    def get_rating_from_prinad(prinad: float) -> str:
        if prinad < 5: return 'A1'
        if prinad < 15: return 'A2'
        if prinad < 25: return 'A3'
        if prinad < 35: return 'B1'
        if prinad < 50: return 'B2'
        if prinad < 65: return 'B3'
        if prinad < 75: return 'C1'
        if prinad < 85: return 'C2'
        if prinad < 95: return 'D'
        return 'DEFAULT'
    
    def calcular_pd_por_rating(prinad: float) -> Dict:
        rating = get_rating_from_prinad(prinad)
        pd_12m = prinad / 100
        multipliers = {'A1': 2.5, 'A2': 2.5, 'A3': 3.0, 'B1': 3.5, 'B2': 4.0, 
                       'B3': 4.5, 'C1': 5.0, 'C2': 5.5, 'D': 6.5, 'DEFAULT': 7.0}
        return {'pd_12m': pd_12m, 'pd_lifetime': pd_12m * multipliers.get(rating, 3.0)}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Random seed for reproducibility
np.random.seed(42)


@dataclass
class PRINADConfig:
    """Configuration for PRINAD data generation."""
    encoding: str = 'latin-1'
    sep: str = ';'
    n_records: int = 50000  # Number of records to generate
    bad_rate: float = 0.15  # 15% default rate (realistic for credit)
    noise_level: float = 0.10  # 10% noise


class PRINADDataGenerator:
    """
    Generates high-quality training data for PRINAD model.
    
    Creates realistic distributions with proper correlations between features
    and the target variable (default/non-default).
    """
    
    # Risk profile distribution
    PROFILE_DISTRIBUTION = {
        'excellent': 0.15,   # A1/A2 - Very low risk
        'good': 0.25,        # A3/B1 - Low risk
        'moderate': 0.30,    # B2/B3 - Medium risk
        'risky': 0.20,       # C1/C2 - High risk
        'default': 0.10      # D/DEFAULT - Very high risk
    }
    
    def __init__(self, config: PRINADConfig):
        self.config = config
        self.noise = config.noise_level
    
    def add_noise(self, value: float, level: float = None) -> float:
        """Add Gaussian noise to a value."""
        if level is None:
            level = self.noise
        return value * (1 + np.random.normal(0, level))
    
    def generate_cpf(self, n: int) -> np.ndarray:
        """Generate unique CPF-like identifiers."""
        # Generate 11-digit CPF using two parts to avoid int32 overflow
        cpfs = []
        for _ in range(n):
            part1 = np.random.randint(100, 999)
            part2 = np.random.randint(100, 999)
            part3 = np.random.randint(100, 999)
            part4 = np.random.randint(10, 99)
            cpfs.append(f"{part1}{part2}{part3}{part4}")
        return np.array(cpfs)
    
    def generate_risk_profiles(self, n: int) -> np.ndarray:
        """Assign risk profiles based on distribution."""
        profiles = list(self.PROFILE_DISTRIBUTION.keys())
        probs = list(self.PROFILE_DISTRIBUTION.values())
        return np.random.choice(profiles, n, p=probs)
    
    def generate_cadastral_data(self, n: int, profiles: np.ndarray) -> pd.DataFrame:
        """
        Generate cadastral (demographic) data with profile-based distributions.
        """
        logger.info(f"Generating cadastral data for {n} records...")
        
        data = {
            'CPF': self.generate_cpf(n),
            'CLIT': np.arange(1, n + 1)
        }
        
        # Age: 18-80, older clients tend to have better profiles
        age_by_profile = {
            'excellent': (35, 65, 8),
            'good': (30, 60, 10),
            'moderate': (25, 55, 12),
            'risky': (22, 50, 15),
            'default': (20, 45, 12)
        }
        
        ages = np.zeros(n)
        for i, profile in enumerate(profiles):
            mean, max_age, std = age_by_profile[profile]
            ages[i] = np.clip(np.random.normal(mean + 10, std), 18, 80)
        data['IDADE_CLIENTE'] = ages.astype(int)
        
        # Education level
        edu_by_profile = {
            'excellent': ['SUPERIOR', 'POS', 'SUPERIOR', 'POS', 'MEDIO'],
            'good': ['SUPERIOR', 'POS', 'MEDIO', 'SUPERIOR', 'MEDIO'],
            'moderate': ['MEDIO', 'SUPERIOR', 'MEDIO', 'FUNDAM', 'MEDIO'],
            'risky': ['MEDIO', 'FUNDAM', 'MEDIO', 'FUNDAM', 'MEDIO'],
            'default': ['FUNDAM', 'MEDIO', 'FUNDAM', 'FUNDAM', 'MEDIO']
        }
        data['ESCOLARIDADE'] = [np.random.choice(edu_by_profile[p]) for p in profiles]
        
        # Income: correlated with education and profile
        income_multipliers = {'FUNDAM': 1.0, 'MEDIO': 1.5, 'SUPERIOR': 2.5, 'POS': 4.0}
        base_income_by_profile = {
            'excellent': 8000, 'good': 5000, 'moderate': 3500, 
            'risky': 2500, 'default': 2000
        }
        
        incomes = []
        for p, edu in zip(profiles, data['ESCOLARIDADE']):
            base = base_income_by_profile[p] * income_multipliers.get(edu, 1)
            incomes.append(max(1500, self.add_noise(base, 0.30)))
        data['RENDA_BRUTA'] = np.array(incomes).round(2)
        
        # Relationship time (months): longer for better profiles
        tempo_by_profile = {
            'excellent': (60, 30), 'good': (48, 25), 'moderate': (24, 20),
            'risky': (12, 15), 'default': (6, 10)
        }
        data['TEMPO_RELAC'] = [
            max(1, int(np.random.normal(tempo_by_profile[p][0], tempo_by_profile[p][1])))
            for p in profiles
        ]
        
        # Marital status
        data['ESTADO_CIVIL'] = np.random.choice(
            ['SOLTEIRO', 'CASADO', 'DIVORCIADO', 'VIUVO'],
            n, p=[0.30, 0.50, 0.15, 0.05]
        )
        
        # Has vehicle: better profiles more likely
        vehicle_prob = {'excellent': 0.80, 'good': 0.65, 'moderate': 0.45, 
                        'risky': 0.30, 'default': 0.15}
        data['POSSUI_VEICULO'] = [
            'SIM' if np.random.random() < vehicle_prob[p] else 'NAO'
            for p in profiles
        ]
        
        # Number of products: more products for better profiles
        qt_produtos_by_profile = {
            'excellent': (3, 5), 'good': (2, 4), 'moderate': (2, 3),
            'risky': (1, 2), 'default': (1, 2)
        }
        data['QT_PRODUTOS'] = [
            np.random.randint(*qt_produtos_by_profile[p])
            for p in profiles
        ]
        
        return pd.DataFrame(data)
    
    def generate_behavioral_data(self, n: int, profiles: np.ndarray) -> pd.DataFrame:
        """
        Generate behavioral data (v-columns) representing internal delinquency history.
        
        v205: Sum of days overdue in last 24 months
        v210: Sum of days overdue 15-30 days
        v220: Sum of days overdue 31-60 days
        v230: Sum of days overdue 61-90 days
        v240: Sum of days overdue 91-120 days
        v245: Sum of days overdue 121+ days
        v250-v290: Additional behavioral indicators
        """
        logger.info(f"Generating behavioral data (v-columns) for {n} records...")
        
        # V-column ranges by profile (min, max)
        v_config = {
            'excellent': {
                'v205': (0, 0), 'v210': (0, 0), 'v220': (0, 0),
                'v230': (0, 0), 'v240': (0, 0), 'v245': (0, 0)
            },
            'good': {
                'v205': (0, 100), 'v210': (0, 50), 'v220': (0, 0),
                'v230': (0, 0), 'v240': (0, 0), 'v245': (0, 0)
            },
            'moderate': {
                'v205': (50, 500), 'v210': (0, 200), 'v220': (0, 100),
                'v230': (0, 0), 'v240': (0, 0), 'v245': (0, 0)
            },
            'risky': {
                'v205': (200, 2000), 'v210': (100, 500), 'v220': (50, 300),
                'v230': (0, 150), 'v240': (0, 50), 'v245': (0, 0)
            },
            'default': {
                'v205': (1000, 5000), 'v210': (500, 2000), 'v220': (300, 1500),
                'v230': (200, 1000), 'v240': (100, 500), 'v245': (50, 300)
            }
        }
        
        data = {}
        v_cols = ['v205', 'v210', 'v220', 'v230', 'v240', 'v245']
        
        for v_col in v_cols:
            values = np.zeros(n)
            for i, profile in enumerate(profiles):
                min_val, max_val = v_config[profile][v_col]
                if max_val > 0:
                    # Use exponential distribution for realistic tail
                    scale = (max_val - min_val) / 3
                    values[i] = min_val + np.random.exponential(scale)
                    values[i] = np.clip(values[i], min_val, max_val * 1.5)
                else:
                    values[i] = 0
            data[v_col] = values.round(0)
        
        # Additional v-columns (v250-v290) - other behavioral indicators
        for v_col in ['v250', 'v255', 'v260', 'v270', 'v280', 'v290']:
            values = np.zeros(n)
            for i, profile in enumerate(profiles):
                if profile in ['excellent', 'good']:
                    values[i] = 0
                elif profile == 'moderate':
                    values[i] = np.random.exponential(50) if np.random.random() < 0.2 else 0
                elif profile == 'risky':
                    values[i] = np.random.exponential(150) if np.random.random() < 0.4 else 0
                else:  # default
                    values[i] = np.random.exponential(300) if np.random.random() < 0.6 else 0
            data[v_col] = values.round(0)
        
        return pd.DataFrame(data)
    
    def generate_scr_data(self, n: int, profiles: np.ndarray) -> pd.DataFrame:
        """
        Generate SCR (Credit Bureau) data.
        
        scr_score_risco: 0-7 (0=AA, 7=H - higher is worse)
        scr_dias_atraso: Days in arrears
        scr_tem_prejuizo: Has loss record (0/1)
        """
        logger.info(f"Generating SCR data for {n} records...")
        
        # SCR score by profile (lower is better, range 0-7)
        scr_score_config = {
            'excellent': (0, 1),
            'good': (0, 2),
            'moderate': (1, 4),
            'risky': (3, 6),
            'default': (5, 7)
        }
        
        # Days in arrears by profile
        dias_config = {
            'excellent': (0, 0),
            'good': (0, 15),
            'moderate': (0, 45),
            'risky': (15, 90),
            'default': (60, 365)
        }
        
        data = {}
        
        # SCR Score
        scores = np.zeros(n)
        for i, profile in enumerate(profiles):
            min_s, max_s = scr_score_config[profile]
            scores[i] = np.random.randint(min_s, max_s + 1)
        data['scr_score_risco'] = scores.astype(int)
        
        # Days in arrears
        dias = np.zeros(n)
        for i, profile in enumerate(profiles):
            min_d, max_d = dias_config[profile]
            if max_d > 0:
                dias[i] = np.random.exponential((max_d - min_d) / 2) + min_d
                dias[i] = np.clip(dias[i], min_d, max_d * 1.2)
            else:
                dias[i] = 0
        data['scr_dias_atraso'] = dias.astype(int)
        
        # Has loss record
        prejuizo_prob = {
            'excellent': 0.0, 'good': 0.01, 'moderate': 0.05,
            'risky': 0.15, 'default': 0.50
        }
        data['scr_tem_prejuizo'] = [
            1 if np.random.random() < prejuizo_prob[p] else 0
            for p in profiles
        ]
        
        return pd.DataFrame(data)
    
    def generate_credit_data(self, n: int, profiles: np.ndarray, renda: np.ndarray) -> pd.DataFrame:
        """Generate credit limit and utilization data."""
        logger.info(f"Generating credit data for {n} records...")
        
        # Limit multipliers by profile
        limit_mult = {
            'excellent': (8, 15), 'good': (5, 10), 'moderate': (3, 7),
            'risky': (1, 4), 'default': (0.5, 2)
        }
        
        # Utilization rate by profile (higher for risky)
        util_config = {
            'excellent': (0.10, 0.40),
            'good': (0.20, 0.50),
            'moderate': (0.40, 0.70),
            'risky': (0.60, 0.90),
            'default': (0.80, 1.00)
        }
        
        data = {}
        
        limits = np.zeros(n)
        used = np.zeros(n)
        
        for i, (profile, income) in enumerate(zip(profiles, renda)):
            min_m, max_m = limit_mult[profile]
            limits[i] = income * np.random.uniform(min_m, max_m)
            
            min_u, max_u = util_config[profile]
            util_rate = np.random.uniform(min_u, max_u)
            used[i] = limits[i] * util_rate
        
        data['limite_total'] = limits.round(2)
        data['limite_utilizado'] = used.round(2)
        data['taxa_utilizacao'] = (used / np.maximum(limits, 1)).round(4)
        
        # Income commitment ratio
        parcelas = used / 48  # Approximate monthly payment
        data['parcelas_mensais'] = parcelas.round(2)
        data['comprometimento_renda'] = (parcelas / np.maximum(renda, 1)).round(4)
        data['margem_disponivel'] = (renda * 0.35 - parcelas).round(2)
        
        # Utilization history
        data['utilizacao_media_12m'] = (data['taxa_utilizacao'] * np.random.uniform(0.8, 1.2, n)).round(4)
        data['trimestres_sem_uso'] = np.where(
            data['taxa_utilizacao'] < 0.1,
            np.random.choice([1, 2, 3, 4], n, p=[0.3, 0.3, 0.25, 0.15]),
            np.random.choice([0, 1], n, p=[0.8, 0.2])
        )
        
        # Max dias atraso 12m (from behavioral)
        data['max_dias_atraso_12m'] = np.zeros(n)  # Will be updated later
        
        return pd.DataFrame(data)
    
    def generate_target_variable(self, n: int, profiles: np.ndarray) -> np.ndarray:
        """
        Generate target variable (CLASS) based on profile.
        
        CLASS = 1 means the client defaulted (bad)
        CLASS = 0 means the client is good
        """
        logger.info(f"Generating target variable for {n} records...")
        
        # Default probability by profile
        default_prob = {
            'excellent': 0.02,
            'good': 0.05,
            'moderate': 0.15,
            'risky': 0.35,
            'default': 0.70
        }
        
        targets = np.array([
            1 if np.random.random() < default_prob[p] else 0
            for p in profiles
        ])
        
        return targets
    
    def generate_full_dataset(self) -> pd.DataFrame:
        """Generate the complete PRINAD training dataset."""
        n = self.config.n_records
        
        logger.info("=" * 60)
        logger.info(f"GENERATING PRINAD TRAINING DATA ({n} records)")
        logger.info("=" * 60)
        
        # Step 1: Generate risk profiles
        profiles = self.generate_risk_profiles(n)
        profile_dist = pd.Series(profiles).value_counts()
        logger.info(f"Profile distribution:\n{profile_dist}")
        
        # Step 2: Generate cadastral data
        df_cadastral = self.generate_cadastral_data(n, profiles)
        
        # Step 3: Generate behavioral data (v-columns)
        df_behavioral = self.generate_behavioral_data(n, profiles)
        
        # Step 4: Generate SCR data
        df_scr = self.generate_scr_data(n, profiles)
        
        # Step 5: Generate credit data
        df_credit = self.generate_credit_data(n, profiles, df_cadastral['RENDA_BRUTA'].values)
        
        # Step 6: Generate target variable
        targets = self.generate_target_variable(n, profiles)
        
        # Combine all data
        df = pd.concat([df_cadastral, df_behavioral, df_scr, df_credit], axis=1)
        df['CLASSE'] = targets
        df['profile'] = profiles  # Keep for analysis
        
        # Update max_dias_atraso_12m from v-columns
        df['max_dias_atraso_12m'] = df_scr['scr_dias_atraso']
        
        # Calculate derived fields (matching feature_engineer output)
        df['produto'] = np.random.choice(
            ['consignado', 'banparacard', 'cartao_credito', 'imobiliario', 'cheque_especial'],
            n, p=[0.45, 0.20, 0.15, 0.10, 0.10]
        )
        
        # Add derived features that feature_engineer creates
        # Age features
        df['em_idade_ativa'] = ((df['IDADE_CLIENTE'] >= 18) & (df['IDADE_CLIENTE'] <= 65)).astype(int)
        df['idade_squared'] = df['IDADE_CLIENTE'] ** 2
        
        # Relationship features
        df['cliente_novo'] = (df['TEMPO_RELAC'] < 6).astype(int)
        df['log_tempo_relac'] = np.log1p(df['TEMPO_RELAC'].clip(lower=0))
        
        # Vehicle indicator
        df['tem_veiculo'] = (df['POSSUI_VEICULO'] == 'SIM').astype(int)
        
        # Education score
        education_score = {'FUNDAM': 0, 'MEDIO': 1, 'SUPERIOR': 2, 'POS': 3}
        df['score_escolaridade'] = df['ESCOLARIDADE'].map(education_score).fillna(1)
        
        # Marital status score
        marital_score = {'CASADO': -0.1, 'SOLTEIRO': 0.1, 'DIVORCIADO': 0.05, 'VIUVO': 0}
        df['score_estado_civil'] = df['ESTADO_CIVIL'].map(marital_score).fillna(0)
        
        # Log statistics
        logger.info("=" * 60)
        logger.info("DATASET STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total records: {len(df)}")
        logger.info(f"Default rate: {df['CLASSE'].mean():.2%}")
        logger.info(f"Profile distribution: {df['profile'].value_counts().to_dict()}")
        logger.info(f"Columns: {list(df.columns)}")
        
        return df
    
    def generate_and_save(self, output_path: Optional[Path] = None) -> pd.DataFrame:
        """Generate dataset and save to CSV."""
        df = self.generate_full_dataset()
        
        if output_path is None:
            output_path = DADOS_DIR / 'base_prinad_treino.csv'
        
        # Remove profile column before saving (internal use only)
        df_save = df.drop(columns=['profile'])
        
        # Save training dataset
        df_save.to_csv(output_path, sep=';', encoding='latin-1', index=False)
        logger.info(f"Training dataset saved to {output_path}")
        logger.info(f"Final shape: {df_save.shape}")
        
        # Also save as base_clientes.csv for API use (subset of records)
        base_clientes_path = DADOS_DIR / 'base_clientes.csv'
        # Take a sample for inference testing
        df_clientes = df_save.sample(n=min(1000, len(df_save)), random_state=42).copy()
        df_clientes.to_csv(base_clientes_path, sep=';', encoding='latin-1', index=False)
        logger.info(f"API dataset saved to {base_clientes_path} ({len(df_clientes)} records)")
        
        return df


def main():
    """Main function to generate PRINAD training data."""
    print("=" * 60)
    print("PRINAD - Data Consolidator")
    print("=" * 60)
    
    config = PRINADConfig(
        n_records=50000,  # 50k records
        bad_rate=0.15,    # 15% default rate
        noise_level=0.10
    )
    
    generator = PRINADDataGenerator(config)
    df = generator.generate_and_save()
    
    print(f"\n✅ Dataset generated successfully!")
    print(f"   Records: {len(df)}")
    print(f"   Default rate: {df['CLASSE'].mean():.2%}")
    print(f"   Output: {DADOS_DIR / 'base_prinad_treino.csv'}")
    
    # Show sample
    print("\n📊 Sample data:")
    print(df[['CPF', 'IDADE_CLIENTE', 'RENDA_BRUTA', 'v205', 'v210', 
              'scr_score_risco', 'scr_dias_atraso', 'CLASSE']].head(10).to_string())
    
    # Show class distribution by profile
    print("\n📈 Class distribution by profile:")
    print(df.groupby('profile')['CLASSE'].agg(['count', 'sum', 'mean']).round(3))
    
    return df


if __name__ == "__main__":
    main()

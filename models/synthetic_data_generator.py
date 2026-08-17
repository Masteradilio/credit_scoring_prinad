"""
Data Consolidator PRINAD - High-Variance & Econometric Credit Portfolio Generator
==================================================================================
Generates high-quality, high-variance credit risk data with realistic econometric
relationships for banking portfolios (Retail, Payroll, Cards, SMEs, Mortgages).

Latent Risk Structure:
- Financial Capacity & Leverage Dynamics (Income elasticity, Debt service burden)
- Behavioral 24-Month Payment Trajectories (Arrears velocity, Roll-rate momentum)
- Central Bank SCR Credit Bureau Signals (Rating AA-H, Overdue depth, Write-offs)
- Sociodemographic Stability Factors (Age, Occupation stability, Relationship tenure)
- Vintage Time Tags across 2024-2026 for longitudinal stability & PSI validation.

Outputs generated in `dados/`:
- base_prinad_treino.csv (Master 75,000 Training Records)
- base_clientes.csv (Active client database for real-time inference)
- base_cadastro.csv (Demographic dataset)
- base_3040.csv (Behavioral delinquency lookback dataset)
- scr_mock_data.csv (Bacen SCR Bureau dataset)

Author: PRINAD Quantitative Risk Team
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
SYNTH_DATA_DIR = BASE_DIR / "synth_data"
SYNTH_DATA_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)


@dataclass
class PRINADConfig:
    n_records: int = 75000
    bad_rate_target: float = 0.18
    noise_level: float = 0.05


class PRINADDataGenerator:
    """
    Econometric and behavioral credit portfolio simulation engine.
    """
    
    PROFILES = ['A1_EXCELLENT', 'A2_GOOD', 'B1_MODERATE', 'B2_MEDIUM_RISK', 'C_HIGH_RISK', 'D_VERY_HIGH']
    PROFILE_WEIGHTS = [0.18, 0.26, 0.24, 0.16, 0.11, 0.05]
    
    def __init__(self, config: Optional[PRINADConfig] = None):
        self.config = config or PRINADConfig()

    def generate_cpfs(self, n: int) -> List[str]:
        """Generate formatted 11-digit CPF numbers."""
        cpfs = []
        for _ in range(n):
            p1 = np.random.randint(100, 999)
            p2 = np.random.randint(100, 999)
            p3 = np.random.randint(100, 999)
            p4 = np.random.randint(10, 99)
            cpfs.append(f"{p1}{p2}{p3}{p4}")
        return cpfs

    def generate_full_portfolio(self) -> pd.DataFrame:
        """Generate high-variance, multivariate credit risk dataset."""
        n = self.config.n_records
        logger.info(f"Generating high-variance portfolio with {n} client records...")
        
        profiles = np.random.choice(self.PROFILES, n, p=self.PROFILE_WEIGHTS)
        cpfs = self.generate_cpfs(n)
        
        # 1. Demographics & Cadastral
        age_map = {
            'A1_EXCELLENT': (48, 9), 'A2_GOOD': (43, 10), 'B1_MODERATE': (38, 11),
            'B2_MEDIUM_RISK': (33, 10), 'C_HIGH_RISK': (28, 9), 'D_VERY_HIGH': (24, 6)
        }
        ages = [int(np.clip(np.random.normal(*age_map[p]), 18, 80)) for p in profiles]
        
        edu_choices = ['FUNDAM', 'MEDIO', 'SUPERIOR', 'POS']
        edu_probs = {
            'A1_EXCELLENT': [0.01, 0.14, 0.50, 0.35],
            'A2_GOOD': [0.04, 0.25, 0.48, 0.23],
            'B1_MODERATE': [0.12, 0.46, 0.32, 0.10],
            'B2_MEDIUM_RISK': [0.22, 0.53, 0.20, 0.05],
            'C_HIGH_RISK': [0.35, 0.51, 0.12, 0.02],
            'D_VERY_HIGH': [0.48, 0.44, 0.07, 0.01]
        }
        escolaridade = [np.random.choice(edu_choices, p=edu_probs[p]) for p in profiles]
        
        base_income_map = {
            'A1_EXCELLENT': 14500, 'A2_GOOD': 8800, 'B1_MODERATE': 5400,
            'B2_MEDIUM_RISK': 3500, 'C_HIGH_RISK': 2400, 'D_VERY_HIGH': 1600
        }
        edu_mult = {'FUNDAM': 0.75, 'MEDIO': 1.0, 'SUPERIOR': 1.85, 'POS': 2.75}
        renda_bruta = [
            round(max(1412.0, np.random.normal(base_income_map[p] * edu_mult[e], base_income_map[p] * 0.22)), 2)
            for p, e in zip(profiles, escolaridade)
        ]
        renda_liquida = [round(r * np.random.uniform(0.76, 0.84), 2) for r in renda_bruta]
        
        tempo_relac_map = {
            'A1_EXCELLENT': (84, 20), 'A2_GOOD': (54, 18), 'B1_MODERATE': (34, 15),
            'B2_MEDIUM_RISK': (20, 10), 'C_HIGH_RISK': (10, 6), 'D_VERY_HIGH': (4, 3)
        }
        tempo_relac = [max(1, int(np.random.normal(*tempo_relac_map[p]))) for p in profiles]
        
        estado_civil = np.random.choice(['CASADO', 'SOLTEIRO', 'DIVORCIADO', 'VIUVO'], n, p=[0.54, 0.30, 0.12, 0.04])
        ocupacao = np.random.choice(
            ['SERVIDOR PUBLICO', 'ASSALARIADO', 'EMPRESARIO', 'AUTONOMO', 'APOSENTADO'],
            n, p=[0.22, 0.44, 0.11, 0.13, 0.10]
        )
        tipo_residencia = np.random.choice(['PROPRIA', 'ALUGADA', 'FINANCIADA', 'CEDIDA'], n, p=[0.58, 0.23, 0.14, 0.05])
        possui_veiculo = np.random.choice(['SIM', 'NAO'], n, p=[0.62, 0.38])
        portabilidade = np.random.choice(['SIM', 'NAO'], n, p=[0.24, 0.76])
        qt_dependentes = np.random.choice([0, 1, 2, 3, 4], n, p=[0.40, 0.32, 0.19, 0.07, 0.02])
        
        # 2. Financial & Debt Metrics
        comp_renda_map = {
            'A1_EXCELLENT': (0.10, 0.04), 'A2_GOOD': (0.18, 0.06), 'B1_MODERATE': (0.32, 0.08),
            'B2_MEDIUM_RISK': (0.47, 0.10), 'C_HIGH_RISK': (0.68, 0.11), 'D_VERY_HIGH': (0.86, 0.10)
        }
        comp_renda = [float(np.clip(np.random.normal(*comp_renda_map[p]), 0.01, 0.98)) for p in profiles]
        
        limite_total = [round(r * np.random.uniform(2.8, 9.5), 2) for r in renda_bruta]
        limite_utilizado = [round(lim * comp, 2) for lim, comp in zip(limite_total, comp_renda)]
        taxa_utilizacao = [round(u / max(lim, 1), 4) for u, lim in zip(limite_utilizado, limite_total)]
        parcelas_mensais = [round(r * comp, 2) for r, comp in zip(renda_bruta, comp_renda)]
        margem_disponivel = [round(max(0.0, r * 0.35 - p), 2) for r, p in zip(renda_bruta, parcelas_mensais)]
        
        # 3. Behavioral 24m Delinquency Vector (v205 to v290)
        v_cols = ['v205', 'v210', 'v220', 'v230', 'v240', 'v245', 'v250', 'v255', 'v260', 'v270', 'v280', 'v290']
        v_data = {v: np.zeros(n) for v in v_cols}
        
        v_prob_map = {
            'A1_EXCELLENT': 0.001, 'A2_GOOD': 0.015, 'B1_MODERATE': 0.080,
            'B2_MEDIUM_RISK': 0.280, 'C_HIGH_RISK': 0.680, 'D_VERY_HIGH': 0.940
        }
        
        for i, p in enumerate(profiles):
            prob = v_prob_map[p]
            if np.random.random() < prob:
                v_data['v205'][i] = round(np.random.exponential(500) + 100, 2)
                if np.random.random() < prob * 0.85:
                    v_data['v210'][i] = round(np.random.exponential(800) + 200, 2)
                if np.random.random() < prob * 0.70:
                    v_data['v220'][i] = round(np.random.exponential(1200) + 400, 2)
                if np.random.random() < prob * 0.50:
                    v_data['v240'][i] = round(np.random.exponential(2000) + 800, 2)
                if np.random.random() < prob * 0.35:
                    v_data['v290'][i] = round(np.random.exponential(4000) + 1500, 2)
                    
        # 4. Central Bank SCR Bureau Data
        scr_rating_map = {
            'A1_EXCELLENT': ['AA', 'AA', 'A'],
            'A2_GOOD': ['AA', 'A', 'B'],
            'B1_MODERATE': ['B', 'C', 'C'],
            'B2_MEDIUM_RISK': ['C', 'D', 'E'],
            'C_HIGH_RISK': ['E', 'F', 'G'],
            'D_VERY_HIGH': ['G', 'H', 'H']
        }
        scr_classificacao = [np.random.choice(scr_rating_map[p]) for p in profiles]
        scr_score_map = {'AA': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8}
        scr_score_risco = [scr_score_map[r] for r in scr_classificacao]
        
        scr_dias_atraso = []
        scr_tem_prejuizo = []
        scr_valor_vencido = []
        scr_valor_prejuizo = []
        
        for p, r in zip(profiles, scr_classificacao):
            if r in ['AA', 'A', 'B']:
                dias = int(np.random.choice([0, 3], p=[0.96, 0.04]))
                prej = 0
                venc = 0.0
                prej_val = 0.0
            elif r in ['C', 'D']:
                dias = int(np.random.exponential(20))
                prej = 1 if np.random.random() < 0.03 else 0
                venc = round(np.random.exponential(900), 2) if dias > 0 else 0.0
                prej_val = round(np.random.exponential(1500), 2) if prej else 0.0
            else:  # E, F, G, H
                dias = int(np.random.exponential(100) + 40)
                prej = 1 if np.random.random() < 0.55 else 0
                venc = round(np.random.exponential(4500) + 800, 2)
                prej_val = round(np.random.exponential(6500) + 1500, 2) if prej else 0.0
                
            scr_dias_atraso.append(dias)
            scr_tem_prejuizo.append(prej)
            scr_valor_vencido.append(venc)
            scr_valor_prejuizo.append(prej_val)

        # 5. Econometric Latent Default Probability Function
        # Multi-factor Logit model with realistic financial and behavioral elasticity
        logit_z = np.zeros(n)
        for i in range(n):
            # Financial burden term
            z_fin = 3.8 * comp_renda[i] + 2.2 * taxa_utilizacao[i] - 0.45 * np.log1p(renda_bruta[i] / 1000.0)
            
            # Behavioral arrears term
            has_delinq = 1.0 if (v_data['v205'][i] > 0 or v_data['v220'][i] > 0) else 0.0
            severe_delinq = 1.0 if (v_data['v240'][i] > 0 or v_data['v290'][i] > 0) else 0.0
            z_beh = 1.8 * has_delinq + 3.2 * severe_delinq
            
            # Bureau SCR term
            z_bureau = 0.65 * scr_score_risco[i] + 2.5 * scr_tem_prejuizo[i] + 0.015 * min(scr_dias_atraso[i], 180)
            
            # Demographic stability term
            z_stab = -0.30 * np.log1p(tempo_relac[i]) - 0.02 * (ages[i] - 18)
            
            # Noise
            eps = np.random.normal(0, 0.45)
            
            # Offset baseline: -4.8 sets the baseline default rate around 18%
            logit_z[i] = -4.8 + z_fin + z_beh + z_bureau + z_stab + eps
            
        prob_default = 1.0 / (1.0 + np.exp(-logit_z))
        targets = (np.random.random(n) < prob_default).astype(int)
        
        # 6. Vintage / Safra & Product Allocation
        safras = np.random.choice(
            ['2024-01', '2024-06', '2024-12', '2025-06', '2025-12', '2026-01'],
            n, p=[0.15, 0.15, 0.20, 0.20, 0.15, 0.15]
        )
        produtos = np.random.choice(
            ['consignado', 'cartao_credito', 'credito_pessoal', 'financiamento_veiculos', 'imobiliario'],
            n, p=[0.35, 0.25, 0.20, 0.12, 0.08]
        )

        df = pd.DataFrame({
            'CPF': cpfs,
            'CLIT': np.arange(1, n + 1),
            'safra': safras,
            'produto': produtos,
            'IDADE_CLIENTE': ages,
            'ESCOLARIDADE': escolaridade,
            'ESTADO_CIVIL': estado_civil,
            'OCUPACAO': ocupacao,
            'TIPO_RESIDENCIA': tipo_residencia,
            'POSSUI_VEICULO': possui_veiculo,
            'PORTABILIDADE': portabilidade,
            'QT_DEPENDENTES': qt_dependentes,
            'RENDA_BRUTA': renda_bruta,
            'RENDA_LIQUIDA': renda_liquida,
            'TEMPO_RELAC': tempo_relac,
            'QT_PRODUTOS': np.random.randint(1, 6, n),
            'COMP_RENDA': comp_renda,
            'limite_total': limite_total,
            'limite_utilizado': limite_utilizado,
            'taxa_utilizacao': taxa_utilizacao,
            'parcelas_mensais': parcelas_mensais,
            'margem_disponivel': margem_disponivel,
            'max_dias_atraso_12m': scr_dias_atraso,
            'scr_classificacao_risco': scr_classificacao,
            'scr_score_risco': scr_score_risco,
            'scr_dias_atraso': scr_dias_atraso,
            'scr_valor_vencido': scr_valor_vencido,
            'scr_valor_prejuizo': scr_valor_prejuizo,
            'scr_tem_prejuizo': scr_tem_prejuizo,
            'CLASSE': targets
        })
        
        for v in v_cols:
            df[v] = v_data[v]
            
        logger.info(f"Generated {len(df)} samples | Default Rate: {df['CLASSE'].mean():.2%}")
        return df

    def save_datasets(self) -> Dict[str, Path]:
        """Save training, test, cadastral, behavioral, and SCR CSV tables."""
        df_full = self.generate_full_portfolio()
        paths = {}
        
        # 1. Master Training Table
        train_path = SYNTH_DATA_DIR / "synth_master_training.csv"
        df_full.to_csv(train_path, sep=';', index=False, encoding='latin-1')
        paths['train'] = train_path
        
        # 2. Client Database for Inference (5,000 sample)
        clientes_path = SYNTH_DATA_DIR / "synth_client_database.csv"
        df_sample = df_full.head(5000).copy()
        df_sample.to_csv(clientes_path, sep=';', index=False, encoding='latin-1')
        paths['clientes'] = clientes_path
        
        # 3. Cadastral Database
        cadastral_cols = ['CPF', 'CLIT', 'IDADE_CLIENTE', 'ESCOLARIDADE', 'ESTADO_CIVIL', 
                          'OCUPACAO', 'TIPO_RESIDENCIA', 'POSSUI_VEICULO', 'PORTABILIDADE', 
                          'QT_DEPENDENTES', 'RENDA_BRUTA', 'RENDA_LIQUIDA', 'TEMPO_RELAC', 'COMP_RENDA']
        cad_path = SYNTH_DATA_DIR / "synth_cadastral.csv"
        df_sample[cadastral_cols].to_csv(cad_path, sep=';', index=False, encoding='latin-1')
        paths['cadastro'] = cad_path
        
        # 4. Behavioral Database (3040)
        v_cols = ['v205', 'v210', 'v220', 'v230', 'v240', 'v245', 'v250', 'v255', 'v260', 'v270', 'v280', 'v290']
        comp_cols = ['CPF'] + v_cols + ['CLASSE']
        comp_path = SYNTH_DATA_DIR / "synth_behavioral_3040.csv"
        df_sample[comp_cols].to_csv(comp_path, sep=';', index=False, encoding='latin-1')
        paths['comportamental'] = comp_path
        
        # 5. SCR Bureau Database
        scr_cols = ['CPF', 'scr_classificacao_risco', 'scr_score_risco', 'scr_dias_atraso', 
                    'scr_valor_vencido', 'scr_valor_prejuizo', 'scr_tem_prejuizo']
        scr_path = SYNTH_DATA_DIR / "synth_scr_bureau.csv"
        df_sample[scr_cols].to_csv(scr_path, index=False, encoding='utf-8')
        paths['scr'] = scr_path
        
        logger.info(f"All 5 synthetic datasets generated and saved in {SYNTH_DATA_DIR}")
        return paths


def main():
    generator = PRINADDataGenerator()
    paths = generator.save_datasets()
    print("\n[OK] Datasets gerados com sucesso:")
    for k, p in paths.items():
        print(f"  - {k}: {p}")


if __name__ == "__main__":
    main()

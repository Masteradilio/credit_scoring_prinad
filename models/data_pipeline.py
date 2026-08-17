"""
PRINAD - Data Pipeline & Ingestion Module
=========================================
Handles synthetic data loading, CPF normalization, preprocessing, and feature matrix preparation.

Author: PRINAD Quantitative Risk Team
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import logging
import re

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
SYNTH_DATA_DIR = BASE_DIR / "synth_data"


def normalize_cpf(cpf: Any) -> Optional[str]:
    """
    Normalizes CPF into a standardized 11-digit string.
    """
    if pd.isna(cpf):
        return None
    
    if isinstance(cpf, (int, float)):
        cpf_str = str(int(cpf))
    else:
        cpf_str = str(cpf).strip()
        cpf_str = re.sub(r'[^\d]', '', cpf_str)
    
    return cpf_str.zfill(11) if cpf_str else None


def load_prinad_training_data(
    filepath: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load the synthetic master training dataset (synth_master_training.csv).
    
    Returns:
        Tuple of (full DataFrame, X feature matrix, y target series)
    """
    path = filepath or (SYNTH_DATA_DIR / "synth_master_training.csv")
    if not path.exists():
        logger.warning(f"File {path} not found. Running synthetic generator...")
        from synthetic_data_generator import PRINADDataGenerator
        gen = PRINADDataGenerator()
        gen.save_datasets()
        
    logger.info(f"Loading synthetic master dataset from {path}")
    df = pd.read_csv(path, sep=';', encoding='latin-1')
    
    if 'CLASSE' not in df.columns:
        raise ValueError("Dataset must contain target column 'CLASSE'")
        
    df['TARGET'] = df['CLASSE'].astype(int)
    
    # Metadata columns to exclude from feature matrix
    drop_cols = ['CPF', 'CLIT', 'TARGET', 'CLASSE', 'safra']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols].copy()
    y = df['TARGET'].copy()
    
    logger.info(f"Loaded {len(df)} samples | {len(feature_cols)} features | Default Rate: {y.mean():.2%}")
    return df, X, y


def load_client_database(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load active client records for online inference."""
    path = filepath or (SYNTH_DATA_DIR / "synth_client_database.csv")
    if not path.exists():
        df, _, _ = load_prinad_training_data()
        return df.head(1000)
        
    df = pd.read_csv(path, sep=';', encoding='latin-1')
    df['CPF_NORM'] = df['CPF'].apply(normalize_cpf)
    return df

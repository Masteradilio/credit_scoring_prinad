"""
PRINAD - Data Pipeline Module
Handles data loading, preprocessing, and merging of cadastral and behavioral data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DADOS_DIR = BASE_DIR.parent / "dados"


def normalize_cpf(cpf: Any) -> Optional[str]:
    """
    Normalizes CPF to 11-digit string format.
    
    Args:
        cpf: CPF value in any format (int, float, string)
        
    Returns:
        Normalized 11-digit CPF string or None if invalid
    """
    if pd.isna(cpf):
        return None
    
    # Handle numeric types (int, float) separately from string
    if isinstance(cpf, (int, float)):
        # For numeric values, convert and remove decimal part
        cpf_str = str(int(cpf))
    else:
        # For strings, just strip whitespace
        cpf_str = str(cpf).strip()
        # Remove any non-numeric characters (dots, dashes, etc.)
        cpf_str = re.sub(r'[^\d]', '', cpf_str)
    
    # Pad with zeros to 11 digits
    return cpf_str.zfill(11) if cpf_str else None


def load_cadastro(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads and preprocesses the cadastral data (base_cadastro.csv).
    
    Args:
        filepath: Path to CSV file. Uses default if None.
        
    Returns:
        Preprocessed DataFrame with cadastral data
    """
    filepath = filepath or DADOS_DIR / "base_cadastro.csv"
    logger.info(f"Loading cadastro from {filepath}")
    
    df = pd.read_csv(filepath, sep=';', encoding='latin1')
    logger.info(f"Loaded {len(df)} records from cadastro")
    
    # Normalize CPF
    df['CPF_NORM'] = df['CPF'].apply(normalize_cpf)
    df = df.dropna(subset=['CPF_NORM'])
    
    # Clean categorical columns
    categorical_cols = ['PORTABILIDADE', 'ESCOLARIDADE', 'OCUPACAO', 
                        'ESTADO_CIVIL', 'TIPO_RESIDENCIA', 'POSSUI_VEICULO']
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
    
    # Remove duplicates keeping most recent (by TEMPO_RELAC)
    if 'TEMPO_RELAC' in df.columns:
        df = df.sort_values('TEMPO_RELAC', ascending=False)
        df = df.drop_duplicates(subset=['CPF_NORM'], keep='first')
    else:
        df = df.drop_duplicates(subset=['CPF_NORM'], keep='first')
    
    logger.info(f"After preprocessing: {len(df)} unique clients")
    return df


def load_comportamental(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads and preprocesses the behavioral data (base_3040.csv).
    
    Args:
        filepath: Path to CSV file. Uses default if None.
        
    Returns:
        Preprocessed DataFrame with behavioral data
    """
    filepath = filepath or DADOS_DIR / "base_3040.csv"
    logger.info(f"Loading comportamental from {filepath}")
    
    df = pd.read_csv(filepath, sep=';', encoding='latin1')
    logger.info(f"Loaded {len(df)} records from comportamental")
    
    # Normalize CPF
    df['CPF_NORM'] = df['CPF'].apply(normalize_cpf)
    df = df.dropna(subset=['CPF_NORM'])
    
    # Behavioral columns
    v_cols = ['v205', 'v210', 'v220', 'v230', 'v240', 'v245', 
              'v250', 'v255', 'v260', 'v270', 'v280', 'v290']
    
    # Ensure numeric types
    for col in v_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Aggregate by CPF (latest behavior pattern)
    agg_dict = {col: 'max' for col in v_cols if col in df.columns}
    if 'CLASSE' in df.columns:
        agg_dict['CLASSE'] = 'last'
    
    df_agg = df.groupby('CPF_NORM').agg(agg_dict).reset_index()
    
    logger.info(f"After aggregation: {len(df_agg)} unique clients")
    return df_agg


def merge_datasets(df_cadastro: pd.DataFrame, 
                   df_comportamental: pd.DataFrame) -> pd.DataFrame:
    """
    Merges cadastral and behavioral datasets on CPF.
    
    Args:
        df_cadastro: Cadastral DataFrame
        df_comportamental: Behavioral DataFrame
        
    Returns:
        Merged DataFrame with all features
    """
    logger.info("Merging cadastro and comportamental datasets")
    
    # Inner join on normalized CPF
    df_merged = pd.merge(
        df_cadastro,
        df_comportamental,
        on='CPF_NORM',
        how='inner'
    )
    
    logger.info(f"Merged dataset: {len(df_merged)} records")
    
    # Check match rate
    match_rate = len(df_merged) / len(df_cadastro) * 100
    logger.info(f"Match rate: {match_rate:.1f}%")
    
    if match_rate < 50:
        logger.warning("Low match rate! Check CPF normalization.")
    
    return df_merged


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates binary target variable from CLASSE column.
    
    Args:
        df: DataFrame with CLASSE column
        
    Returns:
        DataFrame with TARGET column (0=good, 1=bad)
    """
    df = df.copy()
    
    if 'CLASSE' in df.columns:
        # Map classes to binary target
        # PERFIL_ADIM = 0 (good), PERFIL_ATRASO/PERFIL_INAD = 1 (bad)
        class_mapping = {
            'PERFIL_ADIM': 0,
            'PERFIL_ATRASO': 1,
            'PERFIL_INAD': 1
        }
        df['TARGET'] = df['CLASSE'].map(class_mapping)
        
        # Drop original class column
        df = df.drop(columns=['CLASSE'], errors='ignore')
    
    return df


def prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepares data for model training.
    
    Args:
        df: Merged DataFrame with all features
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    df = df.copy()
    
    # Columns to drop from features
    drop_cols = ['CPF', 'CPF_NORM', 'CLIT', 'TARGET', 'orig_index']
    
    # Identify feature columns
    feature_cols = [col for col in df.columns if col not in drop_cols]
    
    X = df[feature_cols].copy()
    y = df['TARGET'].copy() if 'TARGET' in df.columns else None
    
    logger.info(f"Features: {len(feature_cols)} columns")
    logger.info(f"Samples: {len(X)} records")
    
    if y is not None:
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y


def load_scr_data(filepath: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """
    Loads SCR (Sistema de Informações de Crédito) data.
    
    Args:
        filepath: Path to SCR CSV file. Uses default if None.
        
    Returns:
        DataFrame with SCR data or None if file doesn't exist
    """
    if filepath is None:
        filepath = DADOS_DIR / "scr_mock_data.csv"
    
    if not filepath.exists():
        logger.warning(f"SCR data file not found: {filepath}")
        return None
    
    logger.info(f"Loading SCR data from {filepath}")
    df_scr = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df_scr)} SCR records")
    
    return df_scr


def load_and_prepare_full_dataset(include_scr: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Main function to load and prepare the complete dataset.
    
    Args:
        include_scr: Whether to include SCR data if available
    
    Returns:
        Tuple of (full DataFrame, features DataFrame, target Series)
    """
    # Load datasets
    df_cadastro = load_cadastro()
    df_comportamental = load_comportamental()
    
    # Merge
    df_merged = merge_datasets(df_cadastro, df_comportamental)
    
    # Create target
    df_merged = create_target_variable(df_merged)
    
    # Load and merge SCR data if available
    if include_scr:
        df_scr = load_scr_data()
        if df_scr is not None:
            # SCR data uses index as CPF (client_idx)
            # Merge by position (both are sorted by same order)
            df_merged = df_merged.reset_index(drop=True)
            
            # Select only SCR columns (not CPF which is the index)
            scr_cols = [c for c in df_scr.columns if c.startswith('scr_')]
            
            # Add SCR columns to merged data
            for col in scr_cols:
                df_merged[col] = df_scr[col].values
            
            logger.info(f"Added {len(scr_cols)} SCR features")
    
    # Prepare for training
    X, y = prepare_training_data(df_merged)
    
    return df_merged, X, y


def load_prinad_training_data(filepath: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load the consolidated PRINAD training dataset (base_prinad_treino.csv).
    
    This is the preferred method for training as it uses the consolidated
    dataset generated by data_consolidator_prinad.py with proper variance.
    
    Args:
        filepath: Path to CSV file. Uses default if None.
        
    Returns:
        Tuple of (full DataFrame, features DataFrame, target Series)
    """
    if filepath is None:
        filepath = DADOS_DIR / 'base_prinad_treino.csv'
    
    if not filepath.exists():
        logger.warning(f"Training dataset not found: {filepath}")
        logger.info("Run data_consolidator_prinad.py first to generate it")
        # Fallback to old method
        return load_and_prepare_full_dataset()
    
    logger.info(f"Loading consolidated PRINAD training data from {filepath}")
    
    df = pd.read_csv(filepath, sep=';', encoding='latin-1')
    logger.info(f"Loaded {len(df)} records")
    
    # Create TARGET from CLASSE
    if 'CLASSE' in df.columns:
        df['TARGET'] = df['CLASSE'].astype(int)
    else:
        logger.error("CLASSE column not found in dataset!")
        raise ValueError("Dataset must contain CLASSE column")
    
    # Columns to drop from features
    drop_cols = ['CPF', 'CLIT', 'TARGET', 'CLASSE', 'profile', 'produto']
    
    # Identify feature columns
    feature_cols = [col for col in df.columns if col not in drop_cols]
    
    X = df[feature_cols].copy()
    y = df['TARGET'].copy()
    
    logger.info(f"Features: {len(feature_cols)} columns")
    logger.info(f"Samples: {len(X)} records")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    logger.info(f"Default rate: {y.mean():.2%}")
    
    return df, X, y

def get_client_data(cpf: str, 
                    df_cadastro: Optional[pd.DataFrame] = None,
                    df_comportamental: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieves all data for a specific client.
    
    Args:
        cpf: Client CPF
        df_cadastro: Cadastral DataFrame (loads if None)
        df_comportamental: Behavioral DataFrame (loads if None)
        
    Returns:
        Dictionary with all client data or None if not found
    """
    cpf_norm = normalize_cpf(cpf)
    
    if df_cadastro is None:
        df_cadastro = load_cadastro()
    if df_comportamental is None:
        df_comportamental = load_comportamental()
    
    # Find in cadastro
    cadastro_row = df_cadastro[df_cadastro['CPF_NORM'] == cpf_norm]
    if cadastro_row.empty:
        logger.warning(f"CPF {cpf} not found in cadastro")
        return None
    
    # Find in comportamental
    comportamental_row = df_comportamental[df_comportamental['CPF_NORM'] == cpf_norm]
    
    # Build result dict
    result = cadastro_row.iloc[0].to_dict()
    
    if not comportamental_row.empty:
        comportamental_data = comportamental_row.iloc[0].to_dict()
        result.update(comportamental_data)
    else:
        # Default behavioral values
        v_cols = ['v205', 'v210', 'v220', 'v230', 'v240', 'v245', 
                  'v250', 'v255', 'v260', 'v270', 'v280', 'v290']
        for col in v_cols:
            result[col] = 0.0
    
    return result


if __name__ == "__main__":
    # Test the pipeline
    logger.info("Testing data pipeline...")
    
    df_full, X, y = load_and_prepare_full_dataset()
    
    print(f"\n{'='*50}")
    print("DATA PIPELINE SUMMARY")
    print(f"{'='*50}")
    print(f"Total records: {len(df_full)}")
    print(f"Feature columns: {len(X.columns)}")
    print(f"Target distribution:")
    print(y.value_counts())
    print(f"\nFeature columns: {list(X.columns)}")

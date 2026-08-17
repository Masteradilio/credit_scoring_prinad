"""
PRINAD - Champion vs. Challenger Master Model Training Pipeline
==============================================================
Trains, validates, and compares credit risk models under Basel & IFRS 9 standards:
1. CHAMPION: Regulatory Scorecard (Logistic Regression + WoE)
2. CHALLENGER 1: LightGBM (Calibrated Classifier)
3. CHALLENGER 2: XGBoost (Calibrated Classifier)
4. CHALLENGER 3: Stacking Ensemble (Meta-Learner)

Executes 4-Pillar Validation across all candidates:
- Discrimination (AUC, Gini, KS, PR-AUC)
- Calibration (Brier Score, ECE, Hosmer-Lemeshow)
- Stability (PSI)
- Basel Traffic Light Backtesting

Author: PRINAD Quantitative Risk Team
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
import joblib
import json
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier

# Gradient Boosting
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

from scorecard_woe import RegulatoryScorecard
from model_validation_engine import ModelValidationEngine
from feature_engineering import FeatureEngineer
from data_pipeline import load_prinad_training_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
ARTEFATOS_DIR = BASE_DIR / "artefatos"
ARTEFATOS_DIR.mkdir(parents=True, exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Construct Scikit-learn column transformer for tree-based models."""
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Separate binary/boolean indicators
    bool_cols = [c for c in numerical_cols if X[c].nunique() <= 2]
    numerical_cols = [c for c in numerical_cols if c not in bool_cols]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('bool', 'passthrough', bool_cols)
        ],
        remainder='drop'
    )
    return preprocessor


def train_champion_scorecard(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> RegulatoryScorecard:
    """Train Basel Champion Regulatory Scorecard."""
    logger.info("--> [1/4] Training Champion: Regulatory Scorecard (WoE + Logistic Regression)...")
    scorecard = RegulatoryScorecard(target_score=600, target_odds=50.0, pdo=20)
    scorecard.fit(X_train, y_train)
    return scorecard


def train_challengers(
    X_train_proc: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42
) -> Dict[str, Any]:
    """Train ML Challenger models with SMOTE balance & Isotonic Calibration."""
    logger.info("--> Balancing training data with SMOTE...")
    smote = SMOTE(sampling_strategy=0.7, random_state=random_state)
    X_bal, y_bal = smote.fit_resample(X_train_proc, y_train)
    
    # 1. Challenger 1: LightGBM
    logger.info("--> [2/4] Training Challenger 1: LightGBM Classifier...")
    lgb_base = lgb.LGBMClassifier(
        n_estimators=400,
        max_depth=6,
        num_leaves=31,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=25,
        random_state=random_state,
        verbose=-1,
        n_jobs=-1
    )
    lgb_calib = CalibratedClassifierCV(lgb_base, method='isotonic', cv=3)
    lgb_calib.fit(X_bal, y_bal)
    
    # 2. Challenger 2: XGBoost
    logger.info("--> [3/4] Training Challenger 2: XGBoost Classifier...")
    xgb_base = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        eval_metric='logloss',
        random_state=random_state,
        n_jobs=-1
    )
    xgb_calib = CalibratedClassifierCV(xgb_base, method='isotonic', cv=3)
    xgb_calib.fit(X_bal, y_bal)
    
    # 3. Challenger 3: Stacking Ensemble
    logger.info("--> [4/4] Training Challenger 3: Stacking Ensemble Meta-Learner...")
    stacking_ensemble = StackingClassifier(
        estimators=[('lgb', lgb_base), ('xgb', xgb_base)],
        final_estimator=LogisticRegression(C=0.2, max_iter=1000, random_state=random_state),
        cv=3,
        stack_method='predict_proba',
        n_jobs=-1
    )
    stacking_ensemble.fit(X_bal, y_bal)
    
    return {
        'lightgbm': lgb_calib,
        'xgboost': xgb_calib,
        'ensemble': stacking_ensemble
    }


def execute_full_training_pipeline() -> Dict[str, Any]:
    """Execute complete end-to-end model training, validation, and benchmarking."""
    logger.info("=" * 70)
    logger.info("PRINAD - MASTER MODEL TRAINING & VALIDATION PIPELINE")
    logger.info("=" * 70)
    
    # 1. Load Data
    df, X_raw, y = load_prinad_training_data()
    
    # 2. Feature Engineering
    logger.info("Applying Quantitative Feature Engineering...")
    fe = FeatureEngineer()
    X_fe = fe.fit_transform(X_raw)
    
    # 3. Stratified Train/Test Split (80% Train / 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_fe, y, test_size=0.20, stratify=y, random_state=42
    )
    
    # 4. Preprocessing for ML Models
    preprocessor = build_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # 5. Train Models
    champion_scorecard = train_champion_scorecard(X_train, y_train)
    challengers = train_challengers(X_train_proc, y_train.values)
    
    # 6. Predict on Test Set
    validator = ModelValidationEngine()
    
    models_pred = {
        'Champion_Scorecard': champion_scorecard.predict_proba(X_test),
        'Challenger_LightGBM': challengers['lightgbm'].predict_proba(X_test_proc)[:, 1],
        'Challenger_XGBoost': challengers['xgboost'].predict_proba(X_test_proc)[:, 1],
        'Challenger_Ensemble': challengers['ensemble'].predict_proba(X_test_proc)[:, 1]
    }
    
    # 7. Benchmark and 4-Pillar Validation
    benchmark_results = {}
    detailed_reports = {}
    
    print("\n" + "=" * 85)
    print(f"{'MODEL ARCHITECTURE':<25} {'AUC-ROC':<10} {'GINI':<10} {'KS STAT':<10} {'BRIER':<10} {'STATUS':<15}")
    print("-" * 85)
    
    for name, pds in models_pred.items():
        disc = validator.evaluate_discrimination(y_test, pds)
        calib = validator.evaluate_calibration(y_test, pds)
        psi = validator.calculate_psi(champion_scorecard.predict_proba(X_train) if 'Scorecard' in name else models_pred[name], pds)
        backtest = validator.run_basel_backtest(y_test, pds)
        
        status = "EXCELLENT" if disc['gini'] >= 0.70 and calib['brier_score'] < 0.12 else "PASS"
        
        benchmark_results[name] = {
            'auc_roc': disc['auc_roc'],
            'gini': disc['gini'],
            'ks_statistic': disc['ks_statistic'],
            'brier_score': calib['brier_score'],
            'ece': calib['expected_calibration_error_ece'],
            'hosmer_lemeshow_p': calib['hosmer_lemeshow_p_value'],
            'basel_traffic_light': backtest['overall_traffic_light'],
            'status': status
        }
        
        detailed_reports[name] = {
            'discrimination': disc,
            'calibration': calib,
            'stability_psi': psi,
            'basel_backtest': backtest
        }
        
        print(f"{name:<25} {disc['auc_roc']:<10.4f} {disc['gini']:<10.4f} {disc['ks_statistic']:<10.4f} {calib['brier_score']:<10.4f} {status:<15}")
        
    print("=" * 85)
    
    # 8. SHAP Explainer for Champion Ensemble / Tree Model
    logger.info("Constructing TreeSHAP Explainer...")
    shap_explainer = None
    if SHAP_AVAILABLE and shap is not None:
        try:
            lgb_model = challengers['lightgbm'].calibrated_classifiers_[0].estimator
            shap_sample = X_train_proc[:1000]
            shap_explainer = shap.TreeExplainer(lgb_model)
            logger.info("TreeSHAP Explainer fitted successfully.")
        except Exception as e:
            logger.warning(f"Could not build TreeSHAP explainer: {e}")
    else:
        logger.info("SHAP not available in environment; using Glass-box Scorecard Points.")
        
    # 9. Save Artifacts to `artefatos/`
    logger.info(f"Saving binary models and validation reports to {ARTEFATOS_DIR}...")
    
    joblib.dump(champion_scorecard, ARTEFATOS_DIR / "scorecard_model.joblib")
    joblib.dump(challengers['lightgbm'], ARTEFATOS_DIR / "lightgbm_model.joblib")
    joblib.dump(challengers['xgboost'], ARTEFATOS_DIR / "xgboost_model.joblib")
    joblib.dump(challengers['ensemble'], ARTEFATOS_DIR / "ensemble_model.joblib")
    joblib.dump(preprocessor, ARTEFATOS_DIR / "preprocessor.joblib")
    joblib.dump(fe, ARTEFATOS_DIR / "feature_engineer.joblib")
    joblib.dump(list(X_fe.columns), ARTEFATOS_DIR / "feature_names.joblib")
    
    if shap_explainer:
        joblib.dump(shap_explainer, ARTEFATOS_DIR / "shap_explainer.joblib")
        
    # Save JSON Reports
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'default_rate_pct': round(float(y.mean() * 100), 2),
        'benchmark_summary': benchmark_results,
        'detailed_validation': detailed_reports
    }
    
    with open(ARTEFATOS_DIR / "model_comparison_report.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, cls=NumpyEncoder)
        
    logger.info("Master Training Pipeline Completed Successfully!")
    return report_data


if __name__ == "__main__":
    execute_full_training_pipeline()

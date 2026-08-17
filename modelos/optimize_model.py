"""
PRINAD - Model Optimization Module
Hyperparameter tuning with Optuna and threshold optimization.
Target: Precision > 0.90 and Recall > 0.90 for the minority class.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import logging
from datetime import datetime
import json
import joblib
import warnings

# Optuna for hyperparameter optimization
import optuna
from optuna.samplers import TPESampler

# ML Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    precision_recall_curve, classification_report, confusion_matrix
)
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Gradient Boosting
import xgboost as xgb
import lightgbm as lgb

# Imbalanced learning
from imblearn.over_sampling import SMOTE

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent))
from data_pipeline import load_and_prepare_full_dataset
from feature_engineering import FeatureEngineer

# Suppress warnings during optimization
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELO_DIR = BASE_DIR / "modelo"
MODELO_DIR.mkdir(exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class ModelOptimizer:
    """
    Hyperparameter optimization using Optuna with focus on
    achieving Precision > 0.90 and Recall > 0.90.
    """
    
    def __init__(self, random_state: int = 42, n_trials: int = 30):
        self.random_state = random_state
        self.n_trials = n_trials
        self.best_params = None
        self.best_threshold = 0.5
        self.best_model = None
        self.preprocessor = None
        self.feature_engineer = FeatureEngineer()
        self.label_encoders = {}
        self.optimization_history = []
        
    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load, engineer features, and split data."""
        
        logger.info("Loading and preparing data...")
        df_full, X, y = load_and_prepare_full_dataset()
        
        # Feature engineering
        X_engineered = self.feature_engineer.fit_transform(X)
        X_engineered = X_engineered.fillna(0)
        
        # Create preprocessor
        self.preprocessor = self._create_preprocessor(X_engineered)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_engineered, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        
        # Fit preprocessor
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        return X_train_processed, X_test_processed, y_train.values, y_test.values
    
    def _create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """Create preprocessing pipeline."""
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        bool_cols = [col for col in numerical_cols if X[col].nunique() <= 2]
        numerical_cols = [col for col in numerical_cols if col not in bool_cols]
        
        return ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
                ('bool', 'passthrough', bool_cols)
            ],
            remainder='drop'
        )
    
    def _balance_data(self, X: np.ndarray, y: np.ndarray, 
                      smote_k: int = 5, target_ratio: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
        """Balance data with SMOTE."""
        
        unique, counts = np.unique(y, return_counts=True)
        majority_count = max(counts)
        minority_class = unique[np.argmin(counts)]
        target_minority = int(majority_count * target_ratio)
        
        smote = SMOTE(
            sampling_strategy={int(minority_class): target_minority},
            random_state=self.random_state,
            k_neighbors=smote_k
        )
        
        return smote.fit_resample(X, y)
    
    def _objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Optuna objective function - maximize combined precision + recall."""
        
        # Hyperparameters to tune
        xgb_params = {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 400),
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
            'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('xgb_colsample', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1e-3, 10.0, log=True),
            'scale_pos_weight': trial.suggest_float('xgb_scale_pos_weight', 1.0, 5.0),
        }
        
        lgb_params = {
            'n_estimators': trial.suggest_int('lgb_n_estimators', 100, 400),
            'max_depth': trial.suggest_int('lgb_max_depth', 3, 10),
            'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('lgb_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('lgb_colsample', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('lgb_min_child_samples', 5, 50),
            'reg_alpha': trial.suggest_float('lgb_reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('lgb_reg_lambda', 1e-3, 10.0, log=True),
        }
        
        smote_k = trial.suggest_int('smote_k', 3, 10)
        target_ratio = trial.suggest_float('target_ratio', 0.5, 0.9)
        
        try:
            # Balance training data
            X_train_bal, y_train_bal = self._balance_data(X_train, y_train, smote_k, target_ratio)
            
            # Create models
            xgb_model = xgb.XGBClassifier(
                **xgb_params,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1
            )
            
            lgb_model = lgb.LGBMClassifier(
                **lgb_params,
                random_state=self.random_state,
                verbose=-1,
                n_jobs=-1
            )
            
            # Create ensemble
            ensemble = StackingClassifier(
                estimators=[('xgb', xgb_model), ('lgb', lgb_model)],
                final_estimator=LogisticRegression(C=0.1, max_iter=1000),
                cv=3,
                stack_method='predict_proba',
                n_jobs=-1
            )
            
            # Train
            ensemble.fit(X_train_bal, y_train_bal)
            
            # Predict probabilities on validation
            y_proba = ensemble.predict_proba(X_val)[:, 1]
            
            # Find optimal threshold for our targets
            precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)
            
            # Target: both precision and recall > 0.90
            best_f1 = 0
            best_thresh = 0.5
            
            for i, thresh in enumerate(thresholds):
                if precisions[i] >= 0.85 and recalls[i] >= 0.85:  # Relaxed for search
                    f1 = 2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i])
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thresh = thresh
            
            # Use optimal threshold for evaluation
            y_pred = (y_proba >= best_thresh).astype(int)
            
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_proba)
            
            # Combined objective: weighted sum prioritizing precision and recall
            # Bonus if both are >= 0.90
            score = (precision + recall) / 2
            if precision >= 0.90 and recall >= 0.90:
                score += 0.1  # Bonus
            
            # Store best threshold
            trial.set_user_attr('best_threshold', best_thresh)
            trial.set_user_attr('precision', precision)
            trial.set_user_attr('recall', recall)
            trial.set_user_attr('auc', auc)
            
            return score
            
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return 0.0
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Returns:
            Dictionary with best parameters and metrics
        """
        print("\n" + "="*70)
        print("PRINAD MODEL OPTIMIZATION")
        print("Target: Precision > 0.90 AND Recall > 0.90")
        print("="*70)
        
        # Prepare data
        X_train, X_test, y_train, y_test = self._prepare_data()
        
        # Further split train for optimization
        X_train_opt, X_val, y_train_opt, y_val = train_test_split(
            X_train, y_train, test_size=0.25, stratify=y_train, random_state=self.random_state
        )
        
        print(f"\nOptimization set: {len(X_train_opt)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"\nRunning {self.n_trials} optimization trials...")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        
        # Optimize
        study.optimize(
            lambda trial: self._objective(trial, X_train_opt, y_train_opt, X_val, y_val),
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        # Get best params
        self.best_params = study.best_params
        self.best_threshold = study.best_trial.user_attrs.get('best_threshold', 0.5)
        
        print(f"\n✅ Best trial score: {study.best_value:.4f}")
        print(f"   Threshold: {self.best_threshold:.4f}")
        print(f"   Precision: {study.best_trial.user_attrs.get('precision', 0):.4f}")
        print(f"   Recall: {study.best_trial.user_attrs.get('recall', 0):.4f}")
        
        # Train final model with best params on full training data
        print("\n🔄 Training final model with best parameters...")
        final_metrics = self._train_final_model(X_train, y_train, X_test, y_test)
        
        # Save artifacts
        self._save_artifacts(final_metrics)
        
        return {
            'best_params': self.best_params,
            'best_threshold': self.best_threshold,
            'optimization_score': study.best_value,
            'final_metrics': final_metrics,
            'n_trials': self.n_trials
        }
    
    def _train_final_model(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train final model with optimized parameters."""
        
        # Extract params
        xgb_params = {k.replace('xgb_', ''): v for k, v in self.best_params.items() if k.startswith('xgb_')}
        lgb_params = {k.replace('lgb_', ''): v for k, v in self.best_params.items() if k.startswith('lgb_')}
        smote_k = self.best_params.get('smote_k', 5)
        target_ratio = self.best_params.get('target_ratio', 0.7)
        
        # Balance
        X_train_bal, y_train_bal = self._balance_data(X_train, y_train, smote_k, target_ratio)
        
        logger.info(f"Balanced training: {len(X_train_bal)} samples")
        logger.info(f"Distribution: {np.bincount(y_train_bal.astype(int))}")
        
        # Handle min_child_weight rename for lgb
        if 'min_child_samples' not in lgb_params and 'min_child_weight' in lgb_params:
            lgb_params['min_child_samples'] = lgb_params.pop('min_child_weight')
        if 'colsample' in lgb_params:
            lgb_params['colsample_bytree'] = lgb_params.pop('colsample')
        
        # Handle xgb colsample rename
        if 'colsample' in xgb_params:
            xgb_params['colsample_bytree'] = xgb_params.pop('colsample')
        
        # Create models
        xgb_model = xgb.XGBClassifier(
            **xgb_params,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        
        lgb_model = lgb.LGBMClassifier(
            **lgb_params,
            random_state=self.random_state,
            verbose=-1,
            n_jobs=-1
        )
        
        # Create ensemble
        self.best_model = StackingClassifier(
            estimators=[('xgb', xgb_model), ('lgb', lgb_model)],
            final_estimator=LogisticRegression(C=0.1, max_iter=1000),
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        # Train
        self.best_model.fit(X_train_bal, y_train_bal)
        
        # Evaluate with optimized threshold
        y_proba = self.best_model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= self.best_threshold).astype(int)
        
        # Metrics
        auc = roc_auc_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        gini = 2 * auc - 1
        
        # Also get default threshold metrics for comparison
        y_pred_default = (y_proba >= 0.5).astype(int)
        precision_default = precision_score(y_test, y_pred_default)
        recall_default = recall_score(y_test, y_pred_default)
        
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'auc_roc': round(auc, 4),
            'gini': round(gini, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'threshold': round(self.best_threshold, 4),
            'precision_default_05': round(precision_default, 4),
            'recall_default_05': round(recall_default, 4),
            'confusion_matrix': conf_matrix.tolist(),
            'samples_train': len(X_train_bal),
            'samples_test': len(X_test)
        }
        
        # Print results
        print("\n" + "="*70)
        print("FINAL MODEL RESULTS")
        print("="*70)
        print(f"\n📊 Metrics with optimized threshold ({self.best_threshold:.4f}):")
        print(f"   AUC-ROC:   {auc:.4f}")
        print(f"   Gini:      {gini:.4f}")
        print(f"   Precision: {precision:.4f} {'✅' if precision >= 0.90 else '⚠️'}")
        print(f"   Recall:    {recall:.4f} {'✅' if recall >= 0.90 else '⚠️'}")
        print(f"   F1-Score:  {f1:.4f}")
        
        print(f"\n📊 Comparison with default threshold (0.5):")
        print(f"   Precision: {precision_default:.4f} → {precision:.4f} ({'+' if precision > precision_default else ''}{(precision - precision_default)*100:.1f}%)")
        print(f"   Recall:    {recall_default:.4f} → {recall:.4f} ({'+' if recall > recall_default else ''}{(recall - recall_default)*100:.1f}%)")
        
        print(f"\n📋 Confusion Matrix (rows=actual, cols=predicted):")
        print(f"   [[TN={conf_matrix[0,0]:>5}, FP={conf_matrix[0,1]:>5}]")
        print(f"    [FN={conf_matrix[1,0]:>5}, TP={conf_matrix[1,1]:>5}]]")
        
        print("="*70)
        
        return metrics
    
    def _save_artifacts(self, metrics: Dict[str, Any]):
        """Save optimized model and artifacts."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = MODELO_DIR / "optimized_model.joblib"
        joblib.dump({
            'model': self.best_model,
            'threshold': self.best_threshold,
            'best_params': self.best_params,
            'timestamp': timestamp,
            'version': '2.0.0'
        }, model_path)
        logger.info(f"Optimized model saved to {model_path}")
        
        # Save preprocessor
        preproc_path = MODELO_DIR / "optimized_preprocessor.joblib"
        joblib.dump({
            'preprocessor': self.preprocessor,
            'feature_engineer': self.feature_engineer
        }, preproc_path)
        logger.info(f"Preprocessor saved to {preproc_path}")
        
        # Save optimization report
        report = {
            'timestamp': timestamp,
            'n_trials': self.n_trials,
            'best_params': self.best_params,
            'best_threshold': self.best_threshold,
            'metrics': metrics
        }
        
        report_path = MODELO_DIR / "optimization_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        logger.info(f"Optimization report saved to {report_path}")


def optimize_model(n_trials: int = 30) -> Dict[str, Any]:
    """
    Main function to run model optimization.
    
    Args:
        n_trials: Number of Optuna trials
        
    Returns:
        Optimization results
    """
    optimizer = ModelOptimizer(random_state=42, n_trials=n_trials)
    results = optimizer.optimize()
    
    precision = results['final_metrics']['precision']
    recall = results['final_metrics']['recall']
    
    if precision >= 0.90 and recall >= 0.90:
        print("\n🎉 SUCCESS! Both Precision and Recall are >= 0.90!")
    else:
        print("\n⚠️ Targets not fully met. Consider running with more trials or adjusting cost weights.")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PRINAD Model Optimization")
    parser.add_argument('--trials', type=int, default=30, help='Number of optimization trials')
    args = parser.parse_args()
    
    results = optimize_model(n_trials=args.trials)

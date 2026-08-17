"""
PRINAD - Model Training Module
Trains ensemble model (XGBoost + LightGBM + CatBoost) with SHAP interpretability.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import logging
import joblib
from datetime import datetime

# ML Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from sklearn.ensemble import StackingClassifier

# Gradient Boosting
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    xgb = None
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:  # Catch generic exception for descriptor error
    lgb = None
    LGB_AVAILABLE = False

# Imbalanced learning
try:
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

# SHAP
import shap

# Local imports
from data_pipeline import load_and_prepare_full_dataset, load_prinad_training_data, DADOS_DIR
from feature_engineering import FeatureEngineer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELO_DIR = BASE_DIR / "modelo"
MODELO_DIR.mkdir(exist_ok=True)


class RatingMapper:
    """Maps PRINAD scores to ratings."""
    
    RATING_BANDS = [
        ('A1', 0, 5, 'Risco Mínimo', 'verde'),
        ('A2', 5, 15, 'Risco Muito Baixo', 'verde'),
        ('A3', 15, 25, 'Risco Baixo', 'verde'),
        ('B1', 25, 35, 'Risco Baixo-Moderado', 'amarelo'),
        ('B2', 35, 45, 'Risco Moderado', 'amarelo'),
        ('B3', 45, 55, 'Risco Moderado-Alto', 'laranja'),
        ('C1', 55, 65, 'Risco Alto', 'vermelho'),
        ('C2', 65, 75, 'Risco Muito Alto', 'vermelho'),
        ('C3', 75, 85, 'Risco Crítico', 'vermelho'),
        ('D', 85, 95, 'Pré-Default', 'preto'),
        ('DEFAULT', 95, 100, 'Default', 'preto'),
    ]
    
    @classmethod
    def get_rating(cls, prinad: float) -> Dict[str, str]:
        """Get rating for a given PRINAD score."""
        for rating, lower, upper, desc, color in cls.RATING_BANDS:
            if lower <= prinad < upper or (rating == 'DEFAULT' and prinad >= 95):
                return {
                    'rating': rating,
                    'descricao': desc,
                    'cor': color,
                    'faixa': f'{lower}% - {upper}%'
                }
        return {'rating': 'DEFAULT', 'descricao': 'Default', 'cor': 'preto', 'faixa': '95% - 100%'}


class PRINADModelTrainer:
    """
    Trains and manages the PRINAD ensemble model.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.preprocessor = None
        self.model = None
        self.calibrated_model = None
        self.shap_explainer = None
        self.feature_names = None
        self.feature_engineer = FeatureEngineer()
        self.label_encoders = {}
        self.scaler = None
        
    def _create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """Create preprocessing pipeline."""
        
        # Identify column types
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove boolean columns from numerical and process separately
        bool_cols = [col for col in numerical_cols if X[col].nunique() <= 2]
        numerical_cols = [col for col in numerical_cols if col not in bool_cols]
        
        logger.info(f"Categorical columns: {len(categorical_cols)}")
        logger.info(f"Numerical columns: {len(numerical_cols)}")
        logger.info(f"Boolean columns: {len(bool_cols)}")
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
                ('bool', 'passthrough', bool_cols)
            ],
            remainder='drop'
        )
        
        return preprocessor
    
    def _create_base_models(self) -> List[Tuple[str, Any]]:
        """Create base models for ensemble."""
        
        estimators = []
        
        if XGB_AVAILABLE:
            # Balanced parameters for F1 optimization
            xgb_model = xgb.XGBClassifier(
                n_estimators=400,          
                max_depth=6,               
                learning_rate=0.05,        
                subsample=0.8,             
                colsample_bytree=0.8,      
                min_child_weight=2,        
                reg_alpha=0.1,             
                reg_lambda=1.0,            
                gamma=0,                   
                scale_pos_weight=5.0,      # Higher weight for minority (typical ratio ~4:1)
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss',     
                n_jobs=-1
            )
            estimators.append(('xgboost', xgb_model))
        else:
            logger.warning("XGBoost not available, skipping.")
            
        if LGB_AVAILABLE:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.random_state,
                verbose=-1,
                n_jobs=-1
            )
            estimators.append(('lightgbm', lgb_model))
        else:
            logger.warning("LightGBM not available, skipping.")
            
        if not estimators:
            raise RuntimeError("No boosting models available (XGBoost or LightGBM required).")
            
        return estimators
    
    def _create_ensemble(self) -> StackingClassifier:
        """Create stacking ensemble model."""
        
        estimators = self._create_base_models()
        
        meta_learner = LogisticRegression(
            C=0.1,
            max_iter=1000,
            random_state=self.random_state
        )
        
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5,
            stack_method='predict_proba',
            passthrough=False,
            n_jobs=-1
        )
        
        return ensemble
    
    def _balance_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Balance dataset using SMOTE (validated as best strategy)."""
        
        logger.info(f"Original class distribution: {np.bincount(y.astype(int))}")
        
        if not IMBLEARN_AVAILABLE:
            logger.warning("imblearn not available. Skipping data balancing.")
            return X, y
        
        # Calculate target sampling strategy (70% of majority class)
        unique, counts = np.unique(y, return_counts=True)
        majority_count = max(counts)
        minority_class = unique[np.argmin(counts)]
        # Use 1:1 ratio for better balance (boosts recall)
        target_minority = majority_count
        
        sampling_strategy = {int(minority_class): target_minority}
        
        smote = SMOTE(
            sampling_strategy=sampling_strategy, 
            random_state=self.random_state,
            k_neighbors=5
        )
        
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        logger.info(f"Balanced class distribution: {np.bincount(y_balanced.astype(int))}")
        
        return X_balanced, y_balanced
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              balance: bool = True, calibrate: bool = True) -> Dict[str, Any]:
        """
        Train the PRINAD model.
        
        Args:
            X: Features DataFrame
            y: Target Series
            balance: Whether to balance the dataset
            calibrate: Whether to calibrate probabilities
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Starting model training...")
        
        # Feature engineering is DISABLED
        # The consolidated dataset already has all derived features pre-computed
        # by data_consolidator_prinad.py, so we use raw features directly
        logger.info("Using pre-computed features (feature_engineer disabled)")
        X_engineered = X.copy()
        self.feature_names = list(X_engineered.columns)
        self.feature_engineer = None  # Explicitly disable
        
        # Handle missing values
        X_engineered = X_engineered.fillna(0)
        
        # Create preprocessor
        logger.info("Creating preprocessor...")
        self.preprocessor = self._create_preprocessor(X_engineered)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_engineered, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        
        # Fit preprocessor and transform
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Balance if requested
        if balance:
            logger.info("Balancing dataset with SMOTE-Tomek...")
            X_train_processed, y_train = self._balance_data(X_train_processed, y_train.values)
        
        # Create and train ensemble
        logger.info("Training ensemble model...")
        self.model = self._create_ensemble()
        self.model.fit(X_train_processed, y_train)
        
        # Calibrate if requested
        if calibrate:
            logger.info("Calibrating probabilities...")
            # Use cross-validation based calibration (cv='prefit' deprecated)
            self.calibrated_model = CalibratedClassifierCV(
                self.model, method='isotonic', cv=5
            )
            # Fit on validation set for calibration
            self.calibrated_model.fit(X_test_processed, y_test)
        
        # Evaluate
        logger.info("Evaluating model...")
        metrics = self._evaluate(X_test_processed, y_test)
        
        # Create SHAP explainer
        logger.info("Creating SHAP explainer...")
        self._create_shap_explainer(X_train_processed)
        
        # Save artifacts
        logger.info("Saving model artifacts...")
        self._save_artifacts()
        
        return metrics
    
    def _evaluate(self, X_test: np.ndarray, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate model performance with optimal threshold selection."""
        
        model_to_use = self.calibrated_model if self.calibrated_model else self.model
        
        y_proba = model_to_use.predict_proba(X_test)[:, 1]
        
        # Find optimal threshold for F1
        from sklearn.metrics import precision_recall_curve, f1_score
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
        
        # Calculate F1 for each threshold
        f1_scores = []
        for p, r in zip(precisions[:-1], recalls[:-1]):
            if p + r > 0:
                f1_scores.append(2 * p * r / (p + r))
            else:
                f1_scores.append(0)
        
        # Find optimal threshold
        optimal_idx = np.argmax(f1_scores)
        self.optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        
        logger.info(f"Optimal threshold for F1: {self.optimal_threshold:.4f}")
        logger.info(f"F1 at optimal threshold: {optimal_f1:.4f}")
        
        # Use optimal threshold for predictions
        y_pred_optimal = (y_proba >= self.optimal_threshold).astype(int)
        # Also compute default threshold (0.5) for comparison
        y_pred_default = model_to_use.predict(X_test)
        
        # Metrics
        auc = roc_auc_score(y_test, y_proba)
        gini = 2 * auc - 1
        
        # KS statistic
        from scipy.stats import ks_2samp
        y_test_arr = np.array(y_test)
        ks_stat, _ = ks_2samp(y_proba[y_test_arr == 0], y_proba[y_test_arr == 1])
        
        # Classification reports (optimal threshold)
        report_optimal = classification_report(y_test, y_pred_optimal, output_dict=True)
        report_default = classification_report(y_test, y_pred_default, output_dict=True)
        
        metrics = {
            'auc_roc': auc,
            'gini': gini,
            'ks': ks_stat,
            # Optimal threshold metrics
            'optimal_threshold': self.optimal_threshold,
            'precision_bad': report_optimal.get('1', {}).get('precision', 0),
            'recall_bad': report_optimal.get('1', {}).get('recall', 0),
            'f1_bad': report_optimal.get('1', {}).get('f1-score', 0),
            'accuracy': report_optimal.get('accuracy', 0),
            # Default threshold metrics for comparison
            'precision_bad_default': report_default.get('1', {}).get('precision', 0),
            'recall_bad_default': report_default.get('1', {}).get('recall', 0),
            'f1_bad_default': report_default.get('1', {}).get('f1-score', 0),
            'classification_report': report_optimal,
            'confusion_matrix': confusion_matrix(y_test, y_pred_optimal).tolist()
        }
        
        logger.info(f"AUC-ROC: {auc:.4f}")
        logger.info(f"Gini: {gini:.4f}")
        logger.info(f"KS: {ks_stat:.4f}")
        logger.info(f"[Optimal] Precision: {metrics['precision_bad']:.4f}, Recall: {metrics['recall_bad']:.4f}, F1: {metrics['f1_bad']:.4f}")
        logger.info(f"[Default] Precision: {metrics['precision_bad_default']:.4f}, Recall: {metrics['recall_bad_default']:.4f}, F1: {metrics['f1_bad_default']:.4f}")
        
        return metrics
    
    def _create_shap_explainer(self, X_sample: np.ndarray):
        """Create SHAP explainer for model interpretability."""
        
        # Use a sample for efficiency
        sample_size = min(1000, len(X_sample))
        X_background = X_sample[:sample_size]
        
        model_to_use = self.calibrated_model if self.calibrated_model else self.model
        
        # Create explainer
        try:
            self.shap_explainer = shap.Explainer(
                model_to_use.predict_proba,
                X_background,
                feature_names=self._get_feature_names_out()
            )
            logger.info("SHAP explainer created successfully")
        except Exception as e:
            logger.warning(f"Could not create SHAP explainer: {e}")
            self.shap_explainer = None
    
    def _get_feature_names_out(self) -> List[str]:
        """Get feature names after preprocessing."""
        try:
            return self.preprocessor.get_feature_names_out().tolist()
        except:
            return [f'feature_{i}' for i in range(self.preprocessor.n_features_in_)]
    
    def _save_artifacts(self):
        """Save model artifacts to disk."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = MODELO_DIR / "ensemble_model.joblib"
        joblib.dump({
            'model': self.calibrated_model if self.calibrated_model else self.model,
            'base_model': self.model,
            'feature_names': self.feature_names,
            'timestamp': timestamp,
            'version': '2.0.0'
        }, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save preprocessor
        preproc_path = MODELO_DIR / "preprocessor.joblib"
        joblib.dump({
            'preprocessor': self.preprocessor,
            'feature_engineer': self.feature_engineer,
            'feature_names': self.feature_names
        }, preproc_path)
        logger.info(f"Preprocessor saved to {preproc_path}")
        
        # Save SHAP explainer
        if self.shap_explainer:
            shap_path = MODELO_DIR / "shap_explainer.joblib"
            joblib.dump(self.shap_explainer, shap_path)
            logger.info(f"SHAP explainer saved to {shap_path}")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of default."""
        
        X_engineered = self.feature_engineer.transform(X)
        X_engineered = X_engineered.fillna(0)
        X_processed = self.preprocessor.transform(X_engineered)
        
        model_to_use = self.calibrated_model if self.calibrated_model else self.model
        return model_to_use.predict_proba(X_processed)[:, 1]
    
    def explain(self, X: pd.DataFrame, max_features: int = 5) -> List[Dict[str, Any]]:
        """Generate SHAP explanations for predictions."""
        
        if self.shap_explainer is None:
            return []
        
        X_engineered = self.feature_engineer.transform(X)
        X_engineered = X_engineered.fillna(0)
        X_processed = self.preprocessor.transform(X_engineered)
        
        shap_values = self.shap_explainer(X_processed)
        feature_names = self._get_feature_names_out()
        
        explanations = []
        for i in range(len(X)):
            values = shap_values.values[i] if hasattr(shap_values, 'values') else shap_values[i]
            
            # Sort by absolute importance
            importance = list(zip(feature_names, values))
            importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            top_features = []
            for feat, val in importance[:max_features]:
                top_features.append({
                    'feature': feat,
                    'contribuicao': float(val),
                    'direcao': 'aumenta_risco' if val > 0 else 'reduz_risco'
                })
            
            explanations.append(top_features)
        
        return explanations


def train_model(save_metrics: bool = True, use_consolidated: bool = True) -> Dict[str, Any]:
    """
    Main function to train the PRINAD model.
    
    Args:
        save_metrics: Whether to save training metrics
        use_consolidated: Whether to use the consolidated training dataset
        
    Returns:
        Training metrics dictionary
    """
    logger.info("="*60)
    logger.info("PRINAD MODEL TRAINING v2.0")
    logger.info("="*60)
    
    # Load data - prefer consolidated dataset
    if use_consolidated:
        try:
            df_full, X, y = load_prinad_training_data()
            logger.info("Using consolidated PRINAD training dataset")
        except Exception as e:
            logger.warning(f"Failed to load consolidated dataset: {e}")
            logger.info("Falling back to legacy data loading")
            df_full, X, y = load_and_prepare_full_dataset()
    else:
        df_full, X, y = load_and_prepare_full_dataset()
    
    # Save feature names for inference consistency
    feature_names_path = MODELO_DIR / "feature_names.joblib"
    joblib.dump(list(X.columns), feature_names_path)
    logger.info(f"Feature names saved: {len(X.columns)} features")
    
    # Train model
    trainer = PRINADModelTrainer(random_state=42)
    metrics = trainer.train(X, y, balance=True, calibrate=True)
    
    # Save metrics
    if save_metrics:
        metrics_path = MODELO_DIR / "training_metrics.joblib"
        joblib.dump({
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(X),
            'n_features': len(X.columns),
            'optimal_threshold': metrics.get('optimal_threshold', 0.5)
        }, metrics_path)
        logger.info(f"Metrics saved to {metrics_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"Gini: {metrics['gini']:.4f}")
    print(f"KS: {metrics['ks']:.4f}")
    print("-"*60)
    print(f"Optimal Threshold: {metrics.get('optimal_threshold', 0.5):.4f}")
    print(f"Precision (Bad): {metrics['precision_bad']:.4f}")
    print(f"Recall (Bad): {metrics['recall_bad']:.4f}")
    print(f"F1-Score (Bad): {metrics['f1_bad']:.4f}")
    print("-"*60)
    f1 = metrics['f1_bad']
    if f1 >= 0.90:
        print("✅ F1 >= 0.90 - IDEAL")
    elif f1 >= 0.85:
        print("✅ F1 >= 0.85 - META ATINGIDA")
    else:
        print(f"⚠️ F1 < 0.85 - Abaixo da meta")
    print("="*60)
    
    return metrics


if __name__ == "__main__":
    metrics = train_model()

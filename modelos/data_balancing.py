"""
PRINAD - Unified Data Balancing Module
Handles class imbalancing with focus on model performance metrics.
Strategy selection based on predictive performance, not statistical similarity.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import logging
from datetime import datetime
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler

import sys
sys.path.insert(0, str(Path(__file__).parent))
from data_pipeline import load_and_prepare_full_dataset
from feature_engineering import FeatureEngineer

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


class DataBalancer:
    """
    Unified data balancer that selects the best strategy based on
    model performance metrics (AUC, F1, Precision, Recall).
    """
    
    STRATEGIES = {
        'smote': lambda rs, strategy: SMOTE(
            sampling_strategy=strategy, random_state=rs, k_neighbors=5
        ),
        'borderline': lambda rs, strategy: BorderlineSMOTE(
            sampling_strategy=strategy, random_state=rs, k_neighbors=5, kind='borderline-1'
        ),
        'adasyn': lambda rs, strategy: ADASYN(
            sampling_strategy=strategy, random_state=rs, n_neighbors=5
        ),
        'undersample': lambda rs, strategy: RandomUnderSampler(
            sampling_strategy='majority', random_state=rs
        ),
    }
    
    def __init__(self, random_state: int = 42, target_ratio: float = 0.7):
        """
        Args:
            random_state: Random seed for reproducibility
            target_ratio: Target minority/majority ratio after balancing (0.7 = 70%)
        """
        self.random_state = random_state
        self.target_ratio = target_ratio
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.results = {}
        self.best_strategy = None
        self.best_X = None
        self.best_y = None
        self.feature_names = []
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Encode categoricals and prepare features for balancing."""
        df = df.copy()
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        df = df.fillna(0)
        self.feature_names = list(df.columns)
        return df.values.astype(np.float32)
    
    def _evaluate_on_original(self, X_balanced: np.ndarray, y_balanced: np.ndarray,
                               X_orig: np.ndarray, y_orig: np.ndarray,
                               strategy_name: str) -> Dict[str, Any]:
        """Evaluate balanced data quality using model performance on ORIGINAL data."""
        
        # Train on balanced, validate split
        X_train, X_val, y_train, y_val = train_test_split(
            X_balanced, y_balanced, test_size=0.2, 
            random_state=self.random_state, stratify=y_balanced
        )
        
        # Quick model for evaluation
        model = RandomForestClassifier(
            n_estimators=50, max_depth=10, 
            random_state=self.random_state, n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Metrics on validation set (balanced)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        y_val_pred = model.predict(X_val)
        auc_val = roc_auc_score(y_val, y_val_proba)
        f1_val = f1_score(y_val, y_val_pred)
        
        # Metrics on ORIGINAL data (the real test!)
        y_orig_proba = model.predict_proba(X_orig)[:, 1]
        y_orig_pred = model.predict(X_orig)
        
        auc_orig = roc_auc_score(y_orig, y_orig_proba)
        f1_orig = f1_score(y_orig, y_orig_pred)
        precision_orig = precision_score(y_orig, y_orig_pred)
        recall_orig = recall_score(y_orig, y_orig_pred)
        
        # Combined score prioritizing original data performance
        combined_score = (auc_orig * 0.3 + f1_orig * 0.3 + 
                         precision_orig * 0.2 + recall_orig * 0.2)
        
        return {
            'strategy': strategy_name,
            'samples_original': int(len(X_orig)),
            'samples_balanced': int(len(X_balanced)),
            'synthetic_added': int(len(X_balanced) - len(X_orig)),
            'balance_ratio': round(np.sum(y_balanced == 0) / max(np.sum(y_balanced == 1), 1), 2),
            'auc_validation': round(auc_val, 4),
            'f1_validation': round(f1_val, 4),
            'auc_original': round(auc_orig, 4),
            'f1_original': round(f1_orig, 4),
            'precision_original': round(precision_orig, 4),
            'recall_original': round(recall_orig, 4),
            'combined_score': round(combined_score, 4)
        }
    
    def _apply_strategy(self, X: np.ndarray, y: np.ndarray, 
                        strategy_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Apply a specific balancing strategy."""
        
        unique, counts = np.unique(y, return_counts=True)
        majority_count = max(counts)
        minority_class = unique[np.argmin(counts)]
        target_minority = int(majority_count * self.target_ratio)
        
        if strategy_name == 'undersample':
            sampler = RandomUnderSampler(
                sampling_strategy='majority', 
                random_state=self.random_state
            )
        else:
            sampling_strategy = {int(minority_class): target_minority}
            sampler = self.STRATEGIES[strategy_name](self.random_state, sampling_strategy)
        
        return sampler.fit_resample(X, y)
    
    def find_best_strategy(self, X: pd.DataFrame, y: pd.Series,
                           strategies: List[str] = None) -> Dict[str, Any]:
        """
        Compare balancing strategies and select the best based on model performance.
        
        Args:
            X: Features DataFrame
            y: Target Series
            strategies: List of strategies to test (default: all)
            
        Returns:
            Summary with best strategy and all results
        """
        if strategies is None:
            strategies = list(self.STRATEGIES.keys())
        
        logger.info("="*60)
        logger.info("DATA BALANCING - STRATEGY COMPARISON")
        logger.info("="*60)
        
        # Prepare features
        X_prepared = self.prepare_features(X)
        y_array = y.values if hasattr(y, 'values') else np.array(y)
        
        # Scale
        X_scaled = self.scaler.fit_transform(X_prepared)
        
        all_results = []
        best_score = 0
        best_strategy_name = None
        
        for strategy in strategies:
            logger.info(f"\nTesting: {strategy}")
            
            try:
                X_bal, y_bal = self._apply_strategy(X_scaled, y_array, strategy)
                result = self._evaluate_on_original(X_bal, y_bal, X_scaled, y_array, strategy)
                
                self.results[strategy] = {'X': X_bal, 'y': y_bal, 'metrics': result}
                all_results.append(result)
                
                logger.info(f"  Combined score: {result['combined_score']:.4f}")
                logger.info(f"  AUC (orig): {result['auc_original']:.4f}, "
                           f"F1: {result['f1_original']:.4f}, "
                           f"Prec: {result['precision_original']:.4f}, "
                           f"Rec: {result['recall_original']:.4f}")
                
                if result['combined_score'] > best_score:
                    best_score = result['combined_score']
                    best_strategy_name = strategy
                    
            except Exception as e:
                logger.error(f"  Failed: {e}")
                all_results.append({'strategy': strategy, 'error': str(e)})
        
        # Store best
        self.best_strategy = best_strategy_name
        if best_strategy_name:
            self.best_X = self.results[best_strategy_name]['X']
            self.best_y = self.results[best_strategy_name]['y']
        
        # Print summary
        self._print_summary(all_results, best_strategy_name)
        
        return {
            'best_strategy': best_strategy_name,
            'best_score': best_score,
            'all_results': all_results
        }
    
    def _print_summary(self, results: List[Dict], best: str):
        """Print comparison summary."""
        print("\n" + "="*75)
        print("BALANCING STRATEGY COMPARISON - MODEL PERFORMANCE METRICS")
        print("="*75)
        
        print("\n{:<15} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10}".format(
            "Strategy", "Samples", "AUC", "F1", "Prec", "Recall", "Score"
        ))
        print("-"*75)
        
        for r in sorted(results, key=lambda x: x.get('combined_score', 0), reverse=True):
            if 'error' not in r:
                indicator = " ✓" if r['strategy'] == best else ""
                print("{:<15} {:>8} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}{}".format(
                    r['strategy'],
                    r['samples_balanced'],
                    r['auc_original'],
                    r['f1_original'],
                    r['precision_original'],
                    r['recall_original'],
                    r['combined_score'],
                    indicator
                ))
        
        print("\n" + "="*75)
        print(f"🏆 BEST STRATEGY: {best}")
        print("="*75)
    
    def get_balanced_data(self, strategy: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get balanced data (inverse scaled).
        
        Args:
            strategy: Strategy name (uses best if None)
        """
        if strategy is None:
            strategy = self.best_strategy
        
        if strategy not in self.results:
            raise ValueError(f"Strategy {strategy} not evaluated. Run find_best_strategy first.")
        
        X_scaled = self.results[strategy]['X']
        X_unscaled = self.scaler.inverse_transform(X_scaled)
        return X_unscaled, self.results[strategy]['y']
    
    def save_report(self, filepath: Path, summary: Dict):
        """Save comparison report to JSON."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'best_strategy': summary['best_strategy'],
            'best_score': summary['best_score'],
            'target_ratio': self.target_ratio,
            'results': summary['all_results']
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        logger.info(f"Report saved to {filepath}")


def prepare_balanced_data(save_report: bool = True) -> Tuple[np.ndarray, np.ndarray, str, Dict]:
    """
    Main function to prepare balanced data using the best strategy.
    
    Returns:
        Tuple of (X_balanced, y_balanced, best_strategy, summary)
    """
    print("\n" + "="*60)
    print("PRINAD DATA BALANCING")
    print("="*60)
    
    # Load data
    logger.info("Loading data...")
    df_full, X, y = load_and_prepare_full_dataset()
    
    print(f"\nOriginal: {len(y)} samples")
    print(f"Class 0: {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")
    print(f"Class 1: {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")
    print(f"Imbalance ratio: {sum(y == 0) / sum(y == 1):.2f}:1")
    
    # Feature engineering
    logger.info("Applying feature engineering...")
    fe = FeatureEngineer()
    X_engineered = fe.fit_transform(X)
    
    # Find best strategy
    balancer = DataBalancer(random_state=42, target_ratio=0.7)
    summary = balancer.find_best_strategy(X_engineered, y)
    
    # Save report
    if save_report:
        report_path = MODELO_DIR / "balancing_report.json"
        balancer.save_report(report_path, summary)
    
    # Get best data
    X_balanced, y_balanced = balancer.get_balanced_data()
    
    print(f"\n✅ Balanced: {len(y_balanced)} samples")
    print(f"   Class 0: {sum(y_balanced == 0)}, Class 1: {sum(y_balanced == 1)}")
    print(f"   Ratio: {sum(y_balanced == 0) / sum(y_balanced == 1):.2f}:1")
    
    return X_balanced, y_balanced, balancer.best_strategy, summary


if __name__ == "__main__":
    X, y, strategy, summary = prepare_balanced_data()
    print(f"\n🎯 Ready with {strategy} balanced data!")

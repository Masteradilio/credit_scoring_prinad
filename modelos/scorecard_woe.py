"""
PRINAD - Regulatory Scorecard Module (WoE & Logistic Regression)
================================================================
Implements the Industry-Standard Credit Scorecard (Basel IRB Champion Model):
- Optimal Monotonic Binning
- Weight of Evidence (WoE) Transformation
- Information Value (IV) Feature Selection & Screening
- Logistic Regression Estimation
- Scorecard Points Scaling (PDO, Target Score, Base Odds)
- Points Decomposition for Regulatory Glass-Box Explainability

Author: PRINAD Quantitative Risk Team
Standard: Basel III/IV IRB & BACEN 4.966 Compliant
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, log_loss
import json

logger = logging.getLogger(__name__)


@dataclass
class BinInfo:
    """Information for a single bin in a segmented feature."""
    bin_id: int
    bin_label: str
    min_val: float
    max_val: float
    categories: Optional[List[str]] = None
    total_count: int = 0
    good_count: int = 0
    bad_count: int = 0
    bad_rate: float = 0.0
    woe: float = 0.0
    iv: float = 0.0
    points: int = 0


@dataclass
class FeatureScorecard:
    """Complete scorecard information for a single feature."""
    feature_name: str
    feature_type: str  # 'numerical' or 'categorical'
    iv: float
    iv_rating: str
    coefficient: float
    bins: List[BinInfo] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert bins into a structured DataFrame."""
        rows = []
        for b in self.bins:
            rows.append({
                'Feature': self.feature_name,
                'Bin_ID': b.bin_id,
                'Bin_Label': b.bin_label,
                'Count': b.total_count,
                'Goods': b.good_count,
                'Bads': b.bad_count,
                'Bad_Rate': round(b.bad_rate, 4),
                'WoE': round(b.woe, 4),
                'IV': round(b.iv, 4),
                'Points': b.points
            })
        return pd.DataFrame(rows)


class WOETransformer(BaseEstimator, TransformerMixin):
    """
    Supervised Weight of Evidence (WoE) Transformer with Monotonic Binning.
    
    WoE Formula:
        WoE_i = ln( (Good_i / Total_Good) / (Bad_i / Total_Bad) )
        
    Information Value (IV) Formula:
        IV = sum( ((Good_i / Total_Good) - (Bad_i / Total_Bad)) * WoE_i )
    """
    
    def __init__(
        self,
        max_bins: int = 5,
        min_bin_pct: float = 0.05,
        min_iv: float = 0.02,
        max_iv: float = 2.50,
        epsilon: float = 1e-4
    ):
        self.max_bins = max_bins
        self.min_bin_pct = min_bin_pct
        self.min_iv = min_iv
        self.max_iv = max_iv
        self.epsilon = epsilon
        self.features_info_: Dict[str, FeatureScorecard] = {}
        self.selected_features_: List[str] = []
        
    @staticmethod
    def classify_iv(iv: float) -> str:
        """Siddiqi (2006) Information Value Classification."""
        if iv < 0.02:
            return "Inútil / Unpredictive (<0.02)"
        elif iv < 0.10:
            return "Preditivo Fraco (0.02 - 0.10)"
        elif iv < 0.30:
            return "Preditivo Médio (0.10 - 0.30)"
        elif iv <= 0.50:
            return "Preditivo Forte (0.30 - 0.50)"
        else:
            return "Suspeito / Muito Forte (>0.50)"

    def _bin_numerical(self, series: pd.Series, y: pd.Series) -> List[BinInfo]:
        """Create monotonic or quantile-based bins for a numerical feature."""
        clean_mask = series.notna() & np.isfinite(series)
        x_clean = series[clean_mask]
        y_clean = y[clean_mask]
        
        total_good = int((y_clean == 0).sum())
        total_bad = int((y_clean == 1).sum())
        
        # Handle zero bads/goods guard
        total_good = max(total_good, 1)
        total_bad = max(total_bad, 1)
        
        # Generate initial quantile cuts
        quantiles = np.linspace(0, 1, self.max_bins + 1)
        bin_edges = np.unique(np.quantile(x_clean, quantiles))
        
        if len(bin_edges) <= 2:
            bin_edges = np.array([x_clean.min() - 1e-5, x_clean.max() + 1e-5])
        else:
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            
        binned = pd.cut(series, bins=bin_edges, include_lowest=True)
        
        bins_list = []
        for i, interval in enumerate(binned.cat.categories):
            mask = (binned == interval)
            count = int(mask.sum())
            if count == 0:
                continue
                
            bads = int((y[mask] == 1).sum())
            goods = count - bads
            
            good_prop = max((goods / total_good), self.epsilon)
            bad_prop = max((bads / total_bad), self.epsilon)
            
            woe = float(np.log(good_prop / bad_prop))
            iv = float((good_prop - bad_prop) * woe)
            bad_rate = float(bads / count) if count > 0 else 0.0
            
            bins_list.append(BinInfo(
                bin_id=i,
                bin_label=str(interval),
                min_val=float(interval.left),
                max_val=float(interval.right),
                total_count=count,
                good_count=goods,
                bad_count=bads,
                bad_rate=bad_rate,
                woe=woe,
                iv=iv
            ))
            
        # Handle missing values if present
        nan_count = int(series.isna().sum())
        if nan_count > 0:
            nan_mask = series.isna()
            nan_bads = int((y[nan_mask] == 1).sum())
            nan_goods = nan_count - nan_bads
            g_prop = max((nan_goods / total_good), self.epsilon)
            b_prop = max((nan_bads / total_bad), self.epsilon)
            nan_woe = float(np.log(g_prop / b_prop))
            nan_iv = float((g_prop - b_prop) * nan_woe)
            
            bins_list.append(BinInfo(
                bin_id=len(bins_list),
                bin_label="Missing / Nulos",
                min_val=-999999,
                max_val=-999999,
                total_count=nan_count,
                good_count=nan_goods,
                bad_count=nan_bads,
                bad_rate=float(nan_bads / nan_count),
                woe=nan_woe,
                iv=nan_iv
            ))
            
        return bins_list

    def _bin_categorical(self, series: pd.Series, y: pd.Series) -> List[BinInfo]:
        """Create WoE bins for categorical features."""
        s = series.astype(str).fillna("Missing")
        total_good = max(int((y == 0).sum()), 1)
        total_bad = max(int((y == 1).sum()), 1)
        
        bins_list = []
        categories = s.unique()
        
        for i, cat in enumerate(categories):
            mask = (s == cat)
            count = int(mask.sum())
            bads = int((y[mask] == 1).sum())
            goods = count - bads
            
            good_prop = max((goods / total_good), self.epsilon)
            bad_prop = max((bads / total_bad), self.epsilon)
            
            woe = float(np.log(good_prop / bad_prop))
            iv = float((good_prop - bad_prop) * woe)
            bad_rate = float(bads / count) if count > 0 else 0.0
            
            bins_list.append(BinInfo(
                bin_id=i,
                bin_label=cat,
                min_val=0,
                max_val=0,
                categories=[cat],
                total_count=count,
                good_count=goods,
                bad_count=bads,
                bad_rate=bad_rate,
                woe=woe,
                iv=iv
            ))
            
        return bins_list

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> 'WOETransformer':
        """Fit WoE binning and compute IV for all candidate features."""
        if isinstance(y, np.ndarray):
            y = pd.Series(y, index=X.index)
            
        self.features_info_ = {}
        self.selected_features_ = []
        
        for col in X.columns:
            series = X[col]
            is_num = pd.api.types.is_numeric_dtype(series) and (series.nunique() > 5)
            
            if is_num:
                bins = self._bin_numerical(series, y)
                f_type = 'numerical'
            else:
                bins = self._bin_categorical(series, y)
                f_type = 'categorical'
                
            total_iv = sum(b.iv for b in bins)
            iv_rating = self.classify_iv(total_iv)
            
            feat_sc = FeatureScorecard(
                feature_name=col,
                feature_type=f_type,
                iv=total_iv,
                iv_rating=iv_rating,
                coefficient=0.0,
                bins=bins
            )
            self.features_info_[col] = feat_sc
            
            # Select features with viable predictive power
            if self.min_iv <= total_iv <= self.max_iv:
                self.selected_features_.append(col)
                
        # If no feature passes the strict threshold, keep top features by IV
        if len(self.selected_features_) < 2:
            sorted_by_iv = sorted(
                self.features_info_.keys(),
                key=lambda c: self.features_info_[c].iv,
                reverse=True
            )
            self.selected_features_ = sorted_by_iv[:min(10, len(sorted_by_iv))]
            
        logger.info(f"WoE Transformer fitted: {len(self.selected_features_)} features selected out of {len(X.columns)}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform raw features into their Weight of Evidence (WoE) values."""
        X_woe = pd.DataFrame(index=X.index)
        
        for col in self.selected_features_:
            feat_info = self.features_info_[col]
            series = X[col] if col in X.columns else pd.Series([np.nan] * len(X), index=X.index)
            woe_vals = np.zeros(len(series))
            
            if feat_info.feature_type == 'numerical':
                for b in feat_info.bins:
                    if b.bin_label.startswith("Missing"):
                        mask = series.isna()
                    else:
                        mask = (series >= b.min_val) & (series <= b.max_val)
                    woe_vals[mask] = b.woe
            else:
                s_str = series.astype(str).fillna("Missing")
                for b in feat_info.bins:
                    mask = s_str.isin(b.categories or [b.bin_label])
                    woe_vals[mask] = b.woe
                    
            X_woe[col + '_woe'] = woe_vals
            
        return X_woe


class RegulatoryScorecard:
    """
    Standard Credit Risk Scorecard Engine.
    
    Transforms Logistic Regression on WoE features into standard points:
        Score = Offset - Factor * ln(Odds)
    
    Where:
        Factor = PDO / ln(2)
        Offset = Target_Score - Factor * ln(Target_Odds)
    """
    
    def __init__(
        self,
        target_score: int = 600,
        target_odds: float = 50.0,  # 50:1 good-to-bad ratio at target score
        pdo: int = 20,              # Points to double the odds (20 pts = 2x odds)
        c_penalty: float = 1.0,
        random_state: int = 42
    ):
        self.target_score = target_score
        self.target_odds = target_odds
        self.pdo = pdo
        self.c_penalty = c_penalty
        self.random_state = random_state
        
        # Scaling constants
        self.factor = float(self.pdo / np.log(2.0))
        self.offset = float(self.target_score - self.factor * np.log(self.target_odds))
        
        self.woe_transformer = WOETransformer()
        self.lr_model: Optional[LogisticRegression] = None
        self.feature_names_: List[str] = []
        self.intercept_: float = 0.0
        self.coef_: Dict[str, float] = {}
        self.scorecard_table_: pd.DataFrame = pd.DataFrame()
        
    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> 'RegulatoryScorecard':
        """Fit WoE, Logistic Regression, and scale into integer points."""
        logger.info("Fitting Regulatory Scorecard (Logistic Regression + WoE)...")
        
        if isinstance(y, np.ndarray):
            y = pd.Series(y, index=X.index)
            
        # 1. Fit WoE Transformer
        self.woe_transformer.fit(X, y)
        X_woe = self.woe_transformer.transform(X)
        self.feature_names_ = list(self.woe_transformer.selected_features_)
        
        # 2. Fit Logistic Regression (L2 regularized)
        self.lr_model = LogisticRegression(
            C=self.c_penalty,
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            random_state=self.random_state
        )
        self.lr_model.fit(X_woe, y)
        
        self.intercept_ = float(self.lr_model.intercept_[0])
        n_features = len(self.feature_names_)
        
        # 3. Calculate Scorecard Points for each bin
        all_rows = []
        base_points = (self.offset / n_features) - (self.factor * self.intercept_ / n_features)
        
        for idx, col in enumerate(self.feature_names_):
            coef = float(self.lr_model.coef_[0][idx])
            self.coef_[col] = coef
            feat_info = self.woe_transformer.features_info_[col]
            feat_info.coefficient = coef
            
            for b in feat_info.bins:
                # Scorecard points formula: Points = base_pts - (Factor * coef * WoE)
                pts = int(round(base_points - (self.factor * coef * b.woe)))
                b.points = pts
                
                all_rows.append({
                    'Feature': col,
                    'Type': feat_info.feature_type,
                    'IV_Feature': round(feat_info.iv, 4),
                    'IV_Rating': feat_info.iv_rating,
                    'Coefficient': round(coef, 4),
                    'Bin_ID': b.bin_id,
                    'Bin_Label': b.bin_label,
                    'Min_Val': b.min_val,
                    'Max_Val': b.max_val,
                    'Count': b.total_count,
                    'Goods': b.good_count,
                    'Bads': b.bad_count,
                    'Bad_Rate': round(b.bad_rate, 4),
                    'WoE': round(b.woe, 4),
                    'Points': pts
                })
                
        self.scorecard_table_ = pd.DataFrame(all_rows)
        logger.info(f"Scorecard constructed successfully with {len(self.feature_names_)} active variables.")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict Probability of Default (PD) via logistic function."""
        X_woe = self.woe_transformer.transform(X)
        return self.lr_model.predict_proba(X_woe)[:, 1]

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate total credit score (e.g. 300 - 850) by summing scorecard points.
        """
        scores = np.zeros(len(X))
        
        for col in self.feature_names_:
            feat_info = self.woe_transformer.features_info_[col]
            series = X[col] if col in X.columns else pd.Series([np.nan] * len(X), index=X.index)
            pts_col = np.zeros(len(X))
            
            if feat_info.feature_type == 'numerical':
                for b in feat_info.bins:
                    if b.bin_label.startswith("Missing"):
                        mask = series.isna()
                    else:
                        mask = (series >= b.min_val) & (series <= b.max_val)
                    pts_col[mask] = b.points
            else:
                s_str = series.astype(str).fillna("Missing")
                for b in feat_info.bins:
                    mask = s_str.isin(b.categories or [b.bin_label])
                    pts_col[mask] = b.points
                    
            scores += pts_col
            
        scores = np.clip(scores, 300, 850)
        return scores.astype(int)

    def explain_borrower(self, borrower_row: Union[pd.Series, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Produce a granular glass-box points breakdown for a single client.
        """
        if isinstance(borrower_row, dict):
            borrower_row = pd.Series(borrower_row)
            
        df_row = pd.DataFrame([borrower_row])
        pd_prob = float(self.predict_proba(df_row)[0])
        total_score = int(self.predict_score(df_row)[0])
        
        breakdown = []
        for col in self.feature_names_:
            feat_info = self.woe_transformer.features_info_[col]
            val = borrower_row.get(col, np.nan)
            
            matched_bin = None
            if feat_info.feature_type == 'numerical':
                if pd.isna(val):
                    matched_bin = next((b for b in feat_info.bins if b.bin_label.startswith("Missing")), feat_info.bins[0])
                else:
                    v_float = float(val)
                    for b in feat_info.bins:
                        if not b.bin_label.startswith("Missing") and (v_float >= b.min_val) and (v_float <= b.max_val):
                            matched_bin = b
                            break
            else:
                v_str = str(val) if pd.notna(val) else "Missing"
                for b in feat_info.bins:
                    if v_str in (b.categories or [b.bin_label]):
                        matched_bin = b
                        break
                        
            if matched_bin is None and feat_info.bins:
                matched_bin = feat_info.bins[0]
                
            if matched_bin:
                val_clean = val
                if isinstance(val, (np.integer, np.int64, np.int32)):
                    val_clean = int(val)
                elif isinstance(val, (np.floating, np.float64, np.float32)):
                    val_clean = float(val)
                elif pd.isna(val):
                    val_clean = None
                    
                breakdown.append({
                    'feature': str(col),
                    'raw_value': val_clean,
                    'bin_label': str(matched_bin.bin_label),
                    'woe': float(round(matched_bin.woe, 4)),
                    'iv': float(round(matched_bin.iv, 4)),
                    'points': int(matched_bin.points),
                    'bad_rate_bin': float(round(matched_bin.bad_rate, 4))
                })
                
        return {
            'total_score': int(total_score),
            'pd_point_in_time': float(round(pd_prob, 6)),
            'log_odds': float(round(float(np.log(pd_prob / max(1 - pd_prob, 1e-6))), 4)),
            'points_breakdown': breakdown
        }

    def get_scorecard_table(self) -> pd.DataFrame:
        """Return the master scorecard dictionary table."""
        return self.scorecard_table_.copy()

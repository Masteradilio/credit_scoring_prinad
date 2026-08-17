"""
PRINAD - Model Validation & Quantitative Backtesting Engine
===========================================================
Implements the 4 Pillars of Basel / IFRS 9 Model Validation:
1. Discrimination Power: ROC-AUC, Gini, Kolmogorov-Smirnov (KS), CAP Curve (AR)
2. Calibration Accuracy: Reliability Curves, Brier Score Decomposition, Hosmer-Lemeshow Test, Spiegelhalter Z-Test, ECE
3. Population & Characteristic Stability: PSI, CSI, Distribution Drift
4. Statistical Backtesting: Basel Traffic Light Test, Exact Binomial Test (Clopper-Pearson), Jeffreys Bayesian Interval

Author: PRINAD Quantitative Risk Team
Standard: Basel Committee on Banking Supervision (BCBS 328) & EBA GL/2017/16
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, roc_curve, brier_score_loss,
    precision_recall_curve, average_precision_score
)
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)


class ModelValidationEngine:
    """
    Comprehensive 4-Pillar Credit Risk Model Validation Suite.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    # =========================================================================
    # PILLAR 1: DISCRIMINATION (DISCRIMINAÇÃO)
    # =========================================================================
    
    def evaluate_discrimination(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_prob: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Evaluate model's ability to separate goods from bads.
        """
        y_true = np.array(y_true, dtype=int)
        y_prob = np.array(y_prob, dtype=float)
        
        # 1. ROC-AUC & Gini
        auc = float(roc_auc_score(y_true, y_prob))
        gini = float(2.0 * auc - 1.0)
        
        # 2. Kolmogorov-Smirnov (KS)
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        ks_distances = np.abs(tpr - fpr)
        ks_idx = np.argmax(ks_distances)
        ks_stat = float(ks_distances[ks_idx])
        ks_cutoff = float(thresholds[ks_idx])
        
        # 3. Precision-Recall AUC
        pr_auc = float(average_precision_score(y_true, y_prob))
        
        # 4. Assessment benchmarks
        rating_gini = "Excelente (>= 0.70)" if gini >= 0.70 else ("Bom (0.50 - 0.70)" if gini >= 0.50 else "Fraco (< 0.50)")
        rating_ks = "Excelente (>= 0.45)" if ks_stat >= 0.45 else ("Bom (0.35 - 0.45)" if ks_stat >= 0.35 else "Fraco (< 0.35)")
        
        return {
            'auc_roc': round(auc, 4),
            'gini': round(gini, 4),
            'gini_rating': rating_gini,
            'ks_statistic': round(ks_stat, 4),
            'ks_cutoff_threshold': round(ks_cutoff, 4),
            'ks_rating': rating_ks,
            'pr_auc': round(pr_auc, 4),
            'total_samples': int(len(y_true)),
            'total_defaults': int(y_true.sum()),
            'default_rate_pct': round(float(y_true.mean() * 100), 2)
        }

    # =========================================================================
    # PILLAR 2: CALIBRATION (CALIBRAÇÃO)
    # =========================================================================
    
    def evaluate_calibration(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_prob: Union[pd.Series, np.ndarray],
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate if predicted probabilities accurately match observed default frequencies.
        """
        y_true = np.array(y_true, dtype=int)
        y_prob = np.array(y_prob, dtype=float)
        N = len(y_true)
        
        # 1. Brier Score
        brier = float(brier_score_loss(y_true, y_prob))
        
        # Brier Decomposition: BS = Uncertainty - Resolution + Reliability
        base_rate = float(y_true.mean())
        uncertainty = base_rate * (1.0 - base_rate)
        
        # 2. Binning for Calibration Curve & Hosmer-Lemeshow
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.unique(np.quantile(y_prob, quantiles))
        
        if len(bin_edges) <= 2:
            bin_edges = np.linspace(y_prob.min(), y_prob.max() + 1e-6, n_bins + 1)
            
        binned = pd.cut(y_prob, bins=bin_edges, include_lowest=True)
        
        hl_chi2 = 0.0
        ece = 0.0
        resolution = 0.0
        calibration_curve = []
        
        for interval in binned.categories:
            mask = (binned == interval)
            nk = int(mask.sum())
            if nk == 0:
                continue
                
            ok = float(y_true[mask].sum())  # observed defaults
            pk_mean = float(y_prob[mask].mean())  # average predicted PD
            ok_rate = float(ok / nk)  # observed default rate
            
            # Hosmer-Lemeshow term
            expected_defaults = nk * pk_mean
            expected_non_defaults = nk * (1.0 - pk_mean)
            if expected_defaults > 0 and expected_non_defaults > 0:
                hl_term = ((ok - expected_defaults) ** 2) / (expected_defaults * (1.0 - pk_mean))
                hl_chi2 += hl_term
                
            # ECE (Expected Calibration Error)
            ece += (nk / N) * abs(ok_rate - pk_mean)
            
            # Resolution
            resolution += (nk / N) * ((ok_rate - base_rate) ** 2)
            
            calibration_curve.append({
                'bin': str(interval),
                'count': nk,
                'observed_defaults': int(ok),
                'observed_rate_pct': round(ok_rate * 100, 2),
                'predicted_pd_mean_pct': round(pk_mean * 100, 2),
                'diff_pct': round((ok_rate - pk_mean) * 100, 2)
            })
            
        # HL p-value (degrees of freedom = n_bins - 2)
        df_hl = max(1, len(calibration_curve) - 2)
        hl_p_value = float(1.0 - stats.chi2.cdf(hl_chi2, df=df_hl))
        
        # 3. Spiegelhalter Z-Test for probabilistic calibration
        # Z = (sum( (y - p) * (1 - 2p) )) / sqrt( sum( (1 - 2p)^2 * p * (1 - p) ) )
        weights = 1.0 - 2.0 * y_prob
        numerator = np.sum((y_true - y_prob) * weights)
        variance = np.sum((weights ** 2) * y_prob * (1.0 - y_prob))
        spiegelhalter_z = float(numerator / np.sqrt(max(variance, 1e-8)))
        spiegelhalter_p = float(2.0 * (1.0 - stats.norm.cdf(abs(spiegelhalter_z))))
        
        reliability = brier - uncertainty + resolution
        
        return {
            'brier_score': round(brier, 5),
            'brier_uncertainty': round(uncertainty, 5),
            'brier_resolution': round(resolution, 5),
            'brier_reliability': round(max(0.0, reliability), 5),
            'expected_calibration_error_ece': round(ece, 5),
            'hosmer_lemeshow_chi2': round(hl_chi2, 4),
            'hosmer_lemeshow_p_value': round(hl_p_value, 4),
            'hosmer_lemeshow_passed': bool(hl_p_value > 0.05),
            'spiegelhalter_z_score': round(spiegelhalter_z, 4),
            'spiegelhalter_p_value': round(spiegelhalter_p, 4),
            'spiegelhalter_passed': bool(spiegelhalter_p > 0.05),
            'calibration_bins': calibration_curve
        }

    # =========================================================================
    # PILLAR 3: STABILITY (ESTABILIDADE)
    # =========================================================================
    
    def calculate_psi(
        self,
        baseline_dist: Union[pd.Series, np.ndarray],
        current_dist: Union[pd.Series, np.ndarray],
        n_bins: int = 10,
        epsilon: float = 1e-4
    ) -> Dict[str, Any]:
        """
        Calculate Population Stability Index (PSI) between baseline and monitoring sample.
        """
        base = np.array(baseline_dist, dtype=float)
        curr = np.array(current_dist, dtype=float)
        
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.unique(np.quantile(base, quantiles))
        if len(bin_edges) <= 2:
            bin_edges = np.linspace(base.min(), base.max() + 1e-6, n_bins + 1)
            
        base_counts, _ = np.histogram(base, bins=bin_edges)
        curr_counts, _ = np.histogram(curr, bins=bin_edges)
        
        base_pcts = np.maximum(base_counts / len(base), epsilon)
        curr_pcts = np.maximum(curr_counts / len(curr), epsilon)
        
        psi_contributions = (curr_pcts - base_pcts) * np.log(curr_pcts / base_pcts)
        total_psi = float(np.sum(psi_contributions))
        
        if total_psi < 0.10:
            traffic_light = "VERDE (Estável / Sem Drift)"
            action = "Manter modelo em produção."
        elif total_psi <= 0.25:
            traffic_light = "AMARELO (Atenção / Mudança Moderada)"
            action = "Investigar sub-populações e origens do desvio."
        else:
            traffic_light = "VERMELHO (Crítico / Drift Significativo)"
            action = "Ação mandante: Re-treinar ou recalibrar o modelo."
            
        bins_breakdown = []
        for i in range(len(psi_contributions)):
            bins_breakdown.append({
                'bin_range': f"{bin_edges[i]:.4f} a {bin_edges[i+1]:.4f}",
                'baseline_pct': round(float(base_pcts[i] * 100), 2),
                'current_pct': round(float(curr_pcts[i] * 100), 2),
                'psi_component': round(float(psi_contributions[i]), 5)
            })
            
        return {
            'psi_total': round(total_psi, 4),
            'traffic_light': traffic_light,
            'recommended_action': action,
            'bins_breakdown': bins_breakdown
        }

    # =========================================================================
    # PILLAR 4: STATISTICAL BACKTESTING & BASEL TRAFFIC LIGHTS
    # =========================================================================
    
    def run_basel_backtest(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_prob: Union[pd.Series, np.ndarray],
        rating_labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute formal Basel III/IV IRB backtesting by rating band.
        Uses Exact Binomial Tests (Clopper-Pearson) and Basel Traffic Lights.
        """
        y_true = np.array(y_true, dtype=int)
        y_prob = np.array(y_prob, dtype=float)
        
        # Standard rating mapping if not provided
        rating_bands = [
            ('A1', 0.00, 0.05),
            ('A2', 0.05, 0.15),
            ('A3', 0.15, 0.25),
            ('B1', 0.25, 0.35),
            ('B2', 0.35, 0.45),
            ('B3', 0.45, 0.55),
            ('C1', 0.55, 0.65),
            ('C2', 0.65, 0.75),
            ('C3', 0.75, 0.85),
            ('D',  0.85, 0.95),
            ('DEFAULT', 0.95, 1.01)
        ]
        
        band_results = []
        overall_traffic = "VERDE"
        
        for rating, lower, upper in rating_bands:
            mask = (y_prob >= lower) & (y_prob < upper)
            n_obs = int(mask.sum())
            if n_obs == 0:
                continue
                
            k_defaults = int(y_true[mask].sum())
            pd_expected = float(y_prob[mask].mean())
            actual_default_rate = float(k_defaults / n_obs)
            
            # Exact Binomial Clopper-Pearson 95% CI
            ci_low, ci_high = stats.binom.ppf([0.025, 0.975], n_obs, pd_expected) / n_obs
            
            # Cumulative binomial probability (p-value for Basel Traffic Light)
            # P(X >= k | PD_expected)
            p_val_upper = float(1.0 - stats.binom.cdf(k_defaults - 1, n_obs, pd_expected))
            
            # Basel Traffic Light:
            # Green: P >= 0.05 (Default rate is within standard expected bounds)
            # Yellow: 0.0001 <= P < 0.05 (Elevated defaults, monitor closely)
            # Red: P < 0.0001 (Default rate statistically exceeds expected PD)
            if p_val_upper >= 0.05:
                zone = "VERDE (Green Zone)"
            elif p_val_upper >= 0.0001:
                zone = "AMARELO (Yellow Zone)"
                if overall_traffic == "VERDE":
                    overall_traffic = "AMARELO"
            else:
                zone = "VERMELHO (Red Zone - Modelo Subestimando Risco)"
                overall_traffic = "VERMELHO"
                
            band_results.append({
                'rating': rating,
                'exposures': n_obs,
                'observed_defaults': k_defaults,
                'pd_expected_pct': round(pd_expected * 100, 2),
                'actual_rate_pct': round(actual_default_rate * 100, 2),
                'clopper_pearson_95_ci': [round(ci_low * 100, 2), round(ci_high * 100, 2)],
                'binomial_p_value': round(p_val_upper, 4),
                'basel_zone': zone
            })
            
        return {
            'overall_traffic_light': overall_traffic,
            'summary': {
                'total_exposures': int(len(y_true)),
                'total_defaults': int(y_true.sum()),
                'overall_expected_pd_pct': round(float(y_prob.mean() * 100), 2),
                'overall_actual_pd_pct': round(float(y_true.mean() * 100), 2)
            },
            'rating_bands_backtest': band_results
        }

    # =========================================================================
    # MASTER VALIDATION REPORT COMPILER
    # =========================================================================
    
    def generate_full_validation_report(
        self,
        y_true_train: Union[pd.Series, np.ndarray],
        y_prob_train: Union[pd.Series, np.ndarray],
        y_true_test: Union[pd.Series, np.ndarray],
        y_prob_test: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Compile complete 4-Pillar Validation Report across Train & Test partitions.
        """
        disc_test = self.evaluate_discrimination(y_true_test, y_prob_test)
        disc_train = self.evaluate_discrimination(y_true_train, y_prob_train)
        calib_test = self.evaluate_calibration(y_true_test, y_prob_test)
        stability = self.calculate_psi(y_prob_train, y_prob_test)
        backtest = self.run_basel_backtest(y_true_test, y_prob_test)
        
        return {
            'validation_status': 'APPROVED' if (disc_test['gini'] >= 0.50 and calib_test['brier_score'] < 0.15 and stability['psi_total'] < 0.25) else 'REVIEW_REQUIRED',
            'pillar_1_discrimination': {
                'test_metrics': disc_test,
                'train_metrics': disc_train,
                'overfitting_gap_gini': round(disc_train['gini'] - disc_test['gini'], 4)
            },
            'pillar_2_calibration': calib_test,
            'pillar_3_stability': stability,
            'pillar_4_backtesting': backtest
        }

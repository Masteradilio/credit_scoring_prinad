"""
Unit Tests for Regulatory Scorecard (WoE & Points Scaling)
"""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

MODELOS_DIR = Path(__file__).resolve().parent.parent / "modelos"
if str(MODELOS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELOS_DIR))

from scorecard_woe import WOETransformer, RegulatoryScorecard


@pytest.fixture
def sample_credit_data():
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'RENDA_BRUTA': np.random.uniform(2000, 15000, n),
        'IDADE_CLIENTE': np.random.randint(18, 70, n),
        'COMP_RENDA': np.random.uniform(0.05, 0.85, n),
        'TEMPO_RELAC': np.random.randint(1, 100, n),
        'ESCOLARIDADE': np.random.choice(['FUNDAM', 'MEDIO', 'SUPERIOR', 'POS'], n)
    })
    # Target correlated with income and debt
    z = -1.0 + 2.5 * df['COMP_RENDA'] - 0.0002 * df['RENDA_BRUTA']
    prob = 1.0 / (1.0 + np.exp(-z))
    y = (np.random.random(n) < prob).astype(int)
    return df, y


def test_woe_transformer_fit_transform(sample_credit_data):
    X, y = sample_credit_data
    transformer = WOETransformer(max_bins=4)
    transformer.fit(X, y)
    
    assert len(transformer.selected_features_) > 0
    X_woe = transformer.transform(X)
    assert X_woe.shape[0] == X.shape[0]
    assert X_woe.shape[1] == len(transformer.selected_features_)


def test_scorecard_points_and_scaling(sample_credit_data):
    X, y = sample_credit_data
    scorecard = RegulatoryScorecard(target_score=600, target_odds=50.0, pdo=20)
    scorecard.fit(X, y)
    
    # Check predictions
    probs = scorecard.predict_proba(X)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    
    scores = scorecard.predict_score(X)
    assert np.all(scores >= 250) and np.all(scores <= 900)
    
    # Check borrower explainability
    explanation = scorecard.explain_borrower(X.iloc[0])
    assert 'total_score' in explanation
    assert 'points_breakdown' in explanation
    assert len(explanation['points_breakdown']) == len(scorecard.feature_names_)


def test_scorecard_table_generation(sample_credit_data):
    X, y = sample_credit_data
    scorecard = RegulatoryScorecard()
    scorecard.fit(X, y)
    
    table = scorecard.get_scorecard_table()
    assert not table.empty
    assert 'Feature' in table.columns
    assert 'Points' in table.columns
    assert 'WoE' in table.columns
    assert 'IV_Feature' in table.columns

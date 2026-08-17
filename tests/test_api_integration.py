"""
Integration Tests for PRINAD Quantitative API v3.0
"""
import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient

API_DIR = Path(__file__).resolve().parent.parent / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from api import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "version" in data


def test_simple_classify(client):
    payload = {
        "cpf": "12345678900",
        "model_architecture": "scorecard",
        "loan_amount": 10000.0,
        "asset_class": "retail_other"
    }
    res = client.post("/simple_classify", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "credit_score" in data
    assert "pd_12m_pit_pct" in data
    assert "rating" in data
    assert "estagio_ifrs9" in data
    assert data["credit_score"] >= 300


def test_explained_classify(client):
    payload = {
        "cpf": "12345678900",
        "model_architecture": "scorecard",
        "loan_amount": 12000.0
    }
    res = client.post("/explained_classify", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "scorecard_points_breakdown" in data
    assert "macro_scenarios" in data
    assert "lifetime_curve" in data


def test_multiple_classify_json_and_csv(client):
    payload = {
        "cpfs": ["11111111111", "22222222222"],
        "model_architecture": "scorecard",
        "output_format": "json"
    }
    res_json = client.post("/multiple_classify", json=payload)
    assert res_json.status_code == 200
    assert len(res_json.json()["results"]) == 2
    
    payload["output_format"] = "csv"
    res_csv = client.post("/multiple_classify", json=payload)
    assert res_csv.status_code == 200
    assert "text/csv" in res_csv.headers["content-type"]


def test_macro_stress_simulation(client):
    payload = {
        "pd_baseline": 0.05,
        "gdp_growth": -2.5,
        "selic_rate": 15.0,
        "unemployment_rate": 12.0
    }
    res = client.post("/simulate_macro_stress", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["pd_shocked_pit"] > data["pd_baseline_input"]
    assert "ifrs9_weighted_scenarios" in data


def test_ecl_calculation_endpoint(client):
    payload = {
        "ead": 25000.0,
        "pd_12m": 0.04,
        "pd_lifetime": 0.12,
        "days_past_due": 45,
        "lgd": 0.45
    }
    res = client.post("/calculate_ecl", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["stage"] == 2
    assert data["ecl_amount"] > 0


def test_price_loan_endpoint(client):
    payload = {
        "pd_12m": 0.03,
        "lgd": 0.45,
        "target_net_margin": 0.035
    }
    res = client.post("/price_loan", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["fair_lending_rate_annual_pct"] > 10.0
    assert data["raroc_pct"] > 0


def test_models_benchmark_endpoint(client):
    res = client.get("/models/benchmark")
    assert res.status_code == 200
    data = res.json()
    assert "benchmark_summary" in data

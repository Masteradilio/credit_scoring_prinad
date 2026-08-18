"""
Integration Tests for PRINAD Quantitative API v3.1 & Observability Engine
"""
import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient

API_DIR = Path(__file__).resolve().parent.parent / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from api import app

VALID_RATINGS = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3", "D", "DEFAULT"]


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "version" in data
    assert data.get("observability_active") is True


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
    assert 300 <= data["credit_score"] <= 850
    assert 0.0 <= data["pd_12m_pit"] <= 1.0
    assert data["rating"] in VALID_RATINGS
    assert data["estagio_ifrs9"] in [1, 2, 3]


def test_explained_classify(client):
    payload = {
        "cpf": "12345678900",
        "model_architecture": "scorecard",
        "loan_amount": 15000.0,
        "asset_class": "retail_other"
    }
    res = client.post("/explained_classify", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "scorecard_points_breakdown" in data
    assert "lifetime_curve" in data


def test_multiple_classify_json_and_csv(client):
    payload = {
        "cpfs": ["12345678900", "98765432100"],
        "model_architecture": "scorecard",
        "output_format": "json"
    }
    res = client.post("/multiple_classify", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["total_processed"] == 2
    
    # Test CSV output
    payload["output_format"] = "csv"
    res_csv = client.post("/multiple_classify", json=payload)
    assert res_csv.status_code == 200
    assert "text/csv" in res_csv.headers["content-type"]
    assert "cpf" in res_csv.text.lower()


def test_macro_stress_simulation(client):
    payload = {
        "pd_baseline": 0.05,
        "gdp_growth": -2.5,
        "selic_rate": 14.0,
        "unemployment_rate": 12.0,
        "asset_class": "retail_other"
    }
    res = client.post("/simulate_macro_stress", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["simulated_z_factor"] < 0  # Recession implies negative Z
    assert data["pd_shocked_pit"] > 0.05   # Stress increases PD
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


def test_observability_endpoints(client):
    # 1. Trigger a classification to generate telemetry
    client.post("/simple_classify", json={"cpf": "12345678900", "loan_amount": 5000.0, "asset_class": "retail_other"})
    
    # 2. Test Overview
    res_ov = client.get("/observability/overview")
    assert res_ov.status_code == 200
    ov_data = res_ov.json()
    assert "telemetry" in ov_data
    assert "evals" in ov_data
    assert ov_data["telemetry"]["total_requests"] > 0
    
    # 3. Test Evals
    res_ev = client.get("/observability/evals")
    assert res_ev.status_code == 200
    ev_data = res_ev.json()
    assert "evals_status" in ev_data
    assert "population_drift_psi" in ev_data
    assert "traffic_light" in ev_data["population_drift_psi"]
    
    # 4. Test Telemetry
    res_tel = client.get("/observability/telemetry")
    assert res_tel.status_code == 200
    tel_data = res_tel.json()
    assert "latency_p95_ms" in tel_data
    assert "requests_per_sec" in tel_data

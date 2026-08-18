"""
PRINAD - FastAPI Application v3.1 (Quantitative Credit Risk Engine)
===================================================================
Production REST API for Credit Risk Scoring & Quantitative Banking:
- /simple_classify: Fast Single Borrower Classification (Score, PD, Rating, Stage)
- /explained_classify: Detailed Glass-Box Scorecard Points & Feature Attribution
- /multiple_classify: High-Throughput Batch Processing (JSON or CSV output)
- /simulate_macro_stress: Vasicek Macroeconomic Shock Simulation (PIT <-> TTC)
- /calculate_ecl: IFRS 9 / BACEN 4.966 Expected Credit Loss Engine
- /price_loan: Risk-Adjusted Loan Pricing & RAROC Calculator
- /models/benchmark: Champion vs. Challenger Validation Suite Results
- /observability/*: Real-Time Telemetry, PSI Drift, Model Evals & Regulatory Health
- /health: API & Artifacts Health Verification

Author: PRINAD Quantitative Risk Team
Standard: Basel III/IV IRB & IFRS 9 / BACEN 4.966
"""

from fastapi import FastAPI, HTTPException, Query, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from enum import Enum
import pandas as pd
import numpy as np
import json
import io
import csv
import sys
import time
import logging

logger = logging.getLogger(__name__)

# Paths
CURRENT_DIR = Path(__file__).resolve().parent
MODELS_DIR = CURRENT_DIR.parent / "models"
ARTIFACTS_DIR = CURRENT_DIR.parent / "artifacts"
SYNTH_DATA_DIR = CURRENT_DIR.parent / "synth_data"

if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from classifier import PRINADClassifier, ClassificationResult
from vasicek_macro import VasicekMacroEngine
from lifetime_pd import LifetimePDEngine
from decision_pricing_engine import DecisionAndPricingEngine
from data_pipeline import load_client_database, normalize_cpf
from api_monitoring import (
    observability_engine, 
    observability_router
)

# Global lazy instances
_classifier: Optional[PRINADClassifier] = None
_df_clientes: Optional[pd.DataFrame] = None
vasicek_engine = VasicekMacroEngine()
pricing_engine = DecisionAndPricingEngine()


def get_classifier() -> PRINADClassifier:
    global _classifier
    if _classifier is None:
        _classifier = PRINADClassifier(model_type="scorecard")
    return _classifier


def get_clients() -> pd.DataFrame:
    global _df_clientes
    if _df_clientes is None:
        _df_clientes = load_client_database()
    return _df_clientes


app = FastAPI(
    title="PRINAD - Quantitative Credit Risk API v3.1",
    description="Motor de Risco de Crédito, Probabilidade de Inadimplência (PD), Estresse de Vasicek, IFRS 9 e Observabilidade / EVALS",
    version="3.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Observability & Evals Router
app.include_router(observability_router)


# =============================================================================
# PYDANTIC SCHEMAS
# =============================================================================

class OutputFormat(str, Enum):
    json = "json"
    csv = "csv"


class ModelArchitecture(str, Enum):
    scorecard = "scorecard"
    lightgbm = "lightgbm"
    xgboost = "xgboost"
    ensemble = "ensemble"


class BorrowerRequest(BaseModel):
    cpf: str = Field(..., description="CPF do tomador (apenas números)")
    model_architecture: ModelArchitecture = Field(ModelArchitecture.scorecard, description="Modelo de score (Champion ou Challenger)")
    loan_amount: float = Field(10000.0, description="Valor do empréstimo (EAD R$)")
    asset_class: str = Field("retail_other", description="Modalidade (retail_other, retail_revolving, retail_mortgage, corporate)")


class BatchBorrowerRequest(BaseModel):
    cpfs: List[str] = Field(..., description="Lista de CPFs para processamento em lote")
    model_architecture: ModelArchitecture = Field(ModelArchitecture.scorecard, description="Modelo de score")
    output_format: OutputFormat = Field(OutputFormat.json, description="Formato da resposta")


class MacroStressRequest(BaseModel):
    pd_baseline: float = Field(0.05, description="PD inicial baseline / TTC (ex: 0.05 = 5%)")
    gdp_growth: float = Field(2.0, description="Crescimento anual do PIB (%)")
    selic_rate: float = Field(10.5, description="Taxa Básica Selic (% a.a.)")
    unemployment_rate: float = Field(7.8, description="Taxa de Desemprego (%)")
    asset_class: str = Field("retail_other", description="Modalidade de crédito")


class ECLRequest(BaseModel):
    ead: float = Field(..., description="Exposição no Default (EAD R$)")
    pd_12m: float = Field(..., description="PD 12 meses (0 a 1)")
    pd_lifetime: float = Field(..., description="PD Lifetime (0 a 1)")
    days_past_due: int = Field(0, description="Dias de atraso atual")
    lgd: float = Field(0.45, description="Loss Given Default (0 a 1)")


class LoanPricingRequest(BaseModel):
    pd_12m: float = Field(..., description="PD 12 meses (0 a 1)")
    lgd: float = Field(0.45, description="LGD estimada (0 a 1)")
    target_net_margin: float = Field(0.03, description="Margem de lucro líquido desejada (ex: 0.03 = 3%)")
    asset_class: str = Field("retail_other", description="Modalidade")


# =============================================================================
# CORE API ENDPOINTS
# =============================================================================

@app.get("/health", tags=["Sistema"])
async def health_check():
    """Health check do motor de crédito e banco de dados."""
    t0 = time.perf_counter()
    cls_inst = get_classifier()
    df_c = get_clients()
    status = "healthy" if (cls_inst and cls_inst.is_ready()) else "degraded"
    observability_engine.record_request("/health", 200, (time.perf_counter() - t0) * 1000.0)
    return {
        "status": status,
        "model_loaded": cls_inst is not None and cls_inst.is_ready(),
        "database_records": len(df_c) if df_c is not None else 0,
        "observability_active": True,
        "version": "3.1.0",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/simple_classify", tags=["Classificação"])
async def simple_classify(req: BorrowerRequest):
    """
    Classificação rápida de risco de crédito para esteira de concessão.
    Alimenta automaticamente a telemetria e o buffer de drift (PSI) em tempo real.
    """
    t0 = time.perf_counter()
    cls_inst = get_classifier()
    df_c = get_clients()
    
    if not cls_inst:
        observability_engine.record_request("/simple_classify", 503, (time.perf_counter() - t0) * 1000.0)
        raise HTTPException(status_code=503, detail="Modelo não inicializado")
        
    cpf_norm = normalize_cpf(req.cpf)
    client_row = df_c[df_c['CPF_NORM'] == cpf_norm] if df_c is not None else pd.DataFrame()
    
    if client_row.empty:
        borrower_data = {'CPF': req.cpf, 'IDADE_CLIENTE': 35, 'RENDA_BRUTA': 5000.0, 'COMP_RENDA': 0.30}
    else:
        borrower_data = client_row.iloc[0].to_dict()
        
    res = cls_inst.classify_borrower(
        borrower_data=borrower_data,
        model_type=req.model_architecture.value,
        loan_amount=req.loan_amount,
        asset_class=req.asset_class
    )
    
    # Telemetry & Evals Recording
    duration_ms = (time.perf_counter() - t0) * 1000.0
    observability_engine.record_request("/simple_classify", 200, duration_ms)
    observability_engine.record_prediction(
        score=res.prinad_score,
        pd_pit=res.pd_12m_pit,
        rating=res.rating,
        stage=res.estagio_pe,
        model_arch=req.model_architecture.value
    )
    
    return {
        "cpf": res.cpf,
        "credit_score": res.prinad_score,
        "pd_12m_pit": res.pd_12m_pit,
        "pd_12m_pit_pct": res.pd_12m_pit_pct,
        "rating": res.rating,
        "rating_descricao": res.rating_descricao,
        "estagio_ifrs9": res.estagio_pe,
        "estagio_descricao": res.estagio_descricao,
        "ecl_provision_amount": res.ecl_provision_amount,
        "fair_interest_rate_annual": res.fair_interest_rate_pct,
        "acao_sugerida": res.acao_sugerida,
        "model_used": res.model_architecture,
        "timestamp": res.timestamp
    }


@app.post("/explained_classify", tags=["Classificação"])
async def explained_classify(req: BorrowerRequest):
    """
    Classificação completa com explicabilidade Glass-Box (Scorecard Points) e cenários IFRS 9.
    """
    t0 = time.perf_counter()
    cls_inst = get_classifier()
    df_c = get_clients()
    
    if not cls_inst:
        observability_engine.record_request("/explained_classify", 503, (time.perf_counter() - t0) * 1000.0)
        raise HTTPException(status_code=503, detail="Modelo não inicializado")
        
    cpf_norm = normalize_cpf(req.cpf)
    client_row = df_c[df_c['CPF_NORM'] == cpf_norm] if df_c is not None else pd.DataFrame()
    
    if client_row.empty:
        borrower_data = {'CPF': req.cpf, 'IDADE_CLIENTE': 35, 'RENDA_BRUTA': 5000.0, 'COMP_RENDA': 0.30}
    else:
        borrower_data = client_row.iloc[0].to_dict()
        
    res = cls_inst.classify_borrower(
        borrower_data=borrower_data,
        model_type=req.model_architecture.value,
        loan_amount=req.loan_amount,
        asset_class=req.asset_class
    )
    
    # Telemetry & Evals Recording
    duration_ms = (time.perf_counter() - t0) * 1000.0
    observability_engine.record_request("/explained_classify", 200, duration_ms)
    observability_engine.record_prediction(
        score=res.prinad_score,
        pd_pit=res.pd_12m_pit,
        rating=res.rating,
        stage=res.estagio_pe,
        model_arch=req.model_architecture.value
    )
    
    return res.to_dict()


@app.post("/multiple_classify", tags=["Processamento em Lote"])
async def multiple_classify(req: BatchBorrowerRequest):
    """
    Processamento de múltiplos CPFs em lote (saída JSON ou CSV).
    """
    t0 = time.perf_counter()
    cls_inst = get_classifier()
    df_c = get_clients()
    
    if not cls_inst:
        observability_engine.record_request("/multiple_classify", 503, (time.perf_counter() - t0) * 1000.0)
        raise HTTPException(status_code=503, detail="Modelo não inicializado")
        
    results = []
    for cpf in req.cpfs:
        cpf_norm = normalize_cpf(cpf)
        client_row = df_c[df_c['CPF_NORM'] == cpf_norm] if df_c is not None else pd.DataFrame()
        b_data = client_row.iloc[0].to_dict() if not client_row.empty else {'CPF': cpf}
        
        res = cls_inst.classify_borrower(borrower_data=b_data, model_type=req.model_architecture.value)
        
        # Telemetry & Evals Recording
        observability_engine.record_prediction(
            score=res.prinad_score,
            pd_pit=res.pd_12m_pit,
            rating=res.rating,
            stage=res.estagio_pe,
            model_arch=req.model_architecture.value
        )
        
        results.append({
            'cpf': res.cpf,
            'credit_score': res.prinad_score,
            'pd_12m_pct': res.pd_12m_pit_pct,
            'rating': res.rating,
            'estagio_ifrs9': res.estagio_pe,
            'ecl_provision': res.ecl_provision_amount,
            'fair_rate_pct': res.fair_interest_rate_pct,
            'action': res.acao_sugerida
        })
        
    duration_ms = (time.perf_counter() - t0) * 1000.0
    observability_engine.record_request("/multiple_classify", 200, duration_ms)
    
    if req.output_format == OutputFormat.csv:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        return Response(content=output.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=classificacoes_prinad.csv"})
        
    return {
        "total_requested": len(req.cpfs),
        "total_processed": len(results),
        "results": results
    }


@app.post("/simulate_macro_stress", tags=["Modelagem Macroeconômica"])
async def simulate_macro_stress(req: MacroStressRequest):
    """
    Simula choques macroeconômicos via Equação de Vasicek ASRF.
    """
    t0 = time.perf_counter()
    z_factor = vasicek_engine.calculate_z_factor(req.gdp_growth, req.selic_rate, req.unemployment_rate)
    pd_shocked = vasicek_engine.ttc_to_pit(req.pd_baseline, z_factor, asset_class=req.asset_class)
    ifrs9_multi = vasicek_engine.evaluate_ifrs9_scenarios(req.pd_baseline)
    
    # Record macro stress telemetry
    duration_ms = (time.perf_counter() - t0) * 1000.0
    observability_engine.record_request("/simulate_macro_stress", 200, duration_ms)
    observability_engine.record_prediction(
        score=int(np.clip(850 - pd_shocked * 700, 300, 850)),
        pd_pit=pd_shocked,
        rating="MACRO_SIM",
        stage=1 if pd_shocked < 0.05 else (2 if pd_shocked < 0.15 else 3),
        z_factor=z_factor
    )
    
    return {
        "pd_baseline_input": req.pd_baseline,
        "simulated_z_factor": round(z_factor, 4),
        "pd_shocked_pit": round(pd_shocked, 6),
        "pd_shocked_pit_pct": round(pd_shocked * 100, 3),
        "delta_vs_baseline_pct": round((pd_shocked - req.pd_baseline) * 100, 3),
        "ifrs9_weighted_scenarios": ifrs9_multi
    }


@app.post("/calculate_ecl", tags=["IFRS 9 & Provisão"])
async def calculate_ecl_endpoint(req: ECLRequest):
    """
    Cálculo de Perda Esperada (ECL) com estadiamento IFRS 9 / BACEN 4.966.
    """
    t0 = time.perf_counter()
    ecl_res = pricing_engine.calculate_ecl(
        ead=req.ead,
        pd_12m=req.pd_12m,
        pd_lifetime=req.pd_lifetime,
        days_past_due=req.days_past_due,
        lgd=req.lgd
    )
    observability_engine.record_request("/calculate_ecl", 200, (time.perf_counter() - t0) * 1000.0)
    return {
        "stage": ecl_res.stage,
        "stage_name": ecl_res.stage_name,
        "ead": ecl_res.ead,
        "pd_applied": ecl_res.pd_applied,
        "horizon": ecl_res.horizon,
        "ecl_amount": ecl_res.ecl_amount,
        "ecl_percentage": ecl_res.ecl_percentage,
        "trigger": ecl_res.sicr_trigger
    }


@app.post("/price_loan", tags=["Precificação & RAROC"])
async def price_loan_endpoint(req: LoanPricingRequest):
    """
    Cálculo de Taxa Justa de Empréstimo (Risk-Based Pricing) e RAROC.
    """
    t0 = time.perf_counter()
    pricing_res = pricing_engine.price_credit(
        pd_12m=req.pd_12m,
        lgd=req.lgd,
        target_margin=req.target_net_margin,
        asset_class=req.asset_class
    )
    observability_engine.record_request("/price_loan", 200, (time.perf_counter() - t0) * 1000.0)
    return {
        "cost_of_funds_pct": pricing_res.cost_of_funds_pct,
        "opex_cost_pct": pricing_res.opex_cost_pct,
        "expected_loss_pct": pricing_res.expected_loss_pct,
        "economic_capital_charge_pct": pricing_res.capital_charge_pct,
        "target_margin_pct": pricing_res.target_net_margin_pct,
        "fair_lending_rate_annual_pct": pricing_res.fair_lending_rate_annual,
        "raroc_pct": pricing_res.raroc_percentage
    }


@app.get("/models/benchmark", tags=["Validação & Benchmark"])
async def get_models_benchmark():
    """
    Retorna o relatório completo de validação comparativa Champion vs. Challengers nos 4 pilares.
    """
    t0 = time.perf_counter()
    report_path = ARTIFACTS_DIR / "model_comparison_report.json"
    if not report_path.exists():
        observability_engine.record_request("/models/benchmark", 404, (time.perf_counter() - t0) * 1000.0)
        raise HTTPException(status_code=404, detail="Relatório de benchmark não encontrado.")
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    observability_engine.record_request("/models/benchmark", 200, (time.perf_counter() - t0) * 1000.0)
    return data


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

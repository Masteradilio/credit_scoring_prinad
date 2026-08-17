"""
PRINAD - FastAPI Application v3.0 (Quantitative Credit Risk Engine)
===================================================================
Production REST API for Credit Risk Scoring & Quantitative Banking:
- /simple_classify: Fast Single Borrower Classification (Score, PD, Rating, Stage)
- /explained_classify: Detailed Glass-Box Scorecard Points & Feature Attribution
- /multiple_classify: High-Throughput Batch Processing (JSON or CSV output)
- /simulate_macro_stress: Vasicek Macroeconomic Shock Simulation (PIT <-> TTC)
- /calculate_ecl: IFRS 9 / BACEN 4.966 Expected Credit Loss Engine
- /price_loan: Risk-Adjusted Loan Pricing & RAROC Calculator
- /models/benchmark: Champion vs. Challenger Validation Suite Results
- /health: API & Artifacts Health Verification

Author: PRINAD Quantitative Risk Team
Standard: Basel III/IV IRB & IFRS 9 / BACEN 4.966
"""

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from enum import Enum
import pandas as pd
import json
import io
import csv
import sys

# Paths
CURRENT_DIR = Path(__file__).resolve().parent
MODELOS_DIR = CURRENT_DIR.parent / "modelos"
ARTEFATOS_DIR = CURRENT_DIR.parent / "artefatos"
DADOS_DIR = CURRENT_DIR.parent / "dados"

if str(MODELOS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELOS_DIR))

from classifier import PRINADClassifier, ClassificationResult
from vasicek_macro import VasicekMacroEngine
from lifetime_pd import LifetimePDEngine
from decision_pricing_engine import DecisionAndPricingEngine
from data_pipeline import load_client_database, normalize_cpf

app = FastAPI(
    title="PRINAD - Quantitative Credit Risk API v3.0",
    description="Motor de Risco de Crédito, Probabilidade de Inadimplência (PD), Estresse de Vasicek e IFRS 9 / BACEN 4.966",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Instances
classifier: Optional[PRINADClassifier] = None
df_clientes: Optional[pd.DataFrame] = None
vasicek_engine = VasicekMacroEngine()
pricing_engine = DecisionAndPricingEngine()


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
# STARTUP EVENT
# =============================================================================

@app.on_event("startup")
async def startup_event():
    global classifier, df_clientes
    try:
        df_clientes = load_client_database()
        classifier = PRINADClassifier(model_type="scorecard")
    except Exception as e:
        print(f"Error during API startup: {e}")


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health", tags=["Sistema"])
async def health_check():
    """Health check do motor de crédito."""
    return {
        "status": "healthy" if (classifier and classifier.is_ready()) else "degraded",
        "model_loaded": classifier is not None and classifier.is_ready(),
        "database_records": len(df_clientes) if df_clientes is not None else 0,
        "version": "3.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/simple_classify", tags=["Classificação"])
async def simple_classify(req: BorrowerRequest):
    """
    Classificação rápida de risco de crédito para esteira de concessão.
    """
    if not classifier:
        raise HTTPException(status_code=503, detail="Modelo não inicializado")
        
    cpf_norm = normalize_cpf(req.cpf)
    client_row = df_clientes[df_clientes['CPF_NORM'] == cpf_norm] if df_clientes is not None else pd.DataFrame()
    
    if client_row.empty:
        # Generate clean defaults if CPF not in local database
        borrower_data = {'CPF': req.cpf, 'IDADE_CLIENTE': 35, 'RENDA_BRUTA': 5000.0, 'COMP_RENDA': 0.30}
    else:
        borrower_data = client_row.iloc[0].to_dict()
        
    res = classifier.classify_borrower(
        borrower_data=borrower_data,
        model_type=req.model_architecture.value,
        loan_amount=req.loan_amount,
        asset_class=req.asset_class
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
    if not classifier:
        raise HTTPException(status_code=503, detail="Modelo não inicializado")
        
    cpf_norm = normalize_cpf(req.cpf)
    client_row = df_clientes[df_clientes['CPF_NORM'] == cpf_norm] if df_clientes is not None else pd.DataFrame()
    
    if client_row.empty:
        borrower_data = {'CPF': req.cpf, 'IDADE_CLIENTE': 35, 'RENDA_BRUTA': 5000.0, 'COMP_RENDA': 0.30}
    else:
        borrower_data = client_row.iloc[0].to_dict()
        
    res = classifier.classify_borrower(
        borrower_data=borrower_data,
        model_type=req.model_architecture.value,
        loan_amount=req.loan_amount,
        asset_class=req.asset_class
    )
    return res.to_dict()


@app.post("/multiple_classify", tags=["Processamento em Lote"])
async def multiple_classify(req: BatchBorrowerRequest):
    """
    Processamento de múltiplos CPFs em lote (saída JSON ou CSV).
    """
    if not classifier:
        raise HTTPException(status_code=503, detail="Modelo não inicializado")
        
    results = []
    for cpf in req.cpfs:
        cpf_norm = normalize_cpf(cpf)
        client_row = df_clientes[df_clientes['CPF_NORM'] == cpf_norm] if df_clientes is not None else pd.DataFrame()
        b_data = client_row.iloc[0].to_dict() if not client_row.empty else {'CPF': cpf}
        
        res = classifier.classify_borrower(borrower_data=b_data, model_type=req.model_architecture.value)
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
    z_factor = vasicek_engine.calculate_z_factor(req.gdp_growth, req.selic_rate, req.unemployment_rate)
    pd_shocked = vasicek_engine.ttc_to_pit(req.pd_baseline, z_factor, asset_class=req.asset_class)
    
    ifrs9_multi = vasicek_engine.evaluate_ifrs9_scenarios(req.pd_baseline)
    
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
    ecl_res = pricing_engine.calculate_ecl(
        ead=req.ead,
        pd_12m=req.pd_12m,
        pd_lifetime=req.pd_lifetime,
        days_past_due=req.days_past_due,
        lgd=req.lgd
    )
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
    pricing_res = pricing_engine.price_credit(
        pd_12m=req.pd_12m,
        lgd=req.lgd,
        target_margin=req.target_net_margin,
        asset_class=req.asset_class
    )
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
    report_path = ARTEFATOS_DIR / "model_comparison_report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Relatório de benchmark não encontrado.")
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# -*- coding: utf-8 -*-
"""
API de Analytics e Monitoramento
================================

Endpoints FastAPI para monitoramento de performance do modelo PRINAD.

Autor: Sistema ECL
Data: Janeiro 2026
"""

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import logging

# Importações locais
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from model_monitoring import (
        ModelMonitor, 
        get_model_monitor,
        NivelAlerta
    )
except ImportError:
    # Fallback se importação falhar
    ModelMonitor = None
    get_model_monitor = lambda: None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["Analytics"])


# ============================================================================
# MODELOS DE RESPOSTA
# ============================================================================

class MetricasResponse(BaseModel):
    """Resposta de métricas do modelo."""
    metricas: dict
    status_geral: str
    alertas: list
    ultima_atualizacao: str


class DriftResponse(BaseModel):
    """Resposta de análise de drift."""
    psi_total: float
    nivel_alerta: str
    interpretacao: str
    detalhes_por_faixa: list


class BacktestingResponse(BaseModel):
    """Resposta de backtesting."""
    data_execucao: str
    periodo_analise: str
    resumo: dict
    por_faixa: list


class RelatorioCompletoResponse(BaseModel):
    """Resposta de relatório completo."""
    data_geracao: str
    status_geral: dict
    metricas_atuais: dict
    analise_drift: dict
    backtesting: dict
    evolucao_auc: list
    evolucao_gini: list
    recomendacoes: list


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/model-performance", response_model=MetricasResponse)
async def get_model_performance():
    """
    Retorna métricas atuais de performance do modelo PRINAD.
    
    Métricas:
    - AUC-ROC
    - Gini
    - Precision
    - Recall
    - F1-Score
    - KS Statistic
    """
    monitor = get_model_monitor()
    if monitor is None:
        raise HTTPException(status_code=500, detail="Monitor não disponível")
    
    resultado = monitor.obter_metricas_atuais()
    
    if "erro" in resultado:
        raise HTTPException(status_code=404, detail=resultado["erro"])
    
    return MetricasResponse(
        metricas=resultado["metricas"],
        status_geral=resultado["status_geral"],
        alertas=resultado["alertas"],
        ultima_atualizacao=resultado["ultima_atualizacao"]
    )


@router.get("/drift-report", response_model=DriftResponse)
async def get_drift_report():
    """
    Retorna relatório de drift do modelo (PSI - Population Stability Index).
    
    Níveis de alerta:
    - Verde (PSI < 0.10): Sem mudança significativa
    - Amarelo (PSI 0.10-0.25): Mudança moderada, investigar
    - Vermelho (PSI > 0.25): Mudança significativa, ação requerida
    """
    monitor = get_model_monitor()
    if monitor is None:
        raise HTTPException(status_code=500, detail="Monitor não disponível")
    
    psi = monitor.calcular_psi()
    
    return DriftResponse(
        psi_total=psi.psi_total,
        nivel_alerta=psi.nivel_alerta.value,
        interpretacao=psi.interpretacao,
        detalhes_por_faixa=psi.detalhes_por_faixa
    )


@router.get("/accuracy-trend")
async def get_accuracy_trend(
    metrica: str = Query("auc_roc", description="Métrica a visualizar"),
    meses: int = Query(6, ge=1, le=24, description="Quantidade de meses")
):
    """
    Retorna evolução temporal de uma métrica de acurácia.
    
    Métricas disponíveis:
    - auc_roc
    - gini
    - precision
    - recall
    - f1_score
    - ks_statistic
    """
    monitor = get_model_monitor()
    if monitor is None:
        raise HTTPException(status_code=500, detail="Monitor não disponível")
    
    metricas_validas = ["auc_roc", "gini", "precision", "recall", "f1_score", "ks_statistic"]
    if metrica not in metricas_validas:
        raise HTTPException(
            status_code=400, 
            detail=f"Métrica inválida. Opções: {metricas_validas}"
        )
    
    dados = monitor.obter_evolucao_temporal(metrica, meses)
    
    return {
        "metrica": metrica,
        "periodo_meses": meses,
        "dados": dados
    }


@router.post("/backtest", response_model=BacktestingResponse)
async def run_backtest():
    """
    Executa backtesting do modelo.
    
    Compara PD esperado vs taxa de default realizada por faixa.
    Retorna análise de calibração do modelo.
    """
    monitor = get_model_monitor()
    if monitor is None:
        raise HTTPException(status_code=500, detail="Monitor não disponível")
    
    resultado = monitor.executar_backtesting()
    
    return BacktestingResponse(
        data_execucao=resultado["data_execucao"],
        periodo_analise=resultado["periodo_analise"],
        resumo=resultado["resumo"],
        por_faixa=resultado["por_faixa"]
    )


@router.get("/full-report", response_model=RelatorioCompletoResponse)
async def get_full_report():
    """
    Gera relatório completo de monitoramento do modelo.
    
    Inclui:
    - Métricas atuais
    - Análise de drift (PSI)
    - Backtesting
    - Evolução temporal
    - Recomendações
    """
    monitor = get_model_monitor()
    if monitor is None:
        raise HTTPException(status_code=500, detail="Monitor não disponível")
    
    relatorio = monitor.gerar_relatorio_completo()
    
    return RelatorioCompletoResponse(**relatorio)


@router.get("/health")
async def analytics_health():
    """Health check do módulo de analytics."""
    monitor = get_model_monitor()
    
    return {
        "status": "ok" if monitor else "degraded",
        "modulo": "analytics",
        "timestamp": datetime.now().isoformat()
    }


# Export do router
__all__ = ["router"]

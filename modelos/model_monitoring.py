# -*- coding: utf-8 -*-
"""
Módulo de Monitoramento de Modelos
==================================

Dashboard de performance do modelo PRINAD:
- Monitoramento de Drift (PSI)
- Acurácia temporal
- Backtesting

Autor: Sistema ECL
Data: Janeiro 2026
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class NivelAlerta(str, Enum):
    """Níveis de alerta do monitoramento."""
    VERDE = "verde"       # Normal
    AMARELO = "amarelo"   # Atenção
    VERMELHO = "vermelho" # Crítico


@dataclass
class MetricasModelo:
    """Métricas de performance do modelo."""
    auc_roc: float
    gini: float
    precision: float
    recall: float
    f1_score: float
    ks_statistic: float
    data_calculo: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "auc_roc": round(self.auc_roc, 4),
            "gini": round(self.gini, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "ks_statistic": round(self.ks_statistic, 4),
            "data_calculo": self.data_calculo.isoformat()
        }


@dataclass
class ResultadoPSI:
    """Resultado do cálculo de PSI (Population Stability Index)."""
    psi_total: float
    nivel_alerta: NivelAlerta
    detalhes_por_faixa: List[Dict]
    interpretacao: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "psi_total": round(self.psi_total, 4),
            "nivel_alerta": self.nivel_alerta.value,
            "detalhes_por_faixa": self.detalhes_por_faixa,
            "interpretacao": self.interpretacao
        }


class ModelMonitor:
    """
    Monitor de performance do modelo PRINAD.
    
    Features:
    - Cálculo de PSI para detectar drift
    - Monitoramento de métricas ao longo do tempo
    - Alertas automáticos de degradação
    - Backtesting periódico
    """
    
    # Thresholds de alertas
    PSI_THRESHOLD_AMARELO = 0.10
    PSI_THRESHOLD_VERMELHO = 0.25
    AUC_THRESHOLD_AMARELO = 0.85
    AUC_THRESHOLD_VERMELHO = 0.75
    
    def __init__(self):
        """Inicializa o monitor."""
        self._historico_metricas: List[MetricasModelo] = []
        self._distribuicao_baseline: Optional[np.ndarray] = None
        self._faixas_baseline: Optional[np.ndarray] = None
        
        # Carregar métricas históricas (mock)
        self._carregar_metricas_mock()
    
    def _carregar_metricas_mock(self):
        """Carrega métricas históricas mockadas para demonstração."""
        base_date = datetime.now() - timedelta(days=180)
        
        # Simular evolução das métricas ao longo de 6 meses
        for i in range(6):
            data = base_date + timedelta(days=i * 30)
            
            # Ligeira degradação ao longo do tempo (realista)
            degradacao = i * 0.005
            
            metricas = MetricasModelo(
                auc_roc=0.9986 - degradacao,
                gini=0.9972 - degradacao,
                precision=0.9535 - degradacao * 0.5,
                recall=0.9713 - degradacao * 0.3,
                f1_score=0.9623 - degradacao * 0.4,
                ks_statistic=0.8542 - degradacao,
                data_calculo=data
            )
            self._historico_metricas.append(metricas)
        
        # Definir baseline de distribuição (primeira observação)
        np.random.seed(42)
        self._distribuicao_baseline = np.random.beta(2, 5, 10000)  # Distribuição típica de PD
        self._faixas_baseline = np.percentile(self._distribuicao_baseline, np.arange(0, 101, 10))
    
    def calcular_psi(
        self,
        distribuicao_atual: np.ndarray = None,
        n_faixas: int = 10
    ) -> ResultadoPSI:
        """
        Calcula o PSI (Population Stability Index) para detectar drift.
        
        PSI < 0.10: Sem mudança significativa
        PSI 0.10-0.25: Mudança moderada (investigar)
        PSI > 0.25: Mudança significativa (ação requerida)
        
        Args:
            distribuicao_atual: Array com distribuição atual de scores
            n_faixas: Número de faixas para cálculo
            
        Returns:
            ResultadoPSI com detalhes
        """
        if distribuicao_atual is None:
            # Mock: gerar distribuição com leve drift
            np.random.seed(int(datetime.now().timestamp()) % 1000)
            distribuicao_atual = np.random.beta(2.2, 4.8, 10000)  # Leve mudança
        
        if self._distribuicao_baseline is None:
            return ResultadoPSI(
                psi_total=0.0,
                nivel_alerta=NivelAlerta.VERDE,
                detalhes_por_faixa=[],
                interpretacao="Baseline não definido"
            )
        
        # Calcular distribuição por faixas
        faixas = np.percentile(self._distribuicao_baseline, np.linspace(0, 100, n_faixas + 1))
        
        base_counts, _ = np.histogram(self._distribuicao_baseline, bins=faixas)
        atual_counts, _ = np.histogram(distribuicao_atual, bins=faixas)
        
        # Normalizar para proporções
        base_props = base_counts / base_counts.sum()
        atual_props = atual_counts / atual_counts.sum()
        
        # Evitar divisão por zero
        base_props = np.clip(base_props, 1e-10, 1)
        atual_props = np.clip(atual_props, 1e-10, 1)
        
        # Calcular PSI por faixa
        psi_por_faixa = (atual_props - base_props) * np.log(atual_props / base_props)
        psi_total = np.sum(psi_por_faixa)
        
        # Detalhes por faixa
        detalhes = []
        for i in range(len(psi_por_faixa)):
            detalhes.append({
                "faixa": f"{faixas[i]:.2%} - {faixas[i+1]:.2%}",
                "baseline_pct": round(base_props[i] * 100, 2),
                "atual_pct": round(atual_props[i] * 100, 2),
                "psi_contribuicao": round(psi_por_faixa[i], 4)
            })
        
        # Determinar nível de alerta
        if psi_total >= self.PSI_THRESHOLD_VERMELHO:
            nivel = NivelAlerta.VERMELHO
            interpretacao = "CRÍTICO: Drift significativo detectado. Ação imediata requerida."
        elif psi_total >= self.PSI_THRESHOLD_AMARELO:
            nivel = NivelAlerta.AMARELO
            interpretacao = "ATENÇÃO: Mudança moderada detectada. Investigar causas."
        else:
            nivel = NivelAlerta.VERDE
            interpretacao = "NORMAL: Distribuição estável, sem drift significativo."
        
        logger.info(f"PSI calculado: {psi_total:.4f} ({nivel.value})")
        
        return ResultadoPSI(
            psi_total=psi_total,
            nivel_alerta=nivel,
            detalhes_por_faixa=detalhes,
            interpretacao=interpretacao
        )
    
    def obter_metricas_atuais(self) -> Dict[str, Any]:
        """
        Retorna métricas atuais do modelo.
        
        Returns:
            Dict com métricas e alertas
        """
        if not self._historico_metricas:
            return {"erro": "Nenhuma métrica disponível"}
        
        metricas = self._historico_metricas[-1]
        
        # Verificar alertas
        alertas = []
        if metricas.auc_roc < self.AUC_THRESHOLD_VERMELHO:
            alertas.append({
                "nivel": NivelAlerta.VERMELHO.value,
                "metrica": "AUC-ROC",
                "valor": metricas.auc_roc,
                "threshold": self.AUC_THRESHOLD_VERMELHO,
                "mensagem": "AUC-ROC crítico! Retreinar modelo."
            })
        elif metricas.auc_roc < self.AUC_THRESHOLD_AMARELO:
            alertas.append({
                "nivel": NivelAlerta.AMARELO.value,
                "metrica": "AUC-ROC",
                "valor": metricas.auc_roc,
                "threshold": self.AUC_THRESHOLD_AMARELO,
                "mensagem": "AUC-ROC em atenção. Monitorar."
            })
        
        return {
            "metricas": metricas.to_dict(),
            "status_geral": NivelAlerta.VERDE.value if not alertas else alertas[0]["nivel"],
            "alertas": alertas,
            "ultima_atualizacao": metricas.data_calculo.isoformat()
        }
    
    def obter_evolucao_temporal(
        self,
        metrica: str = "auc_roc",
        ultimos_meses: int = 6
    ) -> List[Dict]:
        """
        Retorna evolução temporal de uma métrica.
        
        Args:
            metrica: Nome da métrica (auc_roc, gini, precision, recall, etc)
            ultimos_meses: Quantidade de meses a retornar
            
        Returns:
            Lista com evolução temporal
        """
        dados = []
        for m in self._historico_metricas[-ultimos_meses:]:
            valor = getattr(m, metrica, None)
            if valor is not None:
                dados.append({
                    "data": m.data_calculo.strftime("%Y-%m"),
                    "valor": round(valor, 4)
                })
        return dados
    
    def executar_backtesting(
        self,
        pd_esperado: List[float] = None,
        default_realizado: List[int] = None
    ) -> Dict[str, Any]:
        """
        Executa backtesting comparando PD esperado vs realizado.
        
        Args:
            pd_esperado: Lista de PDs previstos
            default_realizado: Lista de defaults realizados (0 ou 1)
            
        Returns:
            Dict com resultados do backtesting
        """
        # Mock de dados se não fornecidos
        if pd_esperado is None or default_realizado is None:
            np.random.seed(123)
            n = 1000
            pd_esperado = np.random.beta(2, 5, n)  # PDs previstos
            # Simular defaults com correlação com PD
            prob_default = pd_esperado * 1.1  # Ligeira subestimação
            default_realizado = (np.random.random(n) < prob_default).astype(int)
        
        pd_esperado = np.array(pd_esperado)
        default_realizado = np.array(default_realizado)
        
        # Dividir em faixas de PD
        faixas = [0, 0.05, 0.10, 0.20, 0.50, 1.0]
        resultados_por_faixa = []
        
        for i in range(len(faixas) - 1):
            mask = (pd_esperado >= faixas[i]) & (pd_esperado < faixas[i + 1])
            if mask.sum() > 0:
                pd_medio = pd_esperado[mask].mean()
                default_rate = default_realizado[mask].mean()
                
                # Razão esperado/realizado
                ratio = default_rate / pd_medio if pd_medio > 0 else 1.0
                
                resultados_por_faixa.append({
                    "faixa": f"{faixas[i]:.0%} - {faixas[i+1]:.0%}",
                    "quantidade": int(mask.sum()),
                    "pd_medio_esperado": round(pd_medio * 100, 2),
                    "default_rate_realizado": round(default_rate * 100, 2),
                    "ratio": round(ratio, 3),
                    "status": "OK" if 0.8 <= ratio <= 1.2 else "INVESTIGAR"
                })
        
        # Métricas gerais
        pd_medio_geral = pd_esperado.mean()
        default_rate_geral = default_realizado.mean()
        
        return {
            "data_execucao": datetime.now().isoformat(),
            "periodo_analise": "últimos 12 meses",
            "resumo": {
                "total_observacoes": len(pd_esperado),
                "total_defaults": int(default_realizado.sum()),
                "pd_medio_esperado": round(pd_medio_geral * 100, 2),
                "default_rate_realizado": round(default_rate_geral * 100, 2),
                "calibracao": "OK" if abs(pd_medio_geral - default_rate_geral) < 0.02 else "AJUSTAR"
            },
            "por_faixa": resultados_por_faixa
        }
    
    def gerar_relatorio_completo(self) -> Dict[str, Any]:
        """
        Gera relatório completo de monitoramento.
        
        Returns:
            Dict com relatório completo
        """
        metricas = self.obter_metricas_atuais()
        psi = self.calcular_psi()
        backtesting = self.executar_backtesting()
        
        return {
            "data_geracao": datetime.now().isoformat(),
            "status_geral": self._calcular_status_geral(metricas, psi),
            "metricas_atuais": metricas,
            "analise_drift": psi.to_dict(),
            "backtesting": backtesting,
            "evolucao_auc": self.obter_evolucao_temporal("auc_roc"),
            "evolucao_gini": self.obter_evolucao_temporal("gini"),
            "recomendacoes": self._gerar_recomendacoes(metricas, psi, backtesting)
        }
    
    def _calcular_status_geral(
        self,
        metricas: Dict,
        psi: ResultadoPSI
    ) -> Dict[str, Any]:
        """Calcula status geral do modelo."""
        alertas_vermelhos = 0
        alertas_amarelos = 0
        
        if psi.nivel_alerta == NivelAlerta.VERMELHO:
            alertas_vermelhos += 1
        elif psi.nivel_alerta == NivelAlerta.AMARELO:
            alertas_amarelos += 1
        
        for alerta in metricas.get("alertas", []):
            if alerta["nivel"] == NivelAlerta.VERMELHO.value:
                alertas_vermelhos += 1
            elif alerta["nivel"] == NivelAlerta.AMARELO.value:
                alertas_amarelos += 1
        
        if alertas_vermelhos > 0:
            status = NivelAlerta.VERMELHO
            descricao = "CRÍTICO: Ação imediata necessária"
        elif alertas_amarelos > 0:
            status = NivelAlerta.AMARELO
            descricao = "ATENÇÃO: Monitorar de perto"
        else:
            status = NivelAlerta.VERDE
            descricao = "NORMAL: Modelo performando bem"
        
        return {
            "nivel": status.value,
            "descricao": descricao,
            "alertas_vermelhos": alertas_vermelhos,
            "alertas_amarelos": alertas_amarelos
        }
    
    def _gerar_recomendacoes(
        self,
        metricas: Dict,
        psi: ResultadoPSI,
        backtesting: Dict
    ) -> List[str]:
        """Gera recomendações baseadas na análise."""
        recomendacoes = []
        
        if psi.nivel_alerta != NivelAlerta.VERDE:
            recomendacoes.append("Investigar causas do drift na população")
        
        for alerta in metricas.get("alertas", []):
            if "AUC" in alerta.get("metrica", ""):
                recomendacoes.append("Considerar retreinamento do modelo")
        
        calibracao = backtesting.get("resumo", {}).get("calibracao")
        if calibracao == "AJUSTAR":
            recomendacoes.append("Recalibrar parâmetros de PD")
        
        if not recomendacoes:
            recomendacoes.append("Manter monitoramento regular")
        
        return recomendacoes


# Instância global
_model_monitor: Optional[ModelMonitor] = None


def get_model_monitor() -> ModelMonitor:
    """Obtém ou cria instância do monitor."""
    global _model_monitor
    if _model_monitor is None:
        _model_monitor = ModelMonitor()
    return _model_monitor


__all__ = [
    "NivelAlerta",
    "MetricasModelo",
    "ResultadoPSI",
    "ModelMonitor",
    "get_model_monitor"
]

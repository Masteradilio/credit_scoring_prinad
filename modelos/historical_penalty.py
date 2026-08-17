"""
PRINAD - Historical Penalty Calculator v2.0
Implements the behavioral penalty component based on 24-month lookback.

v2.0 Changes:
- Split penalty into Internal (25%) and External/SCR (25%)
- Changed forgiveness period from 12 to 6 months
- Added SCR-based external penalty calculation
- Both internal AND external must be clean for cure
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DelinquencyLevel(Enum):
    """Credit risk situation levels (professional terminology)."""
    NONE = "clean"              # No issues
    SHORT_TERM = "leve"         # <= 120 days - Minor concern
    LONG_TERM = "moderado"      # 121-180 days - Moderate risk
    SEVERE = "severo"           # > 180 days - Elevated risk
    DEFAULT = "critico"         # Bank's internal default definition


@dataclass
class HistoricalAnalysis:
    """Result of historical behavior analysis."""
    penalidade_interna: float
    penalidade_externa: float
    penalidade_total: float
    nivel_delinquencia: DelinquencyLevel
    meses_desde_ultimo_atraso_interno: int
    meses_desde_ultimo_atraso_externo: int
    elegivel_perdao: bool
    detalhes: Dict[str, Any]


class HistoricalPenaltyCalculator:
    """
    Calculates historical penalty based on 24-month lookback window.
    
    v2.0 Methodology:
    - Lookback: 24 months
    - Internal penalty: Max 0.75 (25% weight)
    - External penalty (SCR): Max 0.75 (25% weight)
    - Forgiveness period: 6 months without delinquency (internal AND external)
    
    Internal Components:
        1. Recency of delinquency (0-0.4)
        2. Duration of delinquency (0-0.25)
        3. Recent regularization bonus penalty (0-0.1)
    
    External Components (SCR):
        1. Risk rating penalty (0-0.35)
        2. Overdue value penalty (0-0.2)
        3. Days overdue penalty (0-0.15)
        4. Prejuizo penalty (0-0.25)
    """
    
    # Configuration parameters
    LOOKBACK_MONTHS = 24
    FORGIVENESS_MONTHS = 6  # Changed from 12 to 6
    MAX_PENALTY_INTERNAL = 0.75  # Changed from 1.5 to 0.75
    MAX_PENALTY_EXTERNAL = 0.75
    
    # Internal penalty weights (reduced to fit 0.75 max)
    PENALTY_RECENCY = {
        'ultimos_6_meses': 0.4,
        'ultimos_12_meses': 0.25,
        'ultimos_24_meses': 0.15
    }
    
    PENALTY_DURATION = {
        'mais_12_meses': 0.25,
        'mais_6_meses': 0.15,
        'mais_3_meses': 0.1
    }
    
    PENALTY_REGULARIZATION = {
        'menos_3_meses': 0.1
    }
    
    # External (SCR) penalty weights
    SCR_RISK_RATING_PENALTY = {
        'AA': 0.0, 'A': 0.0, 'B': 0.05, 'C': 0.1,
        'D': 0.2, 'E': 0.25, 'F': 0.3, 'G': 0.35, 'H': 0.35
    }
    
    # V-column to days mapping
    V_COLS_DAYS = {
        'v205': 30,
        'v210': 60,
        'v220': 90,
        'v230': 120,
        'v240': 150,
        'v245': 180,
        'v250': 210,
        'v255': 240,
        'v260': 270,
        'v270': 300,
        'v280': 330,
        'v290': 360
    }
    
    def __init__(self, forgiveness_months: int = 6, 
                 max_penalty_internal: float = 0.75,
                 max_penalty_external: float = 0.75):
        """
        Initialize the calculator.
        
        Args:
            forgiveness_months: Months without delinquency for forgiveness (default: 6)
            max_penalty_internal: Maximum internal penalty multiplier
            max_penalty_external: Maximum external penalty multiplier
        """
        self.forgiveness_months = forgiveness_months
        self.max_penalty_internal = max_penalty_internal
        self.max_penalty_external = max_penalty_external
    
    def calculate(self, client_data: Dict[str, Any]) -> HistoricalAnalysis:
        """
        Calculate historical penalty for a client.
        
        Args:
            client_data: Dictionary with client behavioral data (v-columns + SCR columns)
            
        Returns:
            HistoricalAnalysis with penalty details
        """
        # Extract behavioral indicators
        behavior_internal = self._extract_internal_behavior(client_data)
        behavior_external = self._extract_external_behavior(client_data)
        
        # Check forgiveness eligibility (both internal AND external must be clean)
        internal_clean = behavior_internal['meses_desde_ultimo_atraso'] >= self.forgiveness_months
        external_clean = behavior_external['meses_desde_ultimo_atraso'] >= self.forgiveness_months
        
        if internal_clean and external_clean:
            return HistoricalAnalysis(
                penalidade_interna=0.0,
                penalidade_externa=0.0,
                penalidade_total=0.0,
                nivel_delinquencia=DelinquencyLevel.NONE,
                meses_desde_ultimo_atraso_interno=behavior_internal['meses_desde_ultimo_atraso'],
                meses_desde_ultimo_atraso_externo=behavior_external['meses_desde_ultimo_atraso'],
                elegivel_perdao=True,
                detalhes={'motivo': f'Cliente elegível para perdão ({self.forgiveness_months}+ meses limpo interno E externo)'}
            )
        
        detalhes = {}
        
        # Calculate internal penalty
        penalidade_interna = self._calculate_internal_penalty(behavior_internal)
        penalidade_interna = min(penalidade_interna, self.max_penalty_internal)
        detalhes['penalidade_interna'] = penalidade_interna
        detalhes['comportamento_interno'] = behavior_internal
        
        # Calculate external penalty (SCR)
        penalidade_externa = self._calculate_external_penalty(behavior_external)
        penalidade_externa = min(penalidade_externa, self.max_penalty_external)
        detalhes['penalidade_externa'] = penalidade_externa
        detalhes['comportamento_externo'] = behavior_external
        
        # Total penalty
        penalidade_total = penalidade_interna + penalidade_externa
        detalhes['penalidade_total'] = penalidade_total
        
        # Determine delinquency level (based on worse of internal/external)
        nivel = self._determine_delinquency_level(behavior_internal, behavior_external)
        
        return HistoricalAnalysis(
            penalidade_interna=penalidade_interna,
            penalidade_externa=penalidade_externa,
            penalidade_total=penalidade_total,
            nivel_delinquencia=nivel,
            meses_desde_ultimo_atraso_interno=behavior_internal['meses_desde_ultimo_atraso'],
            meses_desde_ultimo_atraso_externo=behavior_external['meses_desde_ultimo_atraso'],
            elegivel_perdao=False,
            detalhes=detalhes
        )
    
    def _extract_internal_behavior(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract internal behavioral indicators from client data."""
        
        behavior = {
            'tem_atraso_curto': False,
            'tem_atraso_longo': False,
            'tem_inadimplencia': False,
            'max_dias_atraso': 0,
            'meses_desde_ultimo_atraso': self.LOOKBACK_MONTHS + 1,
            'meses_total_atraso': 0,
            'regularizado_recentemente': False,
            'exposicao_total': 0
        }
        
        # Analyze v-columns
        for col, days in self.V_COLS_DAYS.items():
            value = client_data.get(col, 0)
            if value is None:
                value = 0
            value = float(value)
            
            if value > 0:
                behavior['exposicao_total'] += value
                
                if days <= 120:
                    behavior['tem_atraso_curto'] = True
                elif days <= 180:
                    behavior['tem_atraso_longo'] = True
                else:
                    behavior['tem_inadimplencia'] = True
                
                if days > behavior['max_dias_atraso']:
                    behavior['max_dias_atraso'] = days
        
        # Estimate months since last delinquency
        if behavior['max_dias_atraso'] > 0:
            behavior['meses_desde_ultimo_atraso'] = max(1, 12 - (behavior['max_dias_atraso'] // 30))
        
        # Estimate total months in delinquency
        if behavior['tem_inadimplencia']:
            behavior['meses_total_atraso'] = max(6, behavior['max_dias_atraso'] // 30)
        elif behavior['tem_atraso_longo']:
            behavior['meses_total_atraso'] = max(3, behavior['max_dias_atraso'] // 30)
        elif behavior['tem_atraso_curto']:
            behavior['meses_total_atraso'] = max(1, behavior['max_dias_atraso'] // 30)
        
        # Check if recently regularized
        v290 = float(client_data.get('v290', 0) or 0)
        if behavior['tem_inadimplencia'] and v290 == 0:
            behavior['regularizado_recentemente'] = True
            behavior['meses_desde_ultimo_atraso'] = 3
        
        return behavior
    
    def _extract_external_behavior(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract external (SCR) behavioral indicators from client data."""
        
        behavior = {
            'classificacao_risco': client_data.get('scr_classificacao_risco', 'A'),
            'dias_atraso': int(client_data.get('scr_dias_atraso', 0) or 0),
            'valor_vencido': float(client_data.get('scr_valor_vencido', 0) or 0),
            'valor_prejuizo': float(client_data.get('scr_valor_prejuizo', 0) or 0),
            'tem_prejuizo': int(client_data.get('scr_tem_prejuizo', 0) or 0) == 1,
            'taxa_utilizacao': float(client_data.get('scr_taxa_utilizacao', 0) or 0),
            'score_risco': int(client_data.get('scr_score_risco', 0) or 0),
            'meses_desde_ultimo_atraso': self.LOOKBACK_MONTHS + 1  # Default: clean
        }
        
        # Estimate months since last external issue
        if behavior['dias_atraso'] > 0:
            # Recent issue
            behavior['meses_desde_ultimo_atraso'] = max(1, 6 - (behavior['dias_atraso'] // 30))
        elif behavior['tem_prejuizo']:
            # Has historical prejuizo
            behavior['meses_desde_ultimo_atraso'] = 3
        elif behavior['score_risco'] >= 4:  # Rating D or worse
            behavior['meses_desde_ultimo_atraso'] = 4
        
        return behavior
    
    def _calculate_internal_penalty(self, behavior: Dict[str, Any]) -> float:
        """Calculate internal penalty from v-columns."""
        penalidade = 0.0
        
        # Recency penalty
        meses = behavior['meses_desde_ultimo_atraso']
        if meses <= 6:
            penalidade += self.PENALTY_RECENCY['ultimos_6_meses']
        elif meses <= 12:
            penalidade += self.PENALTY_RECENCY['ultimos_12_meses']
        elif meses <= 24:
            penalidade += self.PENALTY_RECENCY['ultimos_24_meses']
        
        # Duration penalty
        meses_atraso = behavior['meses_total_atraso']
        if meses_atraso > 12:
            penalidade += self.PENALTY_DURATION['mais_12_meses']
        elif meses_atraso > 6:
            penalidade += self.PENALTY_DURATION['mais_6_meses']
        elif meses_atraso > 3:
            penalidade += self.PENALTY_DURATION['mais_3_meses']
        
        # Regularization penalty
        if behavior['regularizado_recentemente']:
            if behavior['meses_desde_ultimo_atraso'] < 3:
                penalidade += self.PENALTY_REGULARIZATION['menos_3_meses']
        
        return penalidade
    
    def _calculate_external_penalty(self, behavior: Dict[str, Any]) -> float:
        """Calculate external penalty from SCR data."""
        penalidade = 0.0
        
        # 1. Risk rating penalty
        rating = behavior.get('classificacao_risco', 'A')
        penalidade += self.SCR_RISK_RATING_PENALTY.get(rating, 0)
        
        # 2. Overdue value penalty
        valor_vencido = behavior.get('valor_vencido', 0)
        if valor_vencido > 10000:
            penalidade += 0.2
        elif valor_vencido > 5000:
            penalidade += 0.15
        elif valor_vencido > 1000:
            penalidade += 0.1
        elif valor_vencido > 0:
            penalidade += 0.05
        
        # 3. Days overdue penalty
        dias_atraso = behavior.get('dias_atraso', 0)
        if dias_atraso > 90:
            penalidade += 0.15
        elif dias_atraso > 60:
            penalidade += 0.1
        elif dias_atraso > 30:
            penalidade += 0.05
        
        # 4. Prejuizo penalty
        if behavior.get('tem_prejuizo', False):
            penalidade += 0.25
        
        # 5. High utilization penalty (stress indicator)
        if behavior.get('taxa_utilizacao', 0) > 0.9:
            penalidade += 0.05
        
        return penalidade
    
    def _determine_delinquency_level(self, 
                                     behavior_int: Dict[str, Any],
                                     behavior_ext: Dict[str, Any]) -> DelinquencyLevel:
        """Determine the delinquency severity level (worse of internal/external)."""
        
        # Internal level
        if behavior_int['tem_inadimplencia']:
            int_level = DelinquencyLevel.SEVERE
        elif behavior_int['tem_atraso_longo']:
            int_level = DelinquencyLevel.LONG_TERM
        elif behavior_int['tem_atraso_curto']:
            int_level = DelinquencyLevel.SHORT_TERM
        else:
            int_level = DelinquencyLevel.NONE
        
        # External level
        score = behavior_ext.get('score_risco', 0)
        if score >= 7 or behavior_ext.get('tem_prejuizo', False):
            ext_level = DelinquencyLevel.SEVERE
        elif score >= 5:
            ext_level = DelinquencyLevel.LONG_TERM
        elif score >= 3:
            ext_level = DelinquencyLevel.SHORT_TERM
        else:
            ext_level = DelinquencyLevel.NONE
        
        # Return the worse level
        levels_order = [DelinquencyLevel.NONE, DelinquencyLevel.SHORT_TERM, 
                       DelinquencyLevel.LONG_TERM, DelinquencyLevel.SEVERE]
        
        int_idx = levels_order.index(int_level)
        ext_idx = levels_order.index(ext_level)
        
        return levels_order[max(int_idx, ext_idx)]
    
    def apply_penalty(self, pd_base: float, client_data: Dict[str, Any]) -> Tuple[float, HistoricalAnalysis]:
        """
        Apply historical penalty to base PD.
        
        Args:
            pd_base: Base PD from ML model (0-100)
            client_data: Client behavioral data (v-columns + SCR)
            
        Returns:
            Tuple of (final PRINAD, analysis details)
        """
        analysis = self.calculate(client_data)
        
        # Apply formula: PRINAD = PD_Base × (1 + Pen_Interna + Pen_Externa)
        prinad = pd_base * (1 + analysis.penalidade_total)
        
        # Cap at 100%
        prinad = min(100.0, prinad)
        
        return prinad, analysis


def calculate_historical_penalty(client_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Convenience function to calculate historical penalty.
    
    Args:
        client_data: Dictionary with client v-columns and SCR columns
        
    Returns:
        Tuple of (total penalty multiplier, details dict)
    """
    calculator = HistoricalPenaltyCalculator()
    analysis = calculator.calculate(client_data)
    
    return analysis.penalidade_total, {
        'penalidade_interna': analysis.penalidade_interna,
        'penalidade_externa': analysis.penalidade_externa,
        'nivel': analysis.nivel_delinquencia.value,
        'meses_desde_atraso_interno': analysis.meses_desde_ultimo_atraso_interno,
        'meses_desde_atraso_externo': analysis.meses_desde_ultimo_atraso_externo,
        'elegivel_perdao': analysis.elegivel_perdao,
        **analysis.detalhes
    }


if __name__ == "__main__":
    # Test the historical penalty calculator
    calculator = HistoricalPenaltyCalculator()
    
    # Test cases
    test_cases = [
        {
            "nome": "Cliente Limpo (Interno + Externo)",
            "data": {"v205": 0, "v210": 0, "v290": 0, 
                     "scr_classificacao_risco": "A", "scr_dias_atraso": 0, 
                     "scr_valor_vencido": 0, "scr_tem_prejuizo": 0},
            "pd_base": 10.0
        },
        {
            "nome": "Atraso Apenas Interno",
            "data": {"v205": 1000, "v210": 500, "v290": 0,
                     "scr_classificacao_risco": "A", "scr_dias_atraso": 0,
                     "scr_valor_vencido": 0, "scr_tem_prejuizo": 0},
            "pd_base": 15.0
        },
        {
            "nome": "Atraso Apenas Externo (SCR)",
            "data": {"v205": 0, "v210": 0, "v290": 0,
                     "scr_classificacao_risco": "E", "scr_dias_atraso": 45,
                     "scr_valor_vencido": 3000, "scr_tem_prejuizo": 0,
                     "scr_score_risco": 5, "scr_taxa_utilizacao": 0.8},
            "pd_base": 12.0
        },
        {
            "nome": "Atraso Interno + Externo (Alto Risco)",
            "data": {"v205": 2000, "v240": 5000, "v290": 10000,
                     "scr_classificacao_risco": "G", "scr_dias_atraso": 120,
                     "scr_valor_vencido": 8000, "scr_tem_prejuizo": 1,
                     "scr_score_risco": 7, "scr_taxa_utilizacao": 0.95},
            "pd_base": 20.0
        }
    ]
    
    print("\n" + "="*80)
    print("TESTE DO CALCULADOR DE PENALIDADE HISTÓRICA v2.0")
    print("Pesos: 50% ML | 25% Interno | 25% Externo (SCR)")
    print("Período de Cura: 6 meses (interno E externo limpos)")
    print("="*80)
    
    for case in test_cases:
        prinad, analysis = calculator.apply_penalty(case['pd_base'], case['data'])
        
        print(f"\n{case['nome']}:")
        print(f"  PD Base: {case['pd_base']:.1f}%")
        print(f"  Penalidade Interna: {analysis.penalidade_interna:.2f}")
        print(f"  Penalidade Externa: {analysis.penalidade_externa:.2f}")
        print(f"  Penalidade Total: {analysis.penalidade_total:.2f}")
        print(f"  PRINAD Final: {prinad:.1f}%")
        print(f"  Nível: {analysis.nivel_delinquencia.value}")
        print(f"  Elegível Perdão: {analysis.elegivel_perdao}")

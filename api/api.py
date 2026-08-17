"""
PRINAD - FastAPI Application v2.0
REST API for demonstrative credit-risk classification; regulatory conformity is not evaluated.

Endpoints:
- /health - Health check
- /simple_classify - Simple classification (CPF -> PRINAD, pd_12m, pd_lifetime, rating)
- /explained_classify - Classification with SHAP explanation
- /multiple_classify - Batch classification (list of CPFs)
- /multiple_explained_classify - Batch classification with SHAP
"""

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from enum import Enum
import pandas as pd
import numpy as np
import logging
import json
import sys
import io
import csv

# Add paths
CURRENT_DIR = Path(__file__).resolve().parent
MODELOS_DIR = CURRENT_DIR.parent / "modelos"
sys.path.insert(0, str(MODELOS_DIR))

from classifier import PRINADClassifier, ClassificationResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
# Expects 'dados' folder at the same level as 'api', 'modelos', etc.
DADOS_DIR = BASE_DIR / "dados"
# Create 'dados' folder if it doesn't exist
try:
    DADOS_DIR.mkdir(parents=True, exist_ok=True)
except FileExistsError:
    pass # Directory already exists

# Create FastAPI app
app = FastAPI(
    title="PRINAD API v2.0",
    description="API demonstrativa de classificação de risco com dados sintéticos; conformidade não avaliada",
    version="2.0.0",
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

# =============================================================================
# GLOBAL STATE
# =============================================================================
classifier: Optional[PRINADClassifier] = None
df_clientes: Optional[pd.DataFrame] = None
df_comportamental: Optional[pd.DataFrame] = None
df_scr: Optional[pd.DataFrame] = None

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class OutputFormat(str, Enum):
    json = "json"
    csv = "csv"


class CPFRequest(BaseModel):
    """Request with single CPF."""
    cpf: str = Field(..., description="CPF do cliente (apenas números)")


class MultipleCPFRequest(BaseModel):
    """Request with multiple CPFs."""
    cpfs: List[str] = Field(..., description="Lista de CPFs")
    output_format: OutputFormat = Field(OutputFormat.json, description="Formato de saída")


class SimpleClassificationResponse(BaseModel):
    """Simple classification response - BACEN 4966."""
    cpf: str
    prinad: float = Field(..., description="Score PRINAD (0-100)")
    pd_12m: float = Field(..., description="Probabilidade de Default em 12 meses")
    pd_lifetime: float = Field(..., description="Probabilidade de Default Lifetime")
    rating: str = Field(..., description="Rating (A1-DEFAULT)")
    estagio_pe: int = Field(..., description="Estágio de Perda Esperada / IFRS 9 (1, 2 ou 3)")
    timestamp: str


class ExplainedClassificationResponse(SimpleClassificationResponse):
    """Classification with full explanation."""
    rating_descricao: str
    cor: str
    acao_sugerida: str
    explicacao_shap: List[Dict[str, Any]] = Field(..., description="Explicação SHAP do modelo ML (50%)")
    pd_base: float
    penalidade_historica: float
    model_version: str
    explicacao_completa: Dict[str, Any] = Field(..., description="Explicação completa: composição PRINAD, justificativas de PD e Estágio PE")


class MultipleClassificationResponse(BaseModel):
    """Response for batch classification."""
    total: int
    sucesso: int
    erro: int
    resultados: List[Dict[str, Any]]
    erros: List[Dict[str, Any]]
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    database_loaded: bool
    total_clientes: int
    version: str
    timestamp: str


# =============================================================================
# DATA LOADING
# =============================================================================

def normalize_cpf(cpf: Any) -> str:
    """Normalize CPF to 11-digit string."""
    if pd.isna(cpf):
        return ""
    if isinstance(cpf, (int, float)):
        cpf = str(int(cpf))
    else:
        cpf = str(cpf).strip()
        cpf = ''.join(c for c in cpf if c.isdigit())
    return cpf.zfill(11)


def load_client_database():
    """Load client database from CSV files."""
    global df_clientes, df_comportamental, df_scr
    
    # Load base_clientes.csv (consolidated data)
    clientes_path = DADOS_DIR / "base_clientes.csv"
    if clientes_path.exists():
        df_clientes = pd.read_csv(clientes_path, sep=';', encoding='latin-1')
        df_clientes['CPF_NORM'] = df_clientes['CPF'].apply(normalize_cpf)
        logger.info(f"Loaded {len(df_clientes)} client records from base_clientes.csv")
    else:
        # Fallback: load from base_cadastro.csv
        cadastro_path = DADOS_DIR / "base_cadastro.csv"
        if cadastro_path.exists():
            df_clientes = pd.read_csv(cadastro_path, sep=';', encoding='latin-1')
            df_clientes['CPF_NORM'] = df_clientes['CPF'].apply(normalize_cpf)
            logger.info(f"Loaded {len(df_clientes)} client records from base_cadastro.csv")
        else:
            logger.warning("No client database found!")
            df_clientes = None
    
    # Load behavioral data (base_3040.csv)
    comportamental_path = DADOS_DIR / "base_3040.csv"
    if comportamental_path.exists():
        df_comportamental = pd.read_csv(comportamental_path, sep=';', encoding='latin-1')
        df_comportamental['CPF_NORM'] = df_comportamental['CPF'].apply(normalize_cpf)
        # Aggregate by CPF (keep max values for each v-column)
        v_cols = [c for c in df_comportamental.columns if c.startswith('v')]
        agg_dict = {c: 'max' for c in v_cols if c in df_comportamental.columns}
        if 'CLASSE' in df_comportamental.columns:
            agg_dict['CLASSE'] = 'last'
        df_comportamental = df_comportamental.groupby('CPF_NORM').agg(agg_dict).reset_index()
        logger.info(f"Loaded behavioral data for {len(df_comportamental)} clients")
    else:
        df_comportamental = None
        logger.warning("No behavioral data found!")
    
    # Load SCR data
    scr_path = DADOS_DIR / "scr_mock_data.csv"
    if scr_path.exists():
        df_scr = pd.read_csv(scr_path)
        logger.info(f"Loaded SCR data with {len(df_scr)} records")
    else:
        df_scr = None
        logger.warning("No SCR data found!")


def get_client_data(cpf: str) -> Optional[Dict[str, Any]]:
    """
    Fetch all data for a client by CPF.
    
    Returns dictionary with dados_cadastrais and dados_comportamentais.
    """
    cpf_norm = normalize_cpf(cpf)
    
    if df_clientes is None:
        return None
    
    # Find client in database
    client_row = df_clientes[df_clientes['CPF_NORM'] == cpf_norm]
    if client_row.empty:
        return None
    
    client = client_row.iloc[0].to_dict()
    
    # Check if we have a consolidated dataset (contains derived features)
    # If so, return full dict for both arguments to ensure all features are passed
    if 'em_idade_ativa' in client:
        return {
            'dados_cadastrais': client,
            'dados_comportamentais': client
        }
    
    # Build dados_cadastrais (Legacy logic)
    dados_cadastrais = {
        'IDADE_CLIENTE': client.get('IDADE_CLIENTE'),
        'RENDA_BRUTA': client.get('RENDA_BRUTA'),
        'RENDA_LIQUIDA': client.get('RENDA_LIQUIDA', client.get('RENDA_BRUTA', 0) * 0.8),
        'OCUPACAO': client.get('OCUPACAO', 'ASSALARIADO'),
        'ESCOLARIDADE': client.get('ESCOLARIDADE'),
        'ESTADO_CIVIL': client.get('ESTADO_CIVIL'),
        'QT_DEPENDENTES': client.get('QT_DEPENDENTES', 0),
        'TEMPO_RELAC': client.get('TEMPO_RELAC'),
        'TIPO_RESIDENCIA': client.get('TIPO_RESIDENCIA', 'PROPRIA'),
        'POSSUI_VEICULO': client.get('POSSUI_VEICULO'),
        'PORTABILIDADE': client.get('PORTABILIDADE', 'NAO'),
        'COMP_RENDA': client.get('comprometimento_renda', client.get('COMP_RENDA', 0.3)),
    }
    
    # Add SCR data if available in base_clientes
    if 'scr_score_risco' in client:
        dados_cadastrais['scr_score_risco'] = client.get('scr_score_risco')
        dados_cadastrais['scr_dias_atraso'] = client.get('scr_dias_atraso', 0)
        dados_cadastrais['scr_tem_prejuizo'] = client.get('scr_tem_prejuizo', 0)
    
    # Build dados_comportamentais
    dados_comportamentais = {}
    v_cols = ['v205', 'v210', 'v220', 'v230', 'v240', 'v245', 
              'v250', 'v255', 'v260', 'v270', 'v280', 'v290']
    
    # First try from base_clientes
    for v in v_cols:
        if v in client:
            dados_comportamentais[v] = float(client.get(v, 0))
    
    # If not found, try from df_comportamental
    if df_comportamental is not None and not dados_comportamentais:
        comp_row = df_comportamental[df_comportamental['CPF_NORM'] == cpf_norm]
        if not comp_row.empty:
            comp = comp_row.iloc[0]
            for v in v_cols:
                if v in comp:
                    dados_comportamentais[v] = float(comp.get(v, 0))
    
    # Default to zeros if no behavioral data
    if not dados_comportamentais:
        dados_comportamentais = {v: 0.0 for v in v_cols}
    
    # Use max_dias_atraso_12m if available
    if 'max_dias_atraso_12m' in client:
        dados_comportamentais['dias_atraso'] = int(client.get('max_dias_atraso_12m', 0))
    
    return {
        'cpf': cpf_norm,
        'dados_cadastrais': dados_cadastrais,
        'dados_comportamentais': dados_comportamentais
    }


# =============================================================================
# CLASSIFICATION FUNCTIONS
# =============================================================================

def classify_cpf(cpf: str, include_shap: bool = False) -> Dict[str, Any]:
    """
    Classify a single CPF.
    
    Args:
        cpf: Client CPF
        include_shap: Whether to include SHAP explanation
        
    Returns:
        Classification result as dictionary
    """
    if not classifier or not classifier.is_ready():
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    # Get client data
    client_data = get_client_data(cpf)
    if client_data is None:
        raise HTTPException(status_code=404, detail=f"CPF {cpf} não encontrado na base de dados")
    
    # Classify (with SHAP if requested)
    result = classifier.classify(client_data, include_shap=include_shap)
    
    # Build response
    if include_shap:
        return result.to_dict()
    else:
        return {
            'cpf': result.cpf,
            'prinad': round(result.prinad, 2),
            'pd_12m': round(result.pd_12m, 6),
            'pd_lifetime': round(result.pd_lifetime, 6),
            'rating': result.rating,
            'estagio_pe': result.estagio_pe,
            'timestamp': result.timestamp
        }


def classify_multiple_cpfs(cpfs: List[str], include_shap: bool = False) -> Tuple[List[Dict], List[Dict]]:
    """
    Classify multiple CPFs.
    
    Returns:
        Tuple of (results, errors)
    """
    results = []
    errors = []
    
    for cpf in cpfs:
        try:
            result = classify_cpf(cpf, include_shap)
            results.append(result)
        except HTTPException as e:
            errors.append({'cpf': cpf, 'erro': e.detail})
        except Exception as e:
            errors.append({'cpf': cpf, 'erro': str(e)})
    
    return results, errors


# =============================================================================
# STARTUP EVENT
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model and database on startup."""
    global classifier
    
    try:
        # Load database
        logger.info("Loading client database...")
        load_client_database()
        
        # Load classifier
        logger.info("Loading PRINAD classifier...")
        classifier = PRINADClassifier()
        
        if classifier.is_ready():
            logger.info("PRINAD classifier loaded successfully")
        else:
            logger.warning("Classifier loaded but model artifacts missing - using heuristic fallback")
            
    except Exception as e:
        logger.error(f"Error during startup: {e}")


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
async def health_check():
    """Verificar status da API."""
    return HealthResponse(
        status="healthy" if classifier and classifier.is_ready() else "degraded",
        model_loaded=classifier is not None and classifier.is_ready(),
        database_loaded=df_clientes is not None,
        total_clientes=len(df_clientes) if df_clientes is not None else 0,
        version="2.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.post("/simple_classify", response_model=SimpleClassificationResponse, tags=["Classificação"])
async def simple_classify(request: CPFRequest):
    """
    Classificação simples de risco de crédito.
    
    Recebe um CPF e retorna:
    - PRINAD score (0-100)
    - PD 12 meses
    - PD Lifetime
    - Rating (A1 a DEFAULT)
    - Stage IFRS 9 (1, 2 ou 3)
    """
    result = classify_cpf(request.cpf, include_shap=False)
    return SimpleClassificationResponse(**result)


@app.post("/explained_classify", response_model=ExplainedClassificationResponse, tags=["Classificação"])
async def explained_classify(request: CPFRequest):
    """
    Classificação com explicabilidade SHAP.
    
    Recebe um CPF e retorna a classificação completa incluindo:
    - Todos os campos da classificação simples
    - Explicação SHAP (top 5 features que influenciaram a decisão)
    - Ação sugerida
    - Descrição do rating
    """
    result = classify_cpf(request.cpf, include_shap=True)
    return ExplainedClassificationResponse(**result)


@app.post("/multiple_classify", tags=["Classificação em Lote"])
async def multiple_classify(request: MultipleCPFRequest):
    """
    Classificação em lote (múltiplos CPFs).
    
    Recebe uma lista de CPFs e retorna classificações simples.
    Suporta saída em JSON ou CSV.
    """
    results, errors = classify_multiple_cpfs(request.cpfs, include_shap=False)
    
    if request.output_format == OutputFormat.csv:
        # Return as CSV
        if not results:
            raise HTTPException(status_code=404, detail="Nenhum CPF encontrado")
        
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=classificacoes.csv"}
        )
    else:
        # Return as JSON
        return MultipleClassificationResponse(
            total=len(request.cpfs),
            sucesso=len(results),
            erro=len(errors),
            resultados=results,
            erros=errors,
            timestamp=datetime.now().isoformat()
        )


@app.post("/multiple_explained_classify", tags=["Classificação em Lote"])
async def multiple_explained_classify(request: MultipleCPFRequest):
    """
    Classificação em lote com explicabilidade SHAP.
    
    Recebe uma lista de CPFs e retorna classificações completas com SHAP.
    Suporta saída em JSON ou CSV.
    """
    results, errors = classify_multiple_cpfs(request.cpfs, include_shap=True)
    
    if request.output_format == OutputFormat.csv:
        # For CSV, flatten SHAP explanation
        if not results:
            raise HTTPException(status_code=404, detail="Nenhum CPF encontrado")
        
        # Simplify results for CSV (remove nested structures)
        csv_results = []
        for r in results:
            flat = {k: v for k, v in r.items() if k != 'explicacao_shap'}
            # Add top 3 SHAP features as columns
            shap = r.get('explicacao_shap', [])[:3]
            for i, feat in enumerate(shap, 1):
                flat[f'shap_{i}_feature'] = feat.get('feature', '')
                flat[f'shap_{i}_contribuicao'] = feat.get('contribuicao', 0)
            csv_results.append(flat)
        
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=csv_results[0].keys())
        writer.writeheader()
        writer.writerows(csv_results)
        
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=classificacoes_explicadas.csv"}
        )
    else:
        return MultipleClassificationResponse(
            total=len(request.cpfs),
            sucesso=len(results),
            erro=len(errors),
            resultados=results,
            erros=errors,
            timestamp=datetime.now().isoformat()
        )


@app.get("/clientes", tags=["Dados"])
async def list_clientes(limit: int = Query(100, le=1000)):
    """
    Listar CPFs disponíveis na base de dados.
    
    Útil para testes e validação.
    """
    if df_clientes is None:
        raise HTTPException(status_code=503, detail="Base de dados não carregada")
    
    cpfs = df_clientes['CPF_NORM'].head(limit).tolist()
    return {
        "total_base": len(df_clientes),
        "retornados": len(cpfs),
        "cpfs": cpfs
    }


@app.get("/cliente/{cpf}", tags=["Dados"])
async def get_cliente(cpf: str):
    """
    Obter dados brutos de um cliente pelo CPF.
    """
    client_data = get_client_data(cpf)
    if client_data is None:
        raise HTTPException(status_code=404, detail=f"CPF {cpf} não encontrado")
    return client_data


# =============================================================================
# LEGACY ENDPOINTS (backwards compatibility)
# =============================================================================

class LegacyClassificationRequest(BaseModel):
    """Legacy request format."""
    cpf: str
    dados_cadastrais: Dict[str, Any]
    dados_comportamentais: Dict[str, Any]
    sistema_origem: Optional[str] = None
    produto_credito: Optional[str] = None
    tipo_solicitacao: Optional[str] = None


@app.post("/predict", tags=["Legacy"])
async def predict_legacy(request: LegacyClassificationRequest):
    """
    [LEGACY] Endpoint de predição compatível com versão anterior.
    
    Recebe dados completos do cliente (não busca na base).
    Use /simple_classify ou /explained_classify para novos desenvolvimentos.
    """
    if not classifier or not classifier.is_ready():
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    result = classifier.classify({
        "cpf": request.cpf,
        "dados_cadastrais": request.dados_cadastrais,
        "dados_comportamentais": request.dados_comportamentais
    })
    
    return result.to_dict()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

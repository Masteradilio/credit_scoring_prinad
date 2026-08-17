import pytest
import sys
import os
import json
import logging
from pathlib import Path
from fastapi.testclient import TestClient

# Adicionar o diretório pai (raiz do módulo banpara) ao sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Importar a API
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))
from api import app

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_health_check(client):
    """Testa o endpoint de health check."""
    logger.info("Testando /health...")
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    logger.info(f"Health check response: {data}")
    assert data["database_loaded"] is True
    logger.info(f"Health check OK: {data}")

def test_list_clientes(client):
    """Testa a listagem de clientes."""
    logger.info("Testando /clientes...")
    response = client.get("/clientes?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert data["total_base"] > 0
    assert len(data["cpfs"]) > 0
    logger.info(f"Clientes encontrados: {len(data['cpfs'])}")
    return data["cpfs"][0] # Retorna um CPF para o próximo teste

def test_simple_classify(client, cpf):
    """Testa a classificação simples de um CPF."""
    logger.info(f"Testando /simple_classify para CPF: {cpf}")
    
    payload = {"cpf": cpf}
    response = client.post("/simple_classify", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    # Validações de negócio
    assert "prinad" in data
    assert "rating" in data
    assert "pd_12m" in data
    
    logger.info(f"Classificação OK: Rating {data['rating']}, Score {data['prinad']}")

def test_explained_classify(client, cpf):
    """Testa a classificação com explicação SHAP."""
    logger.info(f"Testando /explained_classify para CPF: {cpf}")
    
    payload = {"cpf": cpf}
    response = client.post("/explained_classify", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "explicacao_shap" in data
    
    logger.info("Explicação SHAP recebida com sucesso")

def test_invalid_cpf(client):
    """Testa comportamento com CPF inexistente."""
    cpf_invalido = "00000000000"
    logger.info(f"Testando CPF inválido: {cpf_invalido}")
    
    payload = {"cpf": cpf_invalido}
    response = client.post("/simple_classify", json=payload)
    
    assert response.status_code == 404
    logger.info("Erro 404 retornado corretamente")

if __name__ == "__main__":
    # Execução manual se rodar o script diretamente
    try:
        # Usar context manager para garantir startup/shutdown events
        with TestClient(app) as client:
            test_health_check(client)
            cpf = test_list_clientes(client)
            test_simple_classify(client, cpf)
            test_explained_classify(client, cpf)
            test_invalid_cpf(client)
            print("\n[SUCCESS] TODOS OS TESTES PASSARAM COM SUCESSO!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[ERROR] ERRO NOS TESTES: {e}")
        sys.exit(1)

# Documentação da API PRINAD v2.0

API REST para classificação de risco de crédito, conforme resolução BACEN 4966 e IFRS 9.

## Visão Geral
- **Tecnologia**: Python (FastAPI)
- **Porta Padrão**: 8000
- **Documentação Interativa**: `/docs` (Swagger UI) e `/redoc`

## Como Executar
1. Certifique-se de ter as dependências instaladas:
   ```bash
   pip install -r requirements.txt
   ```
2. Navegue até o diretório `api`:
   ```bash
   cd api
   ```
3. Execute o servidor:
   ```bash
   python api.py
   ```
   Ou via uvicorn diretamente:
   ```bash
   uvicorn api:app --reload
   ```

## Endpoints Principais

### 1. Classificação Simples
Retorna o score e ratings básicos.

- **URL**: `/simple_classify`
- **Método**: `POST`
- **Corpo da Requisição**:
  ```json
  {
    "cpf": "12345678900"
  }
  ```
- **Resposta de Sucesso (200 OK)**:
  ```json
  {
    "cpf": "12345678900",
    "prinad": 12.5,
    "pd_12m": 0.015,
    "pd_lifetime": 0.045,
    "rating": "A2",
    "estagio_pe": 1,
    "timestamp": "2026-01-27T10:00:00"
  }
  ```

### 2. Classificação Explicada (SHAP)
Retorna a classificação com detalhes e explicação do modelo (quais variáveis influenciaram a decisão).

- **URL**: `/explained_classify`
- **Método**: `POST`
- **Corpo da Requisição**:
  ```json
  {
    "cpf": "12345678900"
  }
  ```
- **Resposta de Sucesso (200 OK)**:
  ```json
  {
    "cpf": "12345678900",
    "prinad": 85.2,
    "rating": "D",
    "rating_descricao": "Pré-Default",
    "acao_sugerida": "Negação ou Garantia Real",
    "explicacao_shap": [
      {
        "feature": "v290",
        "contribuicao": 1.2,
        "direcao": "aumenta_risco"
      },
      {
        "feature": "score_atraso",
        "contribuicao": 0.8,
        "direcao": "aumenta_risco"
      }
    ],
    ...
  }
  ```

### 3. Classificação em Lote (Simples)
Processa múltiplos CPFs de uma vez, retornando apenas scores e ratings.

- **URL**: `/multiple_classify`
- **Método**: `POST`
- **Corpo da Requisição**:
  ```json
  {
    "cpfs": ["11111111111", "22222222222"],
    "output_format": "json"
  }
  ```
- **Nota**: Suporta `output_format: "csv"` para download direto de arquivo.

### 4. Classificação em Lote Explicada
Processa múltiplos CPFs de uma vez, retornando scores, ratings e explicações SHAP.

- **URL**: `/multiple_explained_classify`
- **Método**: `POST`
- **Corpo da Requisição**:
  ```json
  {
    "cpfs": ["11111111111", "22222222222"],
    "output_format": "json"
  }
  ```
- **Nota**: Suporta `output_format: "csv"`. No formato CSV, as top 3 features explicativas são "achatadas" em colunas (`shap_1_feature`, `shap_1_contribuicao`, etc.).

### 5. Health Check
Verifica se a API e o modelo estão carregados.

- **URL**: `/health`
- **Método**: `GET`

## Códigos de Erro
- **404 Not Found**: CPF não encontrado na base de dados carregada.
- **503 Service Unavailable**: Modelo ainda não foi carregado na memória.
- **422 Validation Error**: Formato de CPF inválido.

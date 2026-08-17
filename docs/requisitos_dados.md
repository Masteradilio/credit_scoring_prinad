# Requisitos de Dados - Módulo PRINAD

Este documento descreve o esquema de dados esperado pelo modelo PRINAD. O modelo foi treinado utilizando uma estrutura consolidada (Flat Table) contendo dados cadastrais, financeiros, comportamentais e do SCR.

O arquivo de entrada (CSV) deve conter as colunas listadas abaixo, separadas por ponto-e-vírgula (`;`).

## 1. Identificação e Perfil Cadastral

| Campo | Tipo | Descrição | Exemplo |
|-------|------|-----------|---------|
| `CPF` | String | CPF do cliente (chave primária) | `12345678900` |
| `CLIT` | Int | Código interno do cliente (opcional para o modelo, mas útil para rastreio) | `1` |
| `IDADE_CLIENTE` | Int | Idade em anos | `60` |
| `ESTADO_CIVIL` | String | Estado civil | `SOLTEIRO`, `CASADO` |
| `ESCOLARIDADE` | String | Nível de escolaridade | `MEDIO`, `SUPERIOR` |
| `POSSUI_VEICULO` | String | Indicador de posse de veículo | `SIM`, `NAO` |

## 2. Financeiro e Relacionamento

| Campo | Tipo | Descrição | Exemplo |
|-------|------|-----------|---------|
| `RENDA_BRUTA` | Float | Renda mensal bruta | `10147.06` |
| `TEMPO_RELAC` | Int | Tempo de relacionamento com o banco (meses) | `14` |
| `QT_PRODUTOS` | Int | Quantidade de produtos contratados | `3` |
| `produto` | String | Tipo de produto principal ou carteira | `banparacard`, `consignado` |

## 3. Dados Comportamentais (V-Columns)
Histórico de atrasos (valores monetários ou contagem) em faixas de dias.

| Campo | Dias de Atraso |
|-------|----------------|
| `v205` | 30 dias |
| `v210` | 60 dias |
| `v220` | 90 dias |
| `v230` | 120 dias |
| `v240` | 150 dias |
| `v245` | 180 dias |
| `v250` | 210 dias |
| `v255` | 240 dias |
| `v260` | 270 dias |
| `v270` | 300 dias |
| `v280` | 330 dias |
| `v290` | 360+ dias |

## 4. Dados do SCR (BACEN)

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `scr_score_risco` | Int | Score de risco calculado pelo BACEN/Bureau |
| `scr_dias_atraso` | Int | Máximo de dias de atraso no mercado |
| `scr_tem_prejuizo` | Bool/Int | Indicador de prejuízo no SCR (1=Sim, 0=Não) |

## 5. Métricas de Crédito e Capacidade de Pagamento

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `limite_total` | Float | Limite total de crédito disponível |
| `limite_utilizado` | Float | Valor do limite já utilizado |
| `taxa_utilizacao` | Float | Razão entre utilizado e total (0.0 a 1.0) |
| `parcelas_mensais` | Float | Valor total das parcelas mensais |
| `comprometimento_renda` | Float | % da renda comprometida com parcelas |
| `margem_disponivel` | Float | Valor financeiro livre na renda |
| `utilizacao_media_12m` | Float | Média de utilização do limite nos últimos 12 meses |
| `trimestres_sem_uso` | Int | Quantidade de trimestres sem uso de crédito |
| `max_dias_atraso_12m` | Int | Máximo de dias de atraso no último ano |

## 6. Variáveis Derivadas (Engenharia de Features)
Estas colunas podem ser calculadas automaticamente pelo pipeline se os dados brutos estiverem disponíveis, mas estão presentes na base de treino.

| Campo | Descrição | Fórmula/Lógica |
|-------|-----------|----------------|
| `em_idade_ativa` | Indicador se cliente está em idade laboral (18-65) | `1` se idade entre 18 e 65 |
| `idade_squared` | Idade ao quadrado (para capturar não-linearidade) | `IDADE_CLIENTE ^ 2` |
| `cliente_novo` | Indicador de cliente recente | `1` se TEMPO_RELAC < X meses |
| `log_tempo_relac` | Logaritmo do tempo de relacionamento | `log(TEMPO_RELAC)` |
| `tem_veiculo` | Binário de posse de veículo | `1` se POSSUI_VEICULO == 'SIM' |
| `score_escolaridade` | Score numérico atribuído à escolaridade | Mapeamento (ex: Superior=2, Medio=1) |
| `score_estado_civil` | Score numérico atribuído ao estado civil | Mapeamento de risco |

## 7. Variável Alvo (Apenas Treino)

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `CLASSE` | Int | Classificação real (0=Bom Pagador, 1=Mau Pagador/Default) |

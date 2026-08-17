"""
PRINAD - Credit Risk & Probability of Default (PD) Executive Dashboard
======================================================================
Interactive Quantitative Risk Intelligence Platform:
1. Underwriting Cockpit: Scorecard Points Breakdown & Glass-Box Explainability
2. Macroeconomic Stress Testing: Vasicek PIT <-> TTC & IFRS 9 Multi-Scenario Overlay
3. Champion vs. Challenger Arena: Scorecard vs. LightGBM vs. XGBoost vs. Ensemble
4. 4-Pillar Validation & Basel Backtesting: Binomial Tests & Basel Traffic Lights
5. Risk-Based Pricing & Cut-off Optimizer: RAROC & Profit Maximization Curves

Author: PRINAD Quantitative Risk Team
Standard: Basel III/IV IRB & IFRS 9 / BACEN 4.966
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import sys

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELOS_DIR = BASE_DIR / "modelos"
ARTEFATOS_DIR = BASE_DIR / "artefatos"
DADOS_DIR = BASE_DIR / "dados"

if str(MODELOS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELOS_DIR))

from classifier import PRINADClassifier, RatingMasterScale
from vasicek_macro import VasicekMacroEngine
from lifetime_pd import LifetimePDEngine
from decision_pricing_engine import DecisionAndPricingEngine
from data_pipeline import load_client_database

# Page Configuration
st.set_page_config(
    page_title="PRINAD | Credit Risk Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        color: #ffffff;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .badge-stage1 { background-color: #10b981; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
    .badge-stage2 { background-color: #f59e0b; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
    .badge-stage3 { background-color: #ef4444; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_classifier():
    return PRINADClassifier(model_type="scorecard")


@st.cache_data
def get_benchmark_report():
    report_path = ARTEFATOS_DIR / "model_comparison_report.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def get_client_sample():
    return load_client_database()


def main():
    st.title("🏦 PRINAD - Motor de Probabilidade de Inadimplência (PD) & Risco de Crédito")
    st.caption("Padrão Internacional Basileia III/IV IRB • IFRS 9 / BACEN 4.966 • Modelos Champion-vs-Challenger")
    
    classifier = get_classifier()
    df_clients = get_client_sample()
    benchmark_report = get_benchmark_report()
    
    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Configurações do Modelo")
        model_choice = st.selectbox(
            "Arquitetura de Modelagem:",
            ["scorecard", "lightgbm", "xgboost", "ensemble"],
            format_func=lambda x: {
                "scorecard": "🏆 Champion: Regulatory Scorecard (WoE)",
                "lightgbm": "⚡ Challenger 1: LightGBM (Calibrado)",
                "xgboost": "🌲 Challenger 2: XGBoost (Calibrado)",
                "ensemble": "🤖 Challenger 3: Stacking Ensemble"
            }[x]
        )
        
        st.subheader("👤 Seleção de Tomador de Crédito")
        if df_clients is not None and not df_clients.empty:
            selected_idx = st.selectbox(
                "Escolha um Cliente da Base:",
                range(min(200, len(df_clients))),
                format_func=lambda i: f"CPF {df_clients.iloc[i].get('CPF_NORM', df_clients.iloc[i].get('CPF', ''))} | Renda R$ {df_clients.iloc[i].get('RENDA_BRUTA', 0):,.0f}"
            )
            selected_row = df_clients.iloc[selected_idx].to_dict()
        else:
            selected_row = {
                'CPF': '12345678900', 'IDADE_CLIENTE': 38, 'RENDA_BRUTA': 7500.0,
                'RENDA_LIQUIDA': 6000.0, 'ESCOLARIDADE': 'SUPERIOR', 'OCUPACAO': 'ASSALARIADO',
                'ESTADO_CIVIL': 'CASADO', 'TEMPO_RELAC': 48, 'COMP_RENDA': 0.28,
                'limite_total': 35000.0, 'limite_utilizado': 9800.0, 'scr_dias_atraso': 0
            }
            
        loan_amount = st.number_input("Valor da Operação de Crédito (EAD R$):", min_value=1000.0, max_value=500000.0, value=15000.0, step=1000.0)
        asset_class = st.selectbox(
            "Modalidade da Carteira:",
            ["retail_other", "retail_revolving", "retail_mortgage", "corporate"],
            format_func=lambda x: {
                "retail_other": "Crédito Pessoal / Consignado / Veículos",
                "retail_revolving": "Cartão de Crédito / Rotativo",
                "retail_mortgage": "Financiamento Imobiliário",
                "corporate": "Empresas / Capital de Giro"
            }[x]
        )

    # Classify Current Borrower
    result = classifier.classify_borrower(
        borrower_data=selected_row,
        model_type=model_choice,
        loan_amount=loan_amount,
        asset_class=asset_class
    )

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Cockpit de Concessão Individual",
        "🌐 Simulador Macroeconômico (Vasicek & IFRS 9)",
        "⚔️ Arena Champion vs. Challenger",
        "📊 Validação nos 4 Pilares & Backtesting",
        "💰 Precificação Baseada em Risco & Cut-off"
    ])

    # =========================================================================
    # TAB 1: COCKPIT DE DECISÃO INDIVIDUAL
    # =========================================================================
    with tab1:
        st.subheader("📋 Parecer de Risco do Tomador")
        
        kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
        with kpi1:
            st.metric("Credit Score", f"{result.prinad_score} pts", help="Pontuação calculada de 300 a 850")
        with kpi2:
            st.metric("PD 12 Meses (PIT)", f"{result.pd_12m_pit_pct:.2f}%", help="Point-in-Time Probability of Default")
        with kpi3:
            st.metric("Rating de Crédito", result.rating, help=result.rating_descricao)
        with kpi4:
            st.metric("Estágio IFRS 9 / 4966", f"Stage {result.estagio_pe}", help=result.estagio_descricao)
        with kpi5:
            st.metric("Provisão ECL", f"R$ {result.ecl_provision_amount:,.2f}", help="Perda Esperada IFRS 9")
        with kpi6:
            st.metric("Taxa Justa (RAROC)", f"{result.fair_interest_rate_pct:.2f}% a.a.", help="Precificação ajustada ao risco")
            
        st.info(f"**Recomendação de Política de Crédito:** {result.acao_sugerida} | **Arquitetura Utilizada:** `{result.model_architecture}`")
        
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown("#### 🔍 Decomposição em Pontos (Scorecard Glass-Box)")
            if result.scorecard_points_breakdown:
                pts_df = pd.DataFrame(result.scorecard_points_breakdown)
                pts_df.columns = ['Variável', 'Valor Bruto', 'Faixa / Bin', 'WoE', 'IV', 'Pontos Atribuídos', 'Bad Rate da Faixa']
                st.dataframe(pts_df[['Variável', 'Valor Bruto', 'Faixa / Bin', 'WoE', 'Pontos Atribuídos']], use_container_width=True, hide_index=True)
                
                fig_pts = px.bar(
                    pts_df, x='Pontos Atribuídos', y='Variável', orientation='h',
                    title="Contribuição de Cada Variável no Score Final",
                    color='Pontos Atribuídos', color_continuous_scale='Greens'
                )
                st.plotly_chart(fig_pts, use_container_width=True)
            else:
                st.write("Scorecard points disponíveis no modo Champion (Scorecard).")
                
        with col_right:
            st.markdown("#### 📊 Indicadores Financeiros do Tomador")
            info_data = {
                "Idade": f"{selected_row.get('IDADE_CLIENTE', '-')} anos",
                "Renda Bruta": f"R$ {selected_row.get('RENDA_BRUTA', 0):,.2f}",
                "Comprometimento de Renda": f"{selected_row.get('COMP_RENDA', 0)*100:.1f}%",
                "Limite Utilizado / Total": f"R$ {selected_row.get('limite_utilizado', 0):,.0f} / R$ {selected_row.get('limite_total', 0):,.0f}",
                "Atraso SCR Bacen": f"{selected_row.get('scr_dias_atraso', 0)} dias",
                "Score Risco SCR": f"{selected_row.get('scr_score_risco', '-')}",
                "Tempo de Relacionamento": f"{selected_row.get('TEMPO_RELAC', 0)} meses"
            }
            st.json(info_data)
            
            # Rating Scale Visualization
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result.prinad_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Termômetro de Score (300 a 850)"},
                gauge={
                    'axis': {'range': [300, 850]},
                    'bar': {'color': "#3b82f6"},
                    'steps': [
                        {'range': [300, 500], 'color': "#fee2e2"},
                        {'range': [500, 650], 'color': "#fef3c7"},
                        {'range': [650, 850], 'color': "#d1fae5"}
                    ]
                }
            ))
            fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

    # =========================================================================
    # TAB 2: SIMULADOR MACROECONÔMICO & IFRS 9 (VASICEK)
    # =========================================================================
    with tab2:
        st.subheader("🌐 Estresse Macroeconômico: Modelo de Vasicek & Cenários IFRS 9")
        st.markdown("""
        Conforme o **Bloco 2 do Infográfico**, o modelo computa:
        - **$PD_{TTC}$**: Risco estrutural ao longo do ciclo econômico (Basileia III/IV).
        - **$PD_{PIT}(Z)$**: Risco condicional no ponto no tempo via Equação de Vasicek ASRF.
        """)
        
        sim_col1, sim_col2 = st.columns([1, 2])
        
        with sim_col1:
            st.markdown("##### 🎛️ Choque Macroeconômico Personalizado")
            sim_gdp = st.slider("Crescimento do PIB (% a.a.)", min_value=-5.0, max_value=6.0, value=2.0, step=0.5)
            sim_selic = st.slider("Taxa Básica SELIC (% a.a.)", min_value=5.0, max_value=20.0, value=10.5, step=0.5)
            sim_unemp = st.slider("Taxa de Desemprego (%)", min_value=4.0, max_value=16.0, value=7.8, step=0.5)
            
            vasicek = VasicekMacroEngine(default_asset_class=asset_class)
            z_shock = vasicek.calculate_z_factor(sim_gdp, sim_selic, sim_unemp)
            pd_shocked = vasicek.ttc_to_pit(result.pd_ttc, z_shock, asset_class=asset_class)
            
            st.metric("Fator Sistemático Z", f"{z_shock:+.2f} σ", help="Z > 0 Expansão | Z < 0 Recessão")
            st.metric("PD Estressada", f"{pd_shocked*100:.2f}%", delta=f"{(pd_shocked - result.pd_12m_pit)*100:+.2f}% vs Base")

        with sim_col2:
            st.markdown("##### 📈 Comparativo dos 3 Cenários Ponderados IFRS 9")
            sc_data = result.macro_scenarios.get('scenarios', [])
            if sc_data:
                sc_df = pd.DataFrame([
                    {
                        'Cenário': s['scenario_name'],
                        'Peso': f"{s['probability_weight']*100:.0f}%",
                        'Z-Score': s['z_factor'],
                        'PD Condicional (%)': s['conditional_pd_pct']
                    } for s in sc_data
                ])
                st.dataframe(sc_df, use_container_width=True, hide_index=True)
                
                fig_sc = px.bar(
                    sc_df, x='Cenário', y='PD Condicional (%)', color='Cenário',
                    title="Variação da Probabilidade de Inadimplência por Cenário IFRS 9",
                    text_auto='.2f'
                )
                st.plotly_chart(fig_sc, use_container_width=True)

        st.markdown("---")
        st.markdown("##### ⏳ Estrutura a Termo da PD (Curva Lifetime - Anos 1 a 5)")
        if result.lifetime_curve:
            lt_df = pd.DataFrame(result.lifetime_curve)
            
            fig_lt = go.Figure()
            fig_lt.add_trace(go.Scatter(x=lt_df['Year'], y=lt_df['Cumulative_PD_Pct'], mode='lines+markers+text', text=lt_df['Cumulative_PD_Pct'], textposition="top center", name="PD Acumulada (%)", line=dict(color='#ef4444', width=3)))
            fig_lt.add_trace(go.Bar(x=lt_df['Year'], y=lt_df['Marginal_PD_Pct'], name="PD Marginal Anual (%)", marker_color='#3b82f6', opacity=0.7))
            fig_lt.update_layout(title="Curva de Sobrevivência e PD Marginal Multi-Ano (Matriz de Transição de Markov)", xaxis_title="Ano da Operação", yaxis_title="Percentual (%)", height=350)
            st.plotly_chart(fig_lt, use_container_width=True)

    # =========================================================================
    # TAB 3: ARENA CHAMPION VS. CHALLENGER
    # =========================================================================
    with tab3:
        st.subheader("⚔️ Benchmark de Modelos: Champion Regulatório vs. Challengers de ML")
        st.markdown("""
        Comparativo rigoroso de performance entre o **Scorecard Regulatório (Champion)** e os algoritmos modernos de **Gradient Boosting / Ensemble (Challengers)**.
        """)
        
        if benchmark_report:
            bench_summary = benchmark_report.get('benchmark_summary', {})
            df_bench = pd.DataFrame(bench_summary).T.reset_index()
            df_bench.columns = ['Modelo', 'AUC-ROC', 'Gini', 'KS Stat', 'Brier Score', 'ECE', 'Hosmer-Lemeshow p', 'Semáforo Basileia', 'Status']
            
            st.dataframe(df_bench, use_container_width=True, hide_index=True)
            
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                fig_gini = px.bar(
                    df_bench, x='Modelo', y='Gini', color='Gini',
                    title="Poder Discriminatório (Gini Coefficient = 2 * AUC - 1)",
                    color_continuous_scale='Blues', text_auto='.4f'
                )
                st.plotly_chart(fig_gini, use_container_width=True)
            with col_b2:
                fig_brier = px.bar(
                    df_bench, x='Modelo', y='Brier Score', color='Brier Score',
                    title="Erro de Calibração (Brier Score Loss - Menor é Melhor)",
                    color_continuous_scale='Reds_r', text_auto='.4f'
                )
                st.plotly_chart(fig_brier, use_container_width=True)
        else:
            st.info("Execute `train_model.py` para gerar o relatório consolidado de benchmark.")

    # =========================================================================
    # TAB 4: VALIDAÇÃO NOS 4 PILARES & BACKTESTING
    # =========================================================================
    with tab4:
        st.subheader("📊 Validação de Modelos nos 4 Pilares de Basileia & IFRS 9")
        
        pilar_col1, pilar_col2, pilar_col3, pilar_col4 = st.columns(4)
        with pilar_col1:
            st.markdown("""
            <div class="metric-card">
                <h4>1. Discriminação</h4>
                <p>Separação entre bons e maus pagadores.</p>
                <b>Métricas:</b> Gini, AUC-ROC, KS.
            </div>
            """, unsafe_allow_html=True)
        with pilar_col2:
            st.markdown("""
            <div class="metric-card">
                <h4>2. Calibração</h4>
                <p>Aderência das probabilidades à taxa real.</p>
                <b>Métricas:</b> Brier, ECE, Hosmer-Lemeshow.
            </div>
            """, unsafe_allow_html=True)
        with pilar_col3:
            st.markdown("""
            <div class="metric-card">
                <h4>3. Estabilidade</h4>
                <p>Consistência da população no tempo.</p>
                <b>Métricas:</b> PSI temporal, CSI.
            </div>
            """, unsafe_allow_html=True)
        with pilar_col4:
            st.markdown("""
            <div class="metric-card">
                <h4>4. Backtesting</h4>
                <p>Testes estatísticos de hipótese.</p>
                <b>Métricas:</b> Teste Binomial, Semáforo Basileia.
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        st.markdown("##### 🚦 Semáforo de Basileia & Teste Binomial Exato por Faixa de Rating")
        
        # Display Basel Backtest Matrix
        if benchmark_report:
            scorecard_details = benchmark_report.get('detailed_validation', {}).get('Champion_Scorecard', {})
            backtest_data = scorecard_details.get('basel_backtest', {}).get('rating_bands_backtest', [])
            
            if backtest_data:
                bt_df = pd.DataFrame(backtest_data)
                bt_df.columns = ['Rating', 'Exposições (N)', 'Defaults Observados (k)', 'PD Esperada (%)', 'Taxa Real (%)', 'IC 95% Clopper-Pearson', 'p-value Binomial', 'Zona de Basileia']
                st.dataframe(bt_df, use_container_width=True, hide_index=True)
                
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Bar(x=bt_df['Rating'], y=bt_df['PD Esperada (%)'], name='PD Prevista (%)', marker_color='#3b82f6'))
                fig_bt.add_trace(go.Bar(x=bt_df['Rating'], y=bt_df['Taxa Real (%)'], name='Taxa Real de Default (%)', marker_color='#ef4444'))
                fig_bt.update_layout(title="Backtesting: PD Esperada vs Taxa Realizada por Rating", barmode='group', height=350)
                st.plotly_chart(fig_bt, use_container_width=True)

    # =========================================================================
    # TAB 5: PRECIFICAÇÃO BASEADA EM RISCO & CUT-OFF
    # =========================================================================
    with tab5:
        st.subheader("💰 Precificação de Crédito (RAROC) & Curva de Otimização de Cut-off")
        
        pr_col1, pr_col2 = st.columns([1, 1])
        
        pricing_engine = DecisionAndPricingEngine()
        pricing_breakdown = pricing_engine.price_credit(result.pd_12m_pit, asset_class=asset_class)
        
        with pr_col1:
            st.markdown("##### 🏦 Estrutura da Taxa de Juros Recomendada (Lending Rate)")
            rate_df = pd.DataFrame([
                {'Componente': '1. Custo de Captação (FTP / Selic)', 'Percentual': f"{pricing_breakdown.cost_of_funds_pct:.2f}%"},
                {'Componente': '2. Despesas Operacionais (OpEx)', 'Percentual': f"{pricing_breakdown.opex_cost_pct:.2f}%"},
                {'Componente': '3. Perda Esperada (EL = PD * LGD)', 'Percentual': f"{pricing_breakdown.expected_loss_pct:.2f}%"},
                {'Componente': '4. Custo de Capital Econômico (Hurdle Rate)', 'Percentual': f"{pricing_breakdown.capital_charge_pct:.2f}%"},
                {'Componente': '5. Margem de Lucro Desejada', 'Percentual': f"{pricing_breakdown.target_net_margin_pct:.2f}%"},
                {'Componente': 'Total: Taxa Justa de Empréstimo', 'Percentual': f"{pricing_breakdown.fair_lending_rate_annual:.2f}% a.a."}
            ])
            st.dataframe(rate_df, use_container_width=True, hide_index=True)
            
            fig_waterfall = go.Figure(go.Waterfall(
                orientation="v",
                measure=["relative", "relative", "relative", "relative", "relative", "total"],
                x=["Captação", "OpEx", "Perda Esp. (EL)", "Custo Capital", "Margem Lucro", "Taxa Final"],
                y=[pricing_breakdown.cost_of_funds_pct, pricing_breakdown.opex_cost_pct, pricing_breakdown.expected_loss_pct, pricing_breakdown.capital_charge_pct, pricing_breakdown.target_net_margin_pct, 0],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                text=[f"{v:.1f}%" for v in [pricing_breakdown.cost_of_funds_pct, pricing_breakdown.opex_cost_pct, pricing_breakdown.expected_loss_pct, pricing_breakdown.capital_charge_pct, pricing_breakdown.target_net_margin_pct, pricing_breakdown.fair_lending_rate_annual]]
            ))
            fig_waterfall.update_layout(title="Formação do Preço do Crédito (% a.a.)", height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_waterfall, use_container_width=True)

        with pr_col2:
            st.markdown("##### 🎯 Otimização do Ponto de Corte (Cut-off de Lucro Líquido)")
            if df_clients is not None and not df_clients.empty:
                # Run Cutoff simulation
                sample_pds = classifier.scorecard_model.predict_proba(df_clients) if classifier.scorecard_model else np.random.beta(2, 8, len(df_clients))
                cutoff_sim = pricing_engine.optimize_cutoff(sample_pds, average_loan_amount=loan_amount, average_interest_rate=0.28)
                
                st.metric("Ponto de Corte Ótimo de PD", f"{cutoff_sim['optimal_pd_cutoff_pct']:.1f}%", help="Threshold que maximiza o Lucro Líquido total da carteira")
                st.metric("Lucro Líquido Projetado da Carteira", f"R$ {cutoff_sim['max_net_profit']:,.2f}")
                
                sim_df = pd.DataFrame(cutoff_sim['simulation_curve'])
                fig_cut = go.Figure()
                fig_cut.add_trace(go.Scatter(x=sim_df['pd_cutoff_pct'], y=sim_df['net_profit_total'], mode='lines', name='Lucro Líquido Total (R$)', line=dict(color='#10b981', width=3)))
                fig_cut.update_layout(title="Curva de Rentabilidade Econômica vs Threshold de PD", xaxis_title="Threshold de Corte de PD (%)", yaxis_title="Lucro Líquido Total (R$)", height=320, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_cut, use_container_width=True)


if __name__ == "__main__":
    main()

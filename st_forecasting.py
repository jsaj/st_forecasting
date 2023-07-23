import plotly.graph_objects as go
import streamlit as st

from functions import *

st.set_page_config(
    page_title="Time series annotations", page_icon="⬇"
)

# @st.cache(allow_output_mutation=True)
# @st.cache_data
def load_data():
    df_despesa = pd.read_csv('https://raw.githubusercontent.com/jsaj/st_forecasting/master/datasets/despesa.csv')
    df_receita = pd.read_csv('https://raw.githubusercontent.com/jsaj/st_forecasting/master/datasets/receita.csv')
    return df_despesa, df_receita

    # Carrega os dados ou utiliza os dados em cache
df_despesa, df_receita = load_data()

df_receita_despesa = df_receita.merge(df_despesa, how='left', on=['ds', 'Unidade Gestora']).fillna(0)
df_receita_despesa['ds'] = pd.to_datetime(df_receita_despesa['ds'])

# Lista com todas as opções únicas da coluna 'Cód. Unidade Gestora'
opcoes_unidade_gestora = list(df_receita_despesa['Unidade Gestora'].drop_duplicates())

# Sidebar com o filtro
filtro_unidade_gestora = st.sidebar.selectbox('Unidade Gestora: ',
                                              opcoes_unidade_gestora,
                                              index=opcoes_unidade_gestora.index('FUNDO MUNICIPAL DE SAUDE'))

# Sidebar com o filtro dos meses de previsão
n_periods = st.sidebar.selectbox('Meses de previsão: ', list(range(1, 13)))

# Criar uma barra deslizante (slider) para selecionar a variável exógerna
model_name = st.sidebar.selectbox('Modelo preditivo:', ['ARIMA', 'Prophet'])

# Filtrar o DataFrame com base nas opções selecionadas no filtro
df_filtrado = df_receita_despesa[df_receita_despesa['Unidade Gestora'] == filtro_unidade_gestora]

# Criar uma barra deslizante (slider) para selecionar a variável exógerna
op_exog = st.sidebar.selectbox('Usar variável exógena?:', ['Sim', 'Não'])

if op_exog == 'Sim':

    # Criar uma barra deslizante (slider) para selecionar a variável exógerna
    exog_var = st.sidebar.selectbox('Variável exógena:', ['Valor Empenhado', 'Valor Liquidado', 'Valor Pago'])

    # Criar uma barra deslizante (slider) para selecionar a porcentagem
    porcentagem = st.sidebar.slider('% vs. Var. Exógena:', min_value=0, max_value=100, value=100, step=1)


    # Renomear as colunas para que o Prophet possa reconhecê-las
    df_filtrado = df_filtrado.rename(columns={'Vlr. Receita Realizada': 'y'})

    # Aplicar a função ao DataFrame para criar uma nova coluna com os valores multiplicados
    df_filtrado[exog_var] = df_filtrado[exog_var] * (porcentagem / 100)

    # Criar o modelo de previsão
    if model_name == 'ARIMA':
        predictions = predict_ARIMA(df=df_filtrado, n_periods=n_periods, exog_var=exog_var)
        df_filtrado = df_filtrado.reset_index()
    elif model_name == 'Prophet':
        predictions = predict_prophet(df=df_filtrado, n_periods=n_periods, exog_var=exog_var)


else:
    # Criar o modelo de previsão
    if model_name == 'ARIMA':
        predictions = predict_ARIMA(df=df_filtrado, n_periods=n_periods, exog_var=None)
        df_filtrado = df_filtrado.reset_index()
    elif model_name == 'Prophet':
        predictions = predict_prophet(df=df_filtrado, n_periods=n_periods, exog_var=None)


st.write(df_filtrado)

#
# # Criar o gráfico de linhas
fig = go.Figure()

# Adicionar a série original de receita realizada
fig.add_trace(go.Scatter(x=df_filtrado['ds'], y=df_filtrado['y'], mode='lines', name='Receita Realizada', line=dict(color='blue')))

# Adicionar a série de previsão de receita
fig.add_trace(go.Scatter(x=predictions['ds'], y=predictions['yhat'], mode='lines', name='Receita Predita', line=dict(color='green')))

# Personalizar layout do gráfico
fig.update_layout(title='Receita Realizada vs. Receita Predita',
                  xaxis_title='Data',
                  yaxis_title='Receita',
                  showlegend=True)

# Exibir o gráfico usando Streamlit
st.plotly_chart(fig)




# m = Prophet()
#
# future = m.make_future_dataframe(periods=periods_input)

# @st.experimental_memo(ttl=60 * 60 * 24)
# def get_chart(data):
#     hover = alt.selection_single(
#         fields=["date"],
#         nearest=True,
#         on="mouseover",
#         empty="none",
#     )
#
#     lines = (
#         alt.Chart(data, height=500, title="Evolution of stock prices")
#         .mark_line()
#         .encode(
#             x=alt.X("date", title="Date"),
#             y=alt.Y("price", title="Price"),
#             color="symbol",
#         )
#     )
#
#     # Draw points on the line, and highlight based on selection
#     points = lines.transform_filter(hover).mark_circle(size=65)
#
#     # Draw a rule at the location of the selection
#     tooltips = (
#         alt.Chart(data)
#         .mark_rule()
#         .encode(
#             x="yearmonthdate(date)",
#             y="price",
#             opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
#             tooltip=[
#                 alt.Tooltip("date", title="Date"),
#                 alt.Tooltip("price", title="Price (USD)"),
#             ],
#         )
#         .add_selection(hover)
#     )
#
#     return (lines + points + tooltips).interactive()
#
#
# st.title("⬇ Time series annotations")
#
# st.write("Give more context to your time series using annotations!")
#
# col1, col2, col3 = st.columns(3)
# with col1:
#     ticker = st.text_input("Choose a ticker (⬇💬👇ℹ️ ...)", value="⬇")
# with col2:
#     ticker_dx = st.slider(
#         "Horizontal offset", min_value=-30, max_value=30, step=1, value=0
#     )
# with col3:
#     ticker_dy = st.slider(
#         "Vertical offset", min_value=-30, max_value=30, step=1, value=-10
#     )
#
# # Original time series chart. Omitted `get_chart` for clarity
# source = get_data()
# chart = get_chart(source)
#
# # Input annotations
# ANNOTATIONS = [
#     ("Mar 01, 2008", "Pretty good day for GOOG"),
#     ("Dec 01, 2007", "Something's going wrong for GOOG & AAPL"),
#     ("Nov 01, 2008", "Market starts again thanks to..."),
#     ("Dec 01, 2009", "Small crash for GOOG after..."),
# ]
#
# # Create a chart with annotations
# annotations_df = pd.DataFrame(ANNOTATIONS, columns=["date", "event"])
# annotations_df.date = pd.to_datetime(annotations_df.date)
# annotations_df["y"] = 0
# annotation_layer = (
#     alt.Chart(annotations_df)
#     .mark_text(size=15, text=ticker, dx=ticker_dx, dy=ticker_dy, align="center")
#     .encode(
#         x="date:T",
#         y=alt.Y("y:Q"),
#         tooltip=["event"],
#     )
#     .interactive()
# )
#
# # Display both charts together
# st.altair_chart((chart + annotation_layer).interactive(), use_container_width=True)
#
# st.write("## Code")
#
# st.write(
#     "See more in our public [GitHub"
#     " repository](https://github.com/streamlit/example-app-time-series-annotation)"
# )
#
# st.code(
#     f"""
# import altair as alt
# import pandas as pd
# import streamlit as st
# from vega_datasets import data
#
# @st.experimental_memo
# def get_data():
#     source = data.stocks()
#     source = source[source.date.gt("2004-01-01")]
#     return source
#
# source = get_data()
#
# # Original time series chart. Omitted `get_chart` for clarity
# chart = get_chart(source)
#
# # Input annotations
# ANNOTATIONS = [
#     ("Mar 01, 2008", "Pretty good day for GOOG"),
#     ("Dec 01, 2007", "Something's going wrong for GOOG & AAPL"),
#     ("Nov 01, 2008", "Market starts again thanks to..."),
#     ("Dec 01, 2009", "Small crash for GOOG after..."),
# ]
#
# # Create a chart with annotations
# annotations_df = pd.DataFrame(ANNOTATIONS, columns=["date", "event"])
# annotations_df.date = pd.to_datetime(annotations_df.date)
# annotations_df["y"] = 0
# annotation_layer = (
#     alt.Chart(annotations_df)
#     .mark_text(size=15, text="{ticker}", dx={ticker_dx}, dy={ticker_dy}, align="center")
#     .encode(
#         x="date:T",
#         y=alt.Y("y:Q"),
#         tooltip=["event"],
#     )
#     .interactive()
# )
#
# # Display both charts together
# st.altair_chart((chart + annotation_layer).interactive(), use_container_width=True)
#
# """,
#     "python",
# )
import plotly.express as px
import streamlit as st

from functions import *

st.set_page_config(
    page_title="Time series annotations", page_icon="⬇"
)


# @st.cache(allow_output_mutation=True)
@st.cache_data
def load_data(op_data):
    # df_despesa = pd.read_csv('https://raw.githubusercontent.com/jsaj/st_forecasting/master/datasets/despesa.csv')
    # df_receita = pd.read_csv('https://raw.githubusercontent.com/jsaj/st_forecasting/master/datasets/receita.csv')
    df = pd.read_excel(
        'https://onedrive.live.com/download?resid=71AA33284B297464%21422&authkey=!ABm-ikjLePrrS74&excel=2.xslx',
        sheet_name='{}'.format(op_data))
    return df


# Criar uma barra deslizante (slider) para selecionar qual será a previsão: receitas ou despesas
op_data = st.sidebar.selectbox('O que deseja prever?', ['Receitas', 'Despesas'])

# Carrega os dados ou utiliza os dados em cache
df = load_data(op_data)

df_filtrado = processing_columns_values(df, op_data)

# df_receita_despesa = df_receita.merge(df_despesa, how='left', on=['ds', 'Unidade Gestora']).fillna(0)
# df_receita_despesa['ds'] = pd.to_datetime(df_receita_despesa['ds'])

if op_data == 'Despesas':
    # Sidebar com o filtro
    list_to_filter = ['TODOS'] + list(df['NATUREZA'].drop_duplicates())
    filtro_type_data = st.sidebar.selectbox('Elemento: ',
                                            list_to_filter,
                                            index=list_to_filter.index(
                                                'TODOS'))
else:
    list_to_filter = ['TODAS'] + list(df['ESPÉCIE DA RECEITA'].drop_duplicates())
    filtro_type_data = st.sidebar.selectbox('Espécie da Receita:', list_to_filter,
                                            index=list_to_filter.index('TODAS'))

df_filtrado = processing_data(df, op_data, filtro_type_data)
st.write(df.head())
st.write(df_filtrado.head())

type_periods = st.sidebar.selectbox('Qual o intervalo da previsão? ', ['Mensal', 'Semestral'])
if type_periods == 'Mensal':
    # Sidebar com o filtro dos meses de previsão
    n_periods = st.sidebar.selectbox('Quantos meses?', list(range(1, 13)))
else:
    # Sidebar com o filtro dos semestres de previsão
    n_periods = st.sidebar.selectbox('Quantos semestres? ', list(range(1, 13)))

# Renomear as colunas para que o modelo possa reconhecê-las
df_filtrado.columns = ['ds', 'y']

# Criar uma barra deslizante (slider) para selecionar a variável exógerna
model_name = st.sidebar.selectbox('Modelo preditivo:', ['ARIMA', 'Prophet'])

if filtro_type_data in ['TODAS', 'TODOS']:
    # Criar uma barra deslizante (slider) para selecionar a variável exógerna
    op_exog = st.sidebar.selectbox('Usar variável exógena?:', ['Sim', 'Não'], index=list(['Sim', 'Não']).index('Não'))

    if op_exog == 'Sim':
        # Criar uma barra deslizante (slider) para selecionar a variável exógerna
        if op_data == 'Receitas':
            exog_var = st.sidebar.selectbox('Variável exógena:', list(df['ESPÉCIE DA RECEITA'].drop_duplicates()))
        else:
            exog_var = st.sidebar.selectbox('Variável exógena:', list(df['NATUREZA'].drop_duplicates()))


        df_to_predict = create_exog_table(df, df_filtrado, op_data, exog_var)
        # Criar uma barra deslizante (slider) para selecionar a porcentagem
        porcentagem = st.sidebar.slider('% vs. Var. Exógena:', min_value=0, max_value=100, value=100, step=1)

        # Aplicar a função ao DataFrame para criar uma nova coluna com os valores multiplicados
        df_to_predict[exog_var] = df_to_predict[exog_var] * (porcentagem / 100)

        st.write(df_to_predict)
        # Criar o modelo de previsão
        if model_name == 'ARIMA':
            predictions = predict_ARIMA(df=df_to_predict, n_periods=n_periods, type_periods=type_periods, exog_var=exog_var)
            df_to_predict = df_to_predict.reset_index()
        elif model_name == 'Prophet':
            predictions = predict_prophet(df=df_to_predict, n_periods=n_periods, type_periods=type_periods, exog_var=exog_var)

    else:

        # Criar o modelo de previsão
        if model_name == 'ARIMA':
            predictions = predict_ARIMA(df=df_filtrado, n_periods=n_periods, type_periods=type_periods, exog_var=None)
            df_filtrado = df_filtrado.reset_index()
        elif model_name == 'Prophet':
            predictions = predict_prophet(df=df_filtrado, n_periods=n_periods, type_periods=type_periods, exog_var=None)
else:
    # Criar o modelo de previsão
    if model_name == 'ARIMA':
        predictions = predict_ARIMA(df=df_filtrado, n_periods=n_periods, type_periods=type_periods, exog_var=None)
        df_filtrado = df_filtrado.reset_index()
    elif model_name == 'Prophet':
        predictions = predict_prophet(df=df_filtrado, n_periods=n_periods, type_periods=type_periods, exog_var=None)

# st.write(df_filtrado)

# Converter valores para milhões (M) ou milhares (K)

def format_value(value):
    if abs(value) >= 1e6:
        return '{:.2f}M'.format(value / 1e6)
    elif abs(value) >= 1e3:
        return '{:.2f}K'.format(value / 1e3)
    else:
        return '{:.2f}'.format(value)

# Criar o gráfico de linhas usando Plotly Express
fig = px.line(df_filtrado, x='ds', y='y', text=[format_value(val) for val in df_filtrado['y']],
              labels={'y': '{} atuais'.format(op_data)},
              title='{} atuais vs. {} preditas'.format(op_data, op_data))

# Adicionar a série de previsão de receita
fig.add_scatter(x=predictions['ds'], y=predictions['yhat'], mode='lines+text', text=[format_value(val) for val in predictions['yhat']],
                name='{} preditas'.format(op_data))

# Personalizar layout do gráfico
fig.update_traces(textposition='top center')
fig.update_layout(xaxis_title='Mês-Ano', yaxis_title='{}'.format(op_data), showlegend=True)

# Exibir o gráfico usando Streamlit
st.plotly_chart(fig)

# Calcular a média da previsão
mean_prediction = predictions['yhat'].mean()

df_filtrado = df_filtrado.loc[df_filtrado['ds'] >= '2023-01-01']
# Criar o gráfico de barras usando Plotly Express
fig = px.bar(df_filtrado, x='ds', y='y', text=[format_value(val) for val in df_filtrado['y']],
             labels={'y': '{} atuais'.format(op_data)},
             title='{} atuais vs. {} preditas - Média de previsão: {}'.format(op_data, op_data, format_value(mean_prediction)))

# Adicionar a série de previsão de receita
fig.add_bar(x=predictions['ds'], y=predictions['yhat'], text=[format_value(val) for val in predictions['yhat']],
            name='{} preditas'.format(op_data))

# Personalizar layout do gráfico
fig.update_traces(textposition='outside')
fig.update_layout(xaxis_title='Mês-Ano', yaxis_title='{}'.format(op_data), showlegend=True)

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

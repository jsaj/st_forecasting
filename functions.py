import re
from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from unidecode import unidecode


# Função para remover acentos usando a biblioteca unidecode
def remove_accent(text):
    return unidecode(text)


# Função para remover caracteres especiais usando expressões regulares
def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)


# Função para simplificar o nome em caso de repetição de letras
def simplify_name(name):
    for i in range(1, len(name)):
        if name[i] == name[i - 1]:
            return name[:i]
    return name


# Função que aplica a simplificação apenas para as linhas com "repeat_count" maior que 1
def replace_value_by_id(df, id_col, id_name):
    return df.groupby(id_col)[id_name].transform(
        lambda x: simplify_name(x.iloc[0]) if x.iloc[0].count(r"(.)\1+") > 1 else x.iloc[0])


# def read_receita():
#     meses = [
#         "JANEIRO",
#         "FEVEREIRO",
#         "MARCO",
#         "ABRIL",
#         "MAIO",
#         "JUNHO",
#         "JULHO",
#         "AGOSTO",
#         "SETEMBRO",
#         "OUTUBRO",
#         "NOVEMBRO",
#         "DEZEMBRO"
#     ]
#
#     anos = [2021, 2022, 2023]
#
#     df_receita = []
#
#     for ano in anos:
#         count = 1
#         for mes in meses:
#             if ano == 2023 and mes not in ["AGOSTO", "SETEMBRO", "OUTUBRO", "NOVEMBRO", "DEZEMBRO"]:
#                 file_path = 'C:\\Users\\Jjr_a\\OneDrive\\Documentos\\Estudo\\Forecasting\\Despesa_receita\\RECEITA_ATE_{}_{}.csv'.format(
#                     mes, ano)
#                 df = pd.read_csv(file_path, encoding='ISO-8859-1', delimiter=';')
#
#                 if count < 10:
#                     df['ds'] = '{}-0{}-01'.format(ano, count)
#                 else:
#                     df['ds'] = '{}-{}-01'.format(ano, count)
#             elif ano != 2023:
#                 file_path = 'C:\\Users\\Jjr_a\\OneDrive\\Documentos\\Estudo\\Forecasting\\Despesa_receita\\RECEITA_ATE_{}_{}.csv'.format(
#                     mes, ano)
#                 df = pd.read_csv(file_path, encoding='ISO-8859-1', delimiter=';')
#
#                 if count < 10:
#                     df['ds'] = '{}-0{}-01'.format(ano, count)
#                 else:
#                     df['ds'] = '{}-{}-01'.format(ano, count)
#             # Criar o índice de datas com todos os dias do mês específico
#
#             df['ds'] = pd.to_datetime(df['ds'])
#             if df['Vlr. Receita Prevista'].dtypes == object:
#                 df['Vlr. Receita Prevista'] = df['Vlr. Receita Prevista'].str.replace(',', '.').astype(float)
#             if df['Vlr. Receita Realizada'].dtypes == object:
#                 df['Vlr. Receita Realizada'] = df['Vlr. Receita Realizada'].str.replace(',', '.').astype(float)
#             if df['% Realizado'].dtypes == object:
#                 df['% Realizado'] = df['% Realizado'].str.replace(',', '.').astype(float)
#
#             df_receita.append(df)
#             count += 1
#         count = 0
#     df_receita = pd.concat(df_receita)
#     df_receita = df_receita.loc[df_receita['ds'] <= '2023-08-01']
#     df_receita = df_receita.sort_values(by=['ds', 'Cód. Unidade Gestora'])
#     df_receita = df_receita.drop_duplicates().reset_index(drop=True)
#     for column in df_receita.columns:
#         if df_receita[column].dtypes == object:
#             df_receita[column] = df_receita[column].apply(remove_accent)
#             df_receita[column] = df_receita[column].str.upper()
#             df_receita[column] = df_receita[column].str.strip()
#
#     return df_receita
#
# def read_despesa():
#     meses = [
#         "JANEIRO",
#         "FEVEREIRO",
#         "MARCO",
#         "ABRIL",
#         "MAIO",
#         "JUNHO",
#         "JULHO",
#         "AGOSTO",
#         "SETEMBRO",
#         "OUTUBRO",
#         "NOVEMBRO",
#         "DEZEMBRO"
#     ]
#
#     anos = [2021, 2022, 2023]
#
#     df_despesa = []
#
#
#     for ano in anos:
#         count = 1
#         for mes in meses:
#             if ano == 2023 and mes not in ["AGOSTO", "SETEMBRO", "OUTUBRO", "NOVEMBRO", "DEZEMBRO"]:
#
#                 file_path = 'C:\\Users\\Jjr_a\\OneDrive\\Documentos\\Estudo\\Forecasting\\Despesa_receita\\DESPESA_PAGA_ATE_{}_{}.csv'.format(
#                     mes, ano)
#                 df = pd.read_csv(file_path, encoding='ISO-8859-1', delimiter=';')
#
#                 if count < 10:
#                     df['ds'] = '{}-0{}-01'.format(ano, count)
#                 else:
#                     df['ds'] = '{}-{}-01'.format(ano, count)
#             elif ano != 2023:
#                 file_path = 'C:\\Users\\Jjr_a\\OneDrive\\Documentos\\Estudo\\Forecasting\\Despesa_receita\\DESPESA_PAGA_ATE_{}_{}.csv'.format(
#                     mes, ano)
#                 df = pd.read_csv(file_path, encoding='ISO-8859-1', delimiter=';')
#
#                 if count < 10:
#                     df['ds'] = '{}-0{}-01'.format(ano, count)
#                 else:
#                     df['ds'] = '{}-{}-01'.format(ano, count)
#             # Criar o índice de datas com todos os dias do mês específico
#
#             df['ds'] = pd.to_datetime(df['ds'])
#             if df['Valor Empenhado'].dtypes == object:
#                 df['Valor Empenhado'] = df['Valor Empenhado'].str.replace(',', '.').astype(float)
#             if df['Valor Liquidado'].dtypes == object:
#                 df['Valor Liquidado'] = df['Valor Liquidado'].str.replace(',', '.').astype(float)
#             if df['Valor Pago'].dtypes == object:
#                 df['Valor Pago'] = df['Valor Pago'].str.replace(',', '.').astype(float)
#
#             df_despesa.append(df)
#             count += 1
#         count = 0
#     df_despesa = pd.concat(df_despesa)
#     df_despesa = df_despesa.loc[df_despesa['ds'] <= '2023-08-01']
#
#     df_despesa = df_despesa.rename(columns={'Órgão': 'Unidade Gestora'}).groupby(['Unidade Gestora', 'ds']).agg({'Valor Empenhado': 'sum',
#                                                                                                                  'Valor Liquidado': 'sum',
#                                                                                                                  'Valor Pago': 'sum'}).reset_index()
#     df_despesa = df_despesa.sort_values(by=['ds', 'Unidade Gestora'])
#     df_despesa = df_despesa.drop_duplicates().reset_index(drop=True)
#
#     for column in df_despesa.columns:
#         if df_despesa[column].dtypes == object:
#             df_despesa[column] = df_despesa[column].apply(remove_accent)
#             df_despesa[column] = df_despesa[column].str.upper()
#             df_despesa[column] = df_despesa[column].str.strip()
#     return df_despesa


def get_date_range(start_date, end_date):
    start_date = datetime.strptime(start_date, '%Y-%m-%d').replace(day=1)
    end_date = datetime.strptime(end_date, '%Y-%m-%d').replace(day=1)

    current_date = start_date
    dates_list = []

    while current_date <= end_date:
        dates_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += relativedelta(months=1)

    return dates_list


def create_test_data(train, n_periods, exog_var):
    start_date = datetime.now().replace(day=1) + relativedelta(months=1)
    end_date = start_date + relativedelta(months=n_periods - 1)

    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')

    test = pd.DataFrame(get_date_range(start_date, end_date), columns=['ds'])
    test['ds'] = pd.to_datetime(test['ds'])
    if exog_var != None:
        # Adicionando a coluna 'y' em test com os valores médios dos últimos anos
        test[exog_var] = test['ds'].apply(lambda date: train[train['ds'].dt.year == date.year - 1][exog_var].mean())

    return test


def predict_ARIMA(df, n_periods, type_periods, exog_var=None):
    # # Obter o mês atual em formato de número (de 1 a 12)

    # Separação dos dados em conjuntos de treinamento e teste
    # train = df.loc[df['ds'] < f'2023-0{mes_atual - n_periods}-01']
    # test = df.loc[df['ds'] >= f'2023-0{mes_atual - n_periods}-01']
    train = df
    if type_periods == 'Mensal':
        test = create_test_data(train, n_periods, exog_var)
    else:
        n_periods = n_periods*6
        test = create_test_data(train, n_periods, exog_var)
    # Obter a data atual no formato 'ano-mes-dia', onde o dia é 01

    # Configura a coluna "ds" como índice do conjunto de treinamento
    train.set_index('ds', inplace=True)
    test.set_index('ds', inplace=True)

    if exog_var != None:
        # Treinamento do modelo SARIMAX com a variável exógena
        model = SARIMAX(train["y"], order=(5, 1, 1), exog=train[exog_var])
    else:
        model = SARIMAX(train["y"], order=(5, 1, 1))

    model_fit = model.fit()

    # Previsão para os próximos dois períodos (2 valores no total) com a variável exógena
    if exog_var is not None:
        forecast = model_fit.get_forecast(steps=n_periods, exog=test[exog_var])
    else:
        forecast = model_fit.get_forecast(steps=n_periods)

    predicted_values = forecast.predicted_mean
    predictions = []
    for i, date in enumerate(predicted_values.index):
        predictions.append([date, predicted_values.iloc[i]])

    predictions = pd.DataFrame(predictions, columns=['ds', 'yhat'])
    # predictions = test.merge(predictions, how='left', on=['ds'])
    # print(train.reset_index().rename(columns={'y': 'yhat'})[predictions.columns].tail(1))
    predictions = pd.concat(
        [train.reset_index().rename(columns={'y': 'yhat'})[predictions.columns].tail(1), predictions])
    return predictions


def predict_prophet(df, n_periods, type_periods, exog_var=None):
    # Criar o modelo de previsão
    model = Prophet()

    if exog_var != None:
        # Adicionar a variável exógena ao modelo
        model.add_regressor(exog_var)

    # Treinar o modelo
    train = df
    if type_periods == 'Mensal':
        test = create_test_data(train, n_periods, exog_var)
    else:
        test = create_test_data(train, n_periods*6, exog_var)

    model.fit(train)

    predictions = model.predict(test)

    predictions = predictions[['ds', 'yhat']]
    predictions = pd.concat(
        [train.rename(columns={'y': 'yhat'})[predictions.columns].tail(1), predictions])
    return predictions


# def predict_LSTM(df, n_periods, exog_var=None):
#
#     X = train["ds"].values.astype(np.int64) // 10 ** 9  # Convert datetime to timestamp in seconds
#     X = X.reshape(-1, 1)  # Reshape to fit LSTM input requirements
#     y = train["y"].values
#
#     # Create the neural network
#     model = Sequential()
#     model.add(
#         LSTM(10, input_shape=(1, 1), activation="relu"))  # Input shape needs to be (batch_size, timesteps, input_dim)
#     model.add(Dense(1, activation="linear"))
#
#     # Compile the neural network
#     model.compile(optimizer="adam", loss="mse")
#
#     # Train the neural network
#     model.fit(X, y, epochs=100)
#
#     # Predict the next two months of data
#     next_two_months = pd.date_range(start="2023-07-01", end="2023-09-01", freq="M").values.astype(np.int64) // 10 ** 9
#     next_two_months = next_two_months.reshape(-1, 1)
#     predictions = model.predict(next_two_months)

def processing_columns_values(df, op_data):
    for col in df.columns:
        if df[col].dtypes == object:
            df[col] = df[col].apply(remove_accent)
            df[col] = df[col].apply(remove_special_characters)
            df[col] = df[col].str.upper()
    if op_data == 'Receitas':
        df['ESPÉCIE DA RECEITA'] = df['ESPÉCIE DA RECEITA'].str.replace('OUTRAS TRANSFERENCIAS CORRENTES', 'OTC')
        df['ESPÉCIE DA RECEITA'] = np.where(
            ~df['ESPÉCIE DA RECEITA'].isin(['FPM', 'OTC', 'ISS', 'FUNDEB', 'ICMS', 'CONTRIBUICOES SOCIAIS']),
            'DEMAIS RECEITAS',
            df['ESPÉCIE DA RECEITA'])
    else:
        df['NATUREZA'] = np.where(~df['NATUREZA'].isin(['OUTRAS DESPESAS CORRENTES',
                                                        'PESSOAL E ENCARGOS SOCIAIS']),
                                  'DEMAIS DESPESAS',
                                  df['NATUREZA'])
    return df

def processing_data(df, op_data, filtro_type_data):
    if op_data == 'Receitas':
        if filtro_type_data != 'TODAS':
            df = df.loc[(df['ESPÉCIE DA RECEITA'] == filtro_type_data)
                        & (df['VALOR DA RECEITA'] > 0)].groupby(['MÊS/ANO'])[
                'VALOR DA RECEITA'].sum().reset_index()
        else:
            df = df.groupby(['MÊS/ANO'])['VALOR DA RECEITA'].sum().reset_index()

    else:

        if filtro_type_data != 'TODOS':
            df = df.loc[df['NATUREZA'] == filtro_type_data].groupby(['MÊS/ANO'])[
                'VALOR DA DESPESA'].sum().reset_index()
        else:
            df = df.groupby(['MÊS/ANO'])['VALOR DA DESPESA'].sum().reset_index()
    return df

def create_exog_table(df, df_filtrado, op_data, exog_var):
    if op_data == 'Receitas':
        df_exog = df.loc[df['ESPÉCIE DA RECEITA'] == exog_var]. groupby(['MÊS/ANO'])['VALOR DA RECEITA'].sum().reset_index()

        df_exog.columns = ['ds', 'y']
        df_exog = df_exog.merge(df_filtrado, how='right', on=['ds'])
        df_exog.columns = ['ds',  exog_var, 'y']
    else:
        df_exog = df.loc[df['NATUREZA'] == exog_var].groupby(['MÊS/ANO'])['VALOR DA DESPESA'].sum().reset_index()
        df_exog.columns = ['ds', 'y']
        df_exog = df_exog.merge(df_filtrado, how='right', on=['ds'])
        df_exog.columns = ['ds', exog_var, 'y']

    return df_exog
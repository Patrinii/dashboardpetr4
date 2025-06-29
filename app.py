import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Dashboard PETR4", layout="wide")

@st.cache_data
def carregar_dados():
    df = yf.download("PETR4.SA", start="2023-01-01", end="2025-12-31")
    df = df.reset_index()
    df['Ano'] = df['Date'].dt.year
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df

def preparar_dados(df):
    df['Retorno'] = df['Close'].pct_change()
    df = df.dropna()
    return df

def treinar_modelo(treino, teste):
    colunas = ['Open', 'High', 'Low', 'Close', 'Volume', 'Retorno']
    X_treino = treino[colunas]
    y_treino = treino['Target']
    X_teste = teste[colunas]
    y_teste = teste['Target']

    scaler = StandardScaler()
    X_treino_scaled = scaler.fit_transform(X_treino)
    X_teste_scaled = scaler.transform(X_teste)

    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_treino_scaled, y_treino)

    teste_completo = teste.copy()
    teste_completo['X_scaled'] = list(X_teste_scaled)

    return modelo, X_teste_scaled, y_teste, teste_completo

def avaliar_modelo(modelo, X_teste_scaled, y_teste):
    return modelo.predict(X_teste_scaled)

def simular_retorno(teste_completo, y_pred):
    teste_completo['Predito'] = y_pred
    teste_completo['Retorno'] = teste_completo['Close'].pct_change()
    teste_completo['Lucro'] = teste_completo['Retorno'] * (teste_completo['Predito'].shift(1))
    teste_completo = teste_completo.dropna()
    return teste_completo

# Sidebar
st.sidebar.title("Configurações")
ano_treino = st.sidebar.selectbox("Ano de Treinamento:", [2023, 2024, "2023 e 2024"])
graficos = st.sidebar.multiselect("Gráficos", ["Fechamento", "Distribuição de Alvo", "Lucro Acumulado", "Volume x Preço"])

# Carregamento e preparo dos dados
dados = carregar_dados()
dados = dados.dropna()
dados = preparar_dados(dados)

# Separar treino e teste
if ano_treino == "2023 e 2024":
    treino = dados[dados['Ano'].isin([2023, 2024])]
else:
    treino = dados[dados['Ano'] == ano_treino]

teste = dados[dados['Ano'] == 2025]

# Treinar e avaliar modelo
modelo, X_teste, y_teste, teste_completo = treinar_modelo(treino, teste)
y_pred = avaliar_modelo(modelo, X_teste, y_teste)
teste_completo = simular_retorno(teste_completo, y_pred)

# Título
st.title("📊 Dashboard Financeiro - PETR4.SA")

# Métricas principais
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Acurácia", f"{modelo.score(X_teste, y_teste)*100:.2f}%")
with col2:
    st.metric("Retorno Total", f"R$ {teste_completo['Lucro'].sum():.2f}")
with col3:
    st.metric("Operações com Lucro", f"{(teste_completo['Lucro'] > 0).sum()}")

# Gráficos
if "Fechamento" in graficos:
    st.subheader("📉 Gráfico de Fechamento")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dados['Date'], dados['Close'], color='blue')
    ax.set_xlabel("Data")
    ax.set_ylabel("Preço de Fechamento")
    st.pyplot(fig)

if "Distribuição de Alvo" in graficos:
    st.subheader("🎯 Distribuição da Variável Alvo")
    fig, ax = plt.subplots()
    sns.countplot(x='Target', data=dados, ax=ax)
    st.pyplot(fig)

if "Lucro Acumulado" in graficos:
    st.subheader("💸 Lucro Acumulado")
    dados_plot = teste_completo.copy()
    dados_plot['Lucro_Acumulado'] = dados_plot['Lucro'].cumsum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dados_plot['Date'], dados_plot['Lucro_Acumulado'], color='green')
    ax.set_xlabel("Data")
    ax.set_ylabel("Lucro Acumulado")
    st.pyplot(fig)

if "Volume x Preço" in graficos:
    st.subheader("📊 Dispersão Volume x Preço")
    fig, ax = plt.subplots()
    ax.scatter(dados['Volume'], dados['Close'], alpha=0.5)
    ax.set_xlabel("Volume")
    ax.set_ylabel("Preço")
    st.pyplot(fig)

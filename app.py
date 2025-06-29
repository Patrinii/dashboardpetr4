
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

@st.cache_data
def carregar_dados():
    dados = pd.read_csv("petr4.csv")
    st.write("Colunas carregadas:", dados.columns.tolist())  # Linha útil para debug
    return dados

# Função para preparar os dados
def preparar_dados(dados):
    dados['Date'] = pd.to_datetime(dados['Date'])
    dados['Ano'] = dados['Date'].dt.year
    dados['Target'] = dados['Close'].shift(-1) > dados['Close']
    dados['Target'] = dados['Target'].astype(int)
    dados.dropna(inplace=True)
    return dados

# Função para treinar modelo simples (exemplo)
def treinar_modelo(df):
    from sklearn.ensemble import RandomForestClassifier
    X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = df['Target']
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X, y)
    return modelo, X, y

# Função para simular retorno financeiro
def simular_retorno(teste, y_pred):
    teste = teste.copy()
    teste['Predito'] = y_pred
    teste['Lucro'] = np.where(teste['Predito'] == 1, teste['Close'].shift(-1) - teste['Close'], 0)
    teste['Lucro'] = teste['Lucro'].fillna(0)
    return teste

# Layout
st.set_page_config(page_title="Dashboard PETR4", layout="wide")
st.title("📊 Dashboard de Análise PETR4")

# Carregando e preparando os dados
dados = carregar_dados()
dados = preparar_dados(dados)

# Filtro de ano
anos = st.sidebar.multiselect("Selecione os anos para Treinamento", options=[2023, 2024], default=[2023, 2024])
dados_treino = dados[dados['Ano'].isin(anos)]
dados_teste = dados[dados['Ano'] == 2025]

# Modelo
modelo, X_treino, y_treino = treinar_modelo(dados_treino)
X_teste = dados_teste[['Open', 'High', 'Low', 'Close', 'Volume']]
y_teste = dados_teste['Target']
y_pred = modelo.predict(X_teste)
teste_completo = simular_retorno(dados_teste, y_pred)

# Métricas
st.subheader("📈 Métricas de Avaliação")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Acurácia", f"{accuracy_score(y_teste, y_pred)*100:.2f}%")
col2.metric("Precisão", f"{precision_score(y_teste, y_pred)*100:.2f}%")
col3.metric("Recall (Sens.)", f"{recall_score(y_teste, y_pred)*100:.2f}%")
col4.metric("F1-Score", f"{f1_score(y_teste, y_pred)*100:.2f}%")

# Especificidade
cm = confusion_matrix(y_teste, y_pred)
tn, fp, fn, tp = cm.ravel()
especificidade = tn / (tn + fp)
st.metric("Especificidade", f"{especificidade*100:.2f}%")

# Retornos financeiros
lucros = teste_completo['Lucro']
ganhos = lucros[lucros > 0].sum()
perdas = -lucros[lucros < 0].sum()
retorno_total = lucros.sum()

st.subheader("💰 Retorno Financeiro")
col1, col2, col3 = st.columns(3)
col1.metric("Retorno de Ganhos", f"R$ {ganhos:.2f}")
col2.metric("Retorno de Perdas", f"R$ {perdas:.2f}")
col3.metric("Retorno Total", f"R$ {retorno_total:.2f}")

# Gráficos
st.subheader("📊 Gráficos Interativos")
aba = st.selectbox("Selecione o gráfico", [
    "Gráfico de Fechamento", 
    "Distribuição do Alvo", 
    "Lucro Acumulado", 
    "Volume x Preço", 
    "Série Temporal Completa"
])

if aba == "Gráfico de Fechamento":
    st.line_chart(dados.set_index('Date')['Close'])

elif aba == "Distribuição do Alvo":
    counts = dados['Target'].value_counts(normalize=True)*100
    fig, ax = plt.subplots()
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_title("Distribuição do Alvo (%)")
    st.pyplot(fig)
    st.write("**Percentuais:**")
    st.write(counts.round(2).astype(str) + "%")

elif aba == "Lucro Acumulado":
    teste_completo['Lucro_Acumulado'] = teste_completo['Lucro'].cumsum()
    st.line_chart(teste_completo.set_index('Date')['Lucro_Acumulado'])

elif aba == "Volume x Preço":
    fig, ax = plt.subplots()
    ax.scatter(dados['Volume'], dados['Close'], alpha=0.3)
    ax.set_xlabel("Volume")
    ax.set_ylabel("Preço de Fechamento")
    st.pyplot(fig)

elif aba == "Série Temporal Completa":
    st.line_chart(dados.set_index('Date')[['Open', 'High', 'Low', 'Close']])

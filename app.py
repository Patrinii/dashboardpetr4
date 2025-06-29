
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Carregando os dados
@st.cache_data
def carregar_dados():
    df = pd.read_csv("petr4.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Ano'] = df['Date'].dt.year
    df = df.dropna()
    return df

# Preparando os dados
def preparar_dados(df):
    df['Target'] = df['Target'].astype(int)
    return df

# Simulando um modelo (substitua pela sua função real)
def treinar_e_avaliar(df):
    treino = df[df['Ano'].isin([2023, 2024])]
    teste = df[df['Ano'] == 2025]

    y_true = teste['Target']
    y_pred = np.random.choice([0, 1], size=len(y_true), p=[0.5, 0.5])  # Simulação

    teste['Predito'] = y_pred
    teste['Lucro'] = np.where(teste['Predito'] == 1, teste['Close'].diff().fillna(0), 0)

    return treino, teste, y_true, y_pred

# Layout do Streamlit
st.set_page_config(page_title="Dashboard PETR4", layout="wide")
st.title("📊 Dashboard de Análise - PETR4.SA (2023-2025)")

df = carregar_dados()
df = preparar_dados(df)

treino, teste, y_true, y_pred = treinar_e_avaliar(df)

# Gráfico de linha da série temporal (Close)
st.subheader("📈 Série Temporal de Fechamento")
fig1 = px.line(df, x="Date", y="Close", title="Fechamento das Ações (2023-2025)")
st.plotly_chart(fig1, use_container_width=True)

# Distribuição da variável alvo
st.subheader("🎯 Distribuição da Variável Alvo")
col1, col2 = st.columns(2)
with col1:
    fig2 = px.histogram(df, x="Target", title="Distribuição das Classes (Target)", text_auto=True)
    st.plotly_chart(fig2, use_container_width=True)
with col2:
    dist = df["Target"].value_counts(normalize=True).reset_index()
    dist.columns = ["Classe", "Percentual"]
    dist["Percentual"] = dist["Percentual"] * 100
    st.dataframe(dist)

# Matriz de confusão e métricas
st.subheader("📌 Avaliação do Modelo (ano 2025)")
matriz = confusion_matrix(y_true, y_pred)
acertos = np.trace(matriz)
erros = matriz.sum() - acertos

col1, col2 = st.columns(2)
with col1:
    st.metric("✔️ Acertos", acertos)
    st.metric("❌ Erros", erros)
    st.write("Matriz de Confusão:")
    st.dataframe(pd.DataFrame(matriz, columns=["Previsto 0", "Previsto 1"], index=["Real 0", "Real 1"]))
with col2:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    espec = matriz[0][0] / (matriz[0][0] + matriz[0][1]) if (matriz[0][0] + matriz[0][1]) > 0 else 0

    st.metric("🎯 Acurácia", f"{acc*100:.2f}%")
    st.metric("📌 Precisão", f"{prec*100:.2f}%")
    st.metric("🔁 Sensibilidade (Recall)", f"{rec*100:.2f}%")
    st.metric("📊 F1-Score", f"{f1*100:.2f}%")
    st.metric("🛡️ Especificidade", f"{espec*100:.2f}%")

# Retorno financeiro
st.subheader("💰 Simulação de Retorno Financeiro (2025)")
lucro_total = teste['Lucro'].sum()
lucro_pos = teste[teste['Lucro'] > 0]['Lucro'].sum()
lucro_neg = teste[teste['Lucro'] < 0]['Lucro'].sum()
total_op = len(teste)
percent_ganho = (lucro_pos / total_op) * 100
percent_perda = (lucro_neg / total_op) * 100
percent_final = ((lucro_total) / total_op) * 100

st.write(f"**Retorno Total:** R$ {lucro_total:.2f}")
st.write(f"**Retorno médio por operação com ganho:** {percent_ganho:.2f}%")
st.write(f"**Retorno médio por operação com perda:** {percent_perda:.2f}%")
st.write(f"**Retorno médio final por operação:** {percent_final:.2f}%")

# Lucro acumulado
st.subheader("📈 Lucro Acumulado")
teste['Lucro_Acumulado'] = teste['Lucro'].cumsum()
fig3 = px.line(teste, x="Date", y="Lucro_Acumulado", title="Lucro Acumulado (2025)", markers=True)
st.plotly_chart(fig3, use_container_width=True)

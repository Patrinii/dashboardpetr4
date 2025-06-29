# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Carregar dados
df = pd.read_csv("PETR4_SA_2023_2025.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Ano'] = df['Date'].dt.year
df['Target'] = (df['Close'] > df['Open']).astype(int)

# Sidebar
st.sidebar.title("Filtros")
anos_treino = st.sidebar.multiselect("Selecione o(s) ano(s) de treinamento:", [2023, 2024], default=[2023, 2024])

# Separar treino e teste
treino = df[df['Ano'].isin(anos_treino)]
teste = df[df['Ano'] == 2025]

# Modelo KNN
features = ['Open', 'High', 'Low', 'Close', 'Volume']
scaler = StandardScaler()
X_treino = scaler.fit_transform(treino[features])
y_treino = treino['Target']
X_teste = scaler.transform(teste[features])
y_teste = teste['Target']

modelo = KNeighborsClassifier(n_neighbors=5)
modelo.fit(X_treino, y_treino)
y_pred = modelo.predict(X_teste)

# Métricas
acc = accuracy_score(y_teste, y_pred)
prec = precision_score(y_teste, y_pred, zero_division=0)
rec = recall_score(y_teste, y_pred, zero_division=0)
f1 = f1_score(y_teste, y_pred, zero_division=0)
matriz = confusion_matrix(y_teste, y_pred)
tn, fp, fn, tp = matriz.ravel()
especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0

# Retorno financeiro
teste = teste.copy()
teste['Previsao'] = y_pred
teste['Lucro'] = teste.apply(lambda row: row['Close'] - row['Open'] if row['Previsao'] == 1 and row['Target'] == 1 else 0, axis=1)
teste['Lucro_Acumulado'] = teste['Lucro'].cumsum()
retorno_total = teste['Lucro'].sum()
retorno_pct = (retorno_total / teste['Open'].sum()) * 100

# Página principal
st.title("📊 Dashboard de Análise - PETR4.SA")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Acurácia", f"{acc:.2%}")
col2.metric("Precisão", f"{prec:.2%}")
col3.metric("Recall", f"{rec:.2%}")
col4.metric("F1-score", f"{f1:.2%}")
st.markdown(f"**Especificidade:** {especificidade:.2%}")

# Gráfico de fechamento
fig1 = px.line(df, x='Date', y='Close', title="📈 Fechamento das Ações")
st.plotly_chart(fig1, use_container_width=True)

# Distribuição da variável alvo
fig2 = px.histogram(df, x='Target', title="🎯 Distribuição da Variável Alvo", text_auto=True)
st.plotly_chart(fig2, use_container_width=True)

# Lucro acumulado
fig3 = px.line(teste, x='Date', y='Lucro_Acumulado', title="💸 Lucro Acumulado")
st.plotly_chart(fig3, use_container_width=True)

# Volume x Preço
fig4 = px.scatter(df, x='Volume', y='Close', opacity=0.5, title="📊 Dispersão Volume x Preço")
st.plotly_chart(fig4, use_container_width=True)

# Acertos e erros
acertos = (y_teste == y_pred).sum()
erros = (y_teste != y_pred).sum()
st.markdown(f"**Acertos:** {acertos} | **Erros:** {erros}")

# Retornos
ganhos = teste[teste['Lucro'] > 0]['Lucro'].sum()
perdas = teste[teste['Lucro'] < 0]['Lucro'].sum()
qtde = len(teste)

st.markdown(f"**Retorno Total (R$):** {retorno_total:.2f} | **%:** {retorno_pct:.2f}%")
st.markdown(f"**Ganhos Totais (R$):** {ganhos:.2f}")
st.markdown(f"**Perdas Totais (R$):** {perdas:.2f}")


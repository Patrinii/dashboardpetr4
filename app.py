
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Dashboard B3 - PETR4", layout="wide")

@st.cache_data
def carregar_dados():
    df = pd.read_csv("petr4.csv")
    return df

@st.cache_data
def preparar_dados(df):
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Ano'] = df['Date'].dt.year
    df['Target'] = (df['Close'] > df['Open']).astype(int)
    return df

@st.cache_data
def treinar_modelo(treino, teste):
    colunas = ['Open', 'High', 'Low', 'Close', 'Volume']
    X_treino = treino[colunas]
    y_treino = treino['Target']
    X_teste = teste[colunas]
    y_teste = teste['Target']

    scaler = StandardScaler()
    X_treino_scaled = scaler.fit_transform(X_treino)
    X_teste_scaled = scaler.transform(X_teste)

    modelo = KNeighborsClassifier(n_neighbors=5)
    modelo.fit(X_treino_scaled, y_treino)

    return modelo, X_teste_scaled, y_teste, teste

def avaliar_modelo(modelo, X_teste, y_teste):
    y_pred = modelo.predict(X_teste)
    matriz = confusion_matrix(y_teste, y_pred)
    acuracia = accuracy_score(y_teste, y_pred)
    precisao = precision_score(y_teste, y_pred, zero_division=0)
    recall = recall_score(y_teste, y_pred, zero_division=0)
    f1 = f1_score(y_teste, y_pred, zero_division=0)
    tn, fp, fn, tp = matriz.ravel()
    especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0

    return y_pred, acuracia, precisao, recall, f1, especificidade, tp, tn, fp, fn

def simular_retorno(df, y_pred):
    df = df.copy()
    df['Predito'] = y_pred
    df['Lucro'] = df.apply(lambda linha: linha['Close'] - linha['Open'] if linha['Predito'] == 1 else 0, axis=1)
    df['Lucro'] = df.apply(lambda linha: linha['Lucro'] if linha['Target'] == linha['Predito'] else -abs(linha['Lucro']), axis=1)
    return df

# Carregar e preparar dados
dados = carregar_dados()
dados = dados.dropna()
dados = preparar_dados(dados)

# Interface
st.sidebar.title("Configurações")
ano_treino = st.sidebar.selectbox("Ano de Treinamento:", [2023, 2024, "2023 e 2024"])

if ano_treino == "2023 e 2024":
    treino = dados[dados['Ano'].isin([2023, 2024])]
else:
    treino = dados[dados['Ano'] == int(ano_treino)]

teste = dados[dados['Ano'] == 2025]

# Modelo e avaliação
modelo, X_teste, y_teste, teste_bruto = treinar_modelo(treino, teste)
y_pred, acuracia, precisao, recall, f1, especificidade, tp, tn, fp, fn = avaliar_modelo(modelo, X_teste, y_teste)
teste_completo = simular_retorno(teste_bruto, y_pred)

# Métricas financeiras
ganhos = teste_completo[teste_completo['Lucro'] > 0]['Lucro'].sum()
perdas = abs(teste_completo[teste_completo['Lucro'] < 0]['Lucro'].sum())
total = ganhos - perdas
total_absoluto = teste_completo['Lucro'].abs().sum()
percent_ganhos = (ganhos / total_absoluto * 100) if total_absoluto != 0 else 0
percent_perdas = (perdas / total_absoluto * 100) if total_absoluto != 0 else 0
percent_total = (total / total_absoluto * 100) if total_absoluto != 0 else 0

# Dashboard
st.title("📈 Painel Analítico - PETR4.SA")

col1, col2, col3 = st.columns(3)
col1.metric("Acurácia", f"{acuracia:.2%}")
col2.metric("Retorno Total (R$)", f"{total:.2f}")
col3.metric("F1-Score", f"{f1:.2%}")

st.subheader("📉 Gráfico de Fechamento")
fig1, ax1 = plt.subplots()
ax1.plot(dados['Date'], dados['Close'])
st.pyplot(fig1)

st.subheader("🎯 Distribuição da Variável Alvo")
fig2, ax2 = plt.subplots()
sns.countplot(x='Target', data=dados, ax=ax2)
st.pyplot(fig2)

st.subheader("📊 Volume x Preço")
fig3, ax3 = plt.subplots()
ax3.scatter(dados['Volume'], dados['Close'], alpha=0.5)
st.pyplot(fig3)

st.subheader("💸 Lucro Acumulado")
teste_completo['Lucro_Acumulado'] = teste_completo['Lucro'].cumsum()
fig4, ax4 = plt.subplots()
ax4.plot(teste_completo['Date'], teste_completo['Lucro_Acumulado'])
st.pyplot(fig4)

st.subheader("📊 Resultados da Classificação e Retorno Financeiro")
st.write(f"Acertos (TP + TN): {tp + tn} - {((tp + tn)/len(y_pred)):.2%}")
st.write(f"Erros (FP + FN): {fp + fn} - {((fp + fn)/len(y_pred)):.2%}")
st.write(f"Precisão: {precisao:.2%}")
st.write(f"Recall (Sensibilidade): {recall:.2%}")
st.write(f"Especificidade: {especificidade:.2%}")
st.write(f"Retorno Total: R$ {total:.2f} ({percent_total:.2f}%)")
st.write(f"Ganhos: R$ {ganhos:.2f} ({percent_ganhos:.2f}%)")
st.write(f"Perdas: R$ {perdas:.2f} ({percent_perdas:.2f}%)")

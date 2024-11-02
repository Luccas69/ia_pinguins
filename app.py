import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# Função para carregar e preparar o dataset
@st.cache_data
def carregar_dados():
    penguins = load_penguins()
    penguins = penguins.dropna()  # Remover linhas com dados ausentes
    return penguins

# Função para treinar o modelo
@st.cache_data
def treinar_modelo(X_train, y_train, max_depth):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model

# Função para plotar a matriz de confusão
def plotar_matriz_confusao(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel("Predição")
    plt.ylabel("Valor Real")
    st.pyplot(plt)

# Carregar o dataset
penguins = carregar_dados()

# Título e introdução
st.title("Classificação de Pinguins com Árvores de Decisão")
st.write("Essa aplicação usa um modelo de árvore de decisão para classificar espécies de pinguins com base em características físicas.")

# Exibir os dados
st.subheader("Dados dos Pinguins")
if st.checkbox("Mostrar dados"):
    st.write(penguins)

# Visualização dos dados
st.subheader("Distribuição das Características")
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(penguins, x="bill_length_mm", hue="species", kde=True, ax=axs[0])
sns.histplot(penguins, x="flipper_length_mm", hue="species", kde=True, ax=axs[1])
st.pyplot(fig)

# Configurações para a previsão
st.sidebar.header("Configurações de Previsão")
bill_length = st.sidebar.slider("Comprimento do Bico (mm)", float(penguins['bill_length_mm'].min()), float(penguins['bill_length_mm'].max()), float(penguins['bill_length_mm'].mean()))
bill_depth = st.sidebar.slider("Profundidade do Bico (mm)", float(penguins['bill_depth_mm'].min()), float(penguins['bill_depth_mm'].max()), float(penguins['bill_depth_mm'].mean()))
flipper_length = st.sidebar.slider("Comprimento da Nadadeira (mm)", float(penguins['flipper_length_mm'].min()), float(penguins['flipper_length_mm'].max()), float(penguins['flipper_length_mm'].mean()))
body_mass = st.sidebar.slider("Massa Corporal (g)", float(penguins['body_mass_g'].min()), float(penguins['body_mass_g'].max()), float(penguins['body_mass_g'].mean()))
max_depth = st.sidebar.slider("Profundidade Máxima da Árvore de Decisão", 1, 10, 3)

# Preparar os dados para treinamento da árvore de decisão
# Manter apenas as colunas que queremos para o treinamento do modelo
X = penguins[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
y = penguins['species']

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
model = treinar_modelo(X_train, y_train, max_depth)

# Fazer a previsão com os dados do usuário
input_data = [[bill_length, bill_depth, flipper_length, body_mass]]
prediction = model.predict(input_data)

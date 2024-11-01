import streamlit as st
import pandas as pd
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Carregar o dataset
penguins = load_penguins()
penguins = penguins.dropna()  # Remover linhas com dados ausentes

# Título da aplicação
st.title("Classificação de Pinguins")

# Mostrar os dados
st.subheader("Dados dos Pinguins")
st.write(penguins)

# Selecionar características para previsão
st.sidebar.header("Configurações de Previsão")
bill_length = st.sidebar.slider("Comprimento do Bico (mm)", float(penguins['bill_length_mm'].min()), float(penguins['bill_length_mm'].max()), float(penguins['bill_length_mm'].mean()))
bill_depth = st.sidebar.slider("Profundidade do Bico (mm)", float(penguins['bill_depth_mm'].min()), float(penguins['bill_depth_mm'].max()), float(penguins['bill_depth_mm'].mean()))
flipper_length = st.sidebar.slider("Comprimento da Nadadeira (mm)", float(penguins['flipper_length_mm'].min()), float(penguins['flipper_length_mm'].max()), float(penguins['flipper_length_mm'].mean()))
body_mass = st.sidebar.slider("Massa Corporal (g)", float(penguins['body_mass_g'].min()), float(penguins['body_mass_g'].max()), float(penguins['body_mass_g'].mean()))

# Preparar os dados para treinamento da árvore de decisão
X = penguins.drop(columns=['species', 'sex', 'island'])  # Atributos
y = penguins['species']  # Classe

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Fazer previsões com base nas entradas do usuário
input_data = [[bill_length, bill_depth, flipper_length, body_mass]]
prediction = model.predict(input_data)

# Exibir o resultado da previsão
st.subheader("Resultado da Previsão")
st.write(f"A espécie do pinguim prevista é: **{prediction[0]}**")

# Avaliar a precisão do modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Precisão do modelo: {accuracy:.2f}")

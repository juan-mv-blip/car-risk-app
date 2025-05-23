import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Cargar el modelo y variables
filename = 'modelo-reg-tree-knn-nn.pkl'
model_Tree, model_Knn, model_NN, variables, min_max_scaler = pickle.load(open(filename, 'rb'))

# Extraer columnas dummy desde `variables`
videojuegos_dummies = [v for v in variables if v.startswith("videojuego_")]
plataformas_dummies = [v for v in variables if v.startswith("Plataforma_")]

# Crear diccionarios: mostrar valor limpio pero guardar el nombre dummy exacto
videojuego_map = {v.replace("videojuego_", ""): v for v in videojuegos_dummies}
plataforma_map = {v.replace("Plataforma_", ""): v for v in plataformas_dummies}

# Inputs desde el usuario
st.title("Predicción de presupuesto de videojuegos")

edad = st.slider("Edad", 10, 80, 25)
sexo = st.selectbox("Sexo", ["Hombre", "Mujer"])
consumidor = st.checkbox("¿Es consumidor habitual?")

videojuego = st.selectbox("Videojuego", list(videojuego_map.keys()))
plataforma = st.selectbox("Plataforma", list(plataforma_map.keys()))

# Crear DataFrame base
input_data = pd.DataFrame(columns=variables)
input_data.loc[0] = 0  # Inicializa todo en cero

# Edad normalizada
input_data['Edad'] = min_max_scaler.transform([[edad]])[0][0]

# Activar las dummies seleccionadas
input_data[videojuego_map[videojuego]] = 1
input_data[plataforma_map[plataforma]] = 1
if sexo == "Mujer":
    input_data['Sexo_Mujer'] = 1
if consumidor:
    input_data['Consumidor_habitual_True'] = 1

# Hacer predicciones
pred_tree = model_Tree.predict(input_data)[0]
pred_knn = model_Knn.predict(input_data)[0]
pred_nn = model_NN.predict(input_data)[0]

# Mostrar resultados
st.subheader("Predicciones:")
st.write(f"Árbol de Decisión: ${pred_tree:.2f}")
st.write(f"K-NN: ${pred_knn:.2f}")
st.write(f"Red Neuronal: ${pred_nn:.2f}")




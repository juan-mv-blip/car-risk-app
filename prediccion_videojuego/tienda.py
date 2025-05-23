import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Cargar modelos entrenados y variables
with open("modelo-reg-tree-knn-nn.pkl", "rb") as f:
    model_Tree, model_Knn, model_NN, variables, min_max_scaler = pickle.load(f)

# Selección de modelo por interfaz
st.title("Predicción de Presupuesto para Videojuegos")

modelo_nombre = st.selectbox("Selecciona el modelo a usar", ["Árbol de Decisión", "KNN", "Red Neuronal"])
if modelo_nombre == "Árbol de Decisión":
    modelo = model_Tree
elif modelo_nombre == "KNN":
    modelo = model_Knn
else:
    modelo = model_NN

# Inputs del usuario
st.header("Ingresa los datos del consumidor")

edad = st.slider("Edad", min_value=14, max_value=52, value=25)

videojuegos = ['Battlefield', 'Crysis', 'Dead Space', 'F1', 'Fifa', 'KOA: Reckoning', 'Mass Effect', 'Sim City']
videojuego = st.selectbox("¿Qué videojuego le interesa?", videojuegos)

plataformas = ['PC', 'Play Station', 'Xbox', 'Otros']
plataforma = st.selectbox("Plataforma preferida", plataformas)

sexo = st.selectbox("Sexo", ['Hombre', 'Mujer'])

consumidor_habitual = st.checkbox("¿Es consumidor habitual?", value=True)

# Crear el DataFrame de entrada con columnas dummy
input_dict = {col: 0 for col in variables}
input_dict['Edad'] = edad

# Dummies de videojuego
col_videojuego = f"videojuego_{videojuego}"
if col_videojuego in input_dict:
    input_dict[col_videojuego] = 1

# Dummies de plataforma
col_plataforma = f"Plataforma_{plataforma}"
if col_plataforma in input_dict:
    input_dict[col_plataforma] = 1

# Sexo
if sexo == "Mujer":
    input_dict["Sexo_Mujer"] = 1

# Consumidor habitual
if consumidor_habitual:
    input_dict["Consumidor_habitual_True"] = 1

# Convertir a DataFrame
input_df = pd.DataFrame([input_dict])

# Normalizar la edad si el modelo lo requiere
if modelo_nombre in ["KNN", "Red Neuronal"]:
    input_df["Edad"] = min_max_scaler.transform(input_df[["Edad"]])

# Predicción
prediccion = modelo.predict(input_df)[0]

# Mostrar el resultado
st.subheader("Resultado")
st.success(f"El presupuesto estimado para invertir es: ${prediccion:,.2f}")

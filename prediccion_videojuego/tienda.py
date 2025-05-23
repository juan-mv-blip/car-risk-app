import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Título
st.title("Predicción de Presupuesto para Invertir en Videojuegos 🎮")

# Cargar modelos y variables
with open("modelo-reg-tree-knn-nn.pkl", "rb") as f:
    model_Tree, model_Knn, model_NN, variables, min_max_scaler = pickle.load(f)

# Selección del modelo
modelo_seleccionado = st.selectbox("Selecciona el modelo:", ["Árbol de Regresión", "KNN", "Red Neuronal"])

if modelo_seleccionado == "Árbol de Regresión":
    modelo = model_Tree
elif modelo_seleccionado == "KNN":
    modelo = model_Knn
else:
    modelo = model_NN

# Entrada de usuario
edad = st.slider("Edad", min_value=14, max_value=52, value=30)

videojuego = st.selectbox("Videojuego favorito:", [
    "Battlefield", "Crysis", "Dead Space", "F1", "Fifa", "KOA: Reckoning", "Mass Effect", "Sim City"
])

plataforma = st.selectbox("Plataforma preferida:", ["PC", "Xbox", "Play Station", "Otros"])

sexo = st.radio("Sexo:", ["Hombre", "Mujer"])

consumidor = st.radio("¿Es consumidor habitual?", ["Sí", "No"])

# Convertir entradas a formato de modelo
input_dict = {
    'Edad': edad,
    'videojuego_Battlefield': 0,
    'videojuego_Crysis': 0,
    'videojuego_Dead Space': 0,
    'videojuego_F1': 0,
    'videojuego_Fifa': 0,
    'videojuego_KOA: Reckoning': 0,
    'videojuego_Mass Effect': 0,
    'videojuego_Sim City': 0,
    'Plataforma_Otros': 0,
    'Plataforma_PC': 0,
    'Plataforma_Play Station': 0,
    'Plataforma_Xbox': 0,
    'Sexo_Mujer': 1 if sexo == "Mujer" else 0,
    'Consumidor_habitual_True': 1 if consumidor == "Sí" else 0
}

# Activar la categoría elegida
input_dict[f"videojuego_{videojuego}"] = 1
input_dict[f"Plataforma_{plataforma}"] = 1

# Convertir a DataFrame
input_df = pd.DataFrame([input_dict])

# Reordenar columnas
input_df = input_df.reindex(columns=variables, fill_value=0)

# Normalizar la edad si corresponde
if modelo_seleccionado in ["KNN", "Red Neuronal"]:
    input_df[['Edad']] = min_max_scaler.transform(input_df[['Edad']])

# Botón de predicción
if st.button("Predecir presupuesto estimado"):
    prediccion = modelo.predict(input_df)[0]
    st.success(f"💰 Presupuesto estimado para invertir: **{prediccion:.2f}** unidades")

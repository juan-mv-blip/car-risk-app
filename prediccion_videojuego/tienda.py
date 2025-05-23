# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar el modelo
modelo = joblib.load('modelo-reg-tree-knn-nn.pkl')

# Título de la app
st.title("Predicción de Videojuegos")

# Entradas del usuario
st.header("Ingresa tus datos:")

edad = st.slider("Edad", 14, 52, 25)

videojuegos = [
    "Battlefield", "Crysis", "Dead Space", "F1", "Fifa",
    "KOA: Reckoning", "Mass Effect", "Sim City"
]
videojuego_seleccionado = st.selectbox("Selecciona un videojuego", videojuegos)

plataforma = st.selectbox("Selecciona una plataforma", ["PC", "Play Station", "Xbox", "Otros"])

sexo = st.selectbox("Sexo", ["Hombre", "Mujer"])
consumidor_habitual = st.checkbox("¿Eres consumidor habitual?")

# Crear el vector de características (orden de columnas importante)
data = {
    'Edad': edad,
    'videojuego_Battlefield': int(videojuego_seleccionado == "Battlefield"),
    'videojuego_Crysis': int(videojuego_seleccionado == "Crysis"),
    'videojuego_Dead Space': int(videojuego_seleccionado == "Dead Space"),
    'videojuego_F1': int(videojuego_seleccionado == "F1"),
    'videojuego_Fifa': int(videojuego_seleccionado == "Fifa"),
    'videojuego_KOA: Reckoning': int(videojuego_seleccionado == "KOA: Reckoning"),
    'videojuego_Mass Effect': int(videojuego_seleccionado == "Mass Effect"),
    'videojuego_Sim City': int(videojuego_seleccionado == "Sim City"),
    'Plataforma_Otros': int(plataforma == "Otros"),
    'Plataforma_PC': int(plataforma == "PC"),
    'Plataforma_Play Station': int(plataforma == "Play Station"),
    'Plataforma_Xbox': int(plataforma == "Xbox"),
    'Sexo_Mujer': int(sexo == "Mujer"),
    'Consumidor_habitual_True': int(consumidor_habitual)
}

# Convertir a DataFrame
input_df = pd.DataFrame([data])

# Predicción
if st.button("Predecir"):
    prediccion = modelo.predict(input_df)
    st.success(f"Predicción del modelo: {prediccion[0]:.2f}")

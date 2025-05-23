import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Configuración general de la página
st.set_page_config(page_title="Predicción compra de videojuegos en tienda", page_icon="🕹️", layout="centered")

# Imagen decorativa
st.image("https://raw.githubusercontent.com/juan-mv-blip/car-risk-app/blob/main/prediccion_videojuego/assets/tienda.jpg", use_container_width=True)

# Título y descripción
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>🛒 Predicción de Compra de Videojuegos</h1>
    <p style='text-align: center; font-size: 16px;'>Estima cuánto presupuesto invertiría un usuario en videojuegos según sus características.</p>
""", unsafe_allow_html=True)

# Cargar modelos y recursos
@st.cache_resource
def cargar_modelos():
    with open("modelo-reg-tree-knn-nn.pkl", "rb") as f:
        modelo_tree, modelo_knn, modelo_nn, columnas_modelo, scaler = pickle.load(f)
    return modelo_tree, modelo_knn, modelo_nn, columnas_modelo, scaler

modelo_tree, modelo_knn, modelo_nn, columnas_modelo, scaler = cargar_modelos()

# Entradas del usuario
st.subheader("🎮 Parámetros del usuario")

edad = st.slider("Edad", 14, 52, 25)

videojuego = st.selectbox("Tipo de videojuego", [
    'Mass Effect', 'Sim City', 'Dead Space', 'Battlefield', 'Fifa', 'F1', 'KOA: Reckoning'
])

plataforma = st.selectbox("Plataforma", [
    'Play Station', 'PC', 'Xbox', 'Otros'
])

sexo = st.selectbox("Sexo", ["Hombre", "Mujer"])

consumidor = st.checkbox("¿Es consumidor habitual?")

# Preparar entrada para predicción
def preparar_entrada(edad, sexo, consumidor, videojuego, plataforma):
    datos = pd.DataFrame([{
        "Edad": edad,
        "Sexo": sexo,
        "Consumidor_habitual": "Sí" if consumidor else "No",
        "videojuego": videojuego,
        "Plataforma": plataforma
    }])

    # Codificación one-hot sin drop_first
    datos = pd.get_dummies(datos)

    # Añadir columnas faltantes
    for col in columnas_modelo:
        if col not in datos.columns:
            datos[col] = 0

    # Ordenar columnas
    datos = datos[columnas_modelo]

    # Escalar edad
    datos[["Edad"]] = scaler.transform(datos[["Edad"]])

    return datos

X_input = preparar_entrada(edad, sexo, consumidor, videojuego, plataforma)

# Selección de modelo
modelo_sel = st.selectbox("🧠 Selecciona el modelo", ["Árbol de Decisión", "KNN", "Red Neuronal"])

# Predicción
if st.button("🔍 Predecir presupuesto"):
    if modelo_sel == "Árbol de Decisión":
        pred = modelo_tree.predict(X_input)[0]
    elif modelo_sel == "KNN":
        pred = modelo_knn.predict(X_input)[0]
    else:
        pred = modelo_nn.predict(X_input)[0]

    st.success(f"💰 Presupuesto estimado: **${pred:,.2f}**")




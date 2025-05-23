import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Configuración general
st.set_page_config(page_title="Predicción compra de videojuegos en tienda", layout="wide")

# Estilo personalizado
st.markdown(
    """
    <style>
    body {
        background-color: #0e1117;
        color: white;
    }
    .main {
        background-color: #0e1117;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Imagen de fondo decorativa
st.image("https://raw.githubusercontent.com/juan-mv-blip/car-risk-app/blob/main/prediccion_videojuego/assets/tienda.jpg", use_container_width=True)

# Título principal
st.title("🛒 Predicción compra de videojuegos en tienda")
st.markdown("Selecciona los parámetros del usuario para predecir el presupuesto de inversión.")

# Cargar modelos
@st.cache_resource
def cargar_modelos():
    with open("modelo-reg-tree-knn-nn.pkl", "rb") as f:
        modelo_tree, modelo_knn, modelo_nn, variables, scaler = pickle.load(f)
    return modelo_tree, modelo_knn, modelo_nn, variables, scaler

modelo_tree, modelo_knn, modelo_nn, variables, scaler = cargar_modelos()

# Parámetros del usuario
st.subheader("🎮 Parámetros del usuario:")

edad = st.slider("Edad", 14, 52, 25)

videojuego = st.selectbox(
    "Seleccione el tipo de videojuego",
    ['Mass Effect', 'Sim City', 'Dead Space', 'Battlefield', 'Fifa', 'F1', 'KOA: Reckoning']
)

plataforma = st.selectbox(
    "Plataforma",
    ['Play Station', 'PC', 'Xbox', 'Otros']
)

sexo = st.selectbox("Sexo", ["Hombre", "Mujer"])

consumidor = st.checkbox("Consumidor habitual")

# Transformación a input vector
def generar_input(edad, sexo, consumidor, videojuego, plataforma):
    datos_usuario = {
        "Edad": edad,
        "Sexo": sexo,
        "Consumidor_habitual": "Sí" if consumidor else "No",
        "videojuego": videojuego,
        "Plataforma": plataforma
    }

    df = pd.DataFrame([datos_usuario])

    # Crear dummies exactamente como en entrenamiento
    df = pd.get_dummies(df, columns=['videojuego', 'Plataforma', 'Sexo', 'Consumidor_habitual'], drop_first=True)

    # Asegurar que todas las variables esperadas por el modelo estén presentes
    for col in variables:
        if col not in df.columns:
            df[col] = 0

    # Reordenar columnas
    df = df[variables]

    # Normalizar edad
    df[['Edad']] = scaler.transform(df[['Edad']])

    return df

X_input = generar_input(edad, sexo, consumidor, videojuego, plataforma)

# Botón de predicción
modelo_sel = st.selectbox("Selecciona el modelo", ["Árbol de Decisión", "KNN", "Red Neuronal"])

if st.button("Predecir"):
    if modelo_sel == "Árbol de Decisión":
        resultado = modelo_tree.predict(X_input)[0]
    elif modelo_sel == "KNN":
        resultado = modelo_knn.predict(X_input)[0]
    else:
        resultado = modelo_nn.predict(X_input)[0]

    st.success(f"💰 Presupuesto estimado para invertir: ${resultado:,.2f}")



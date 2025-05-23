import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Configuración general
st.set_page_config(page_title="Predicción de Presupuesto en Videojuegos", layout="wide")

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

st.title("🎮 Predicción de Presupuesto en Videojuegos")
st.markdown("Selecciona características del consumidor y predice el presupuesto de inversión en videojuegos.")

# Cargar modelos
@st.cache_resource
def cargar_modelos():
    with open("modelo-reg-tree-knn-nn.pkl", "rb") as f:
        modelo_tree, modelo_knn, modelo_nn, variables, scaler = pickle.load(f)
    return modelo_tree, modelo_knn, modelo_nn, variables, scaler

modelo_tree, modelo_knn, modelo_nn, variables, scaler = cargar_modelos()

# Inputs del usuario
edad = st.slider("Edad", 10, 70, 25)
sexo = st.selectbox("Sexo", ["Femenino", "Masculino"])
consumidor = st.selectbox("Consumidor habitual", ["Sí", "No"])
videojuego = st.selectbox("Videojuego", ['FIFA', 'Minecraft', 'League of Legends', 'Among Us'])
plataforma = st.selectbox("Plataforma", ['PC', 'PlayStation', 'Xbox', 'Nintendo'])

# Transformación a input vector
def generar_input(edad, sexo, consumidor, videojuego, plataforma):
    # Crear diccionario con datos originales
    datos_usuario = {
        "Edad": edad,
        "Sexo": sexo,
        "Consumidor_habitual": consumidor,
        "videojuego": videojuego,
        "Plataforma": plataforma
    }

    # Convertir a DataFrame y aplicar dummies
    df = pd.DataFrame([datos_usuario])
    df = pd.get_dummies(df, columns=['videojuego', 'Plataforma', 'Sexo', 'Consumidor_habitual'], drop_first=True)

    # Añadir columnas faltantes
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

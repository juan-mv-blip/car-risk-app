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
st.image("assets/fondo.jpg", use_container_width=True)

st.title("🎮 Predicción de Presupuesto en Videojuegos")
st.markdown("Selecciona características del consumidor y predice el presupuesto de inversión en videojuegos.")

# Cargar modelos
@st.cache_resource
def cargar_modelos():
    with open("modelos/modelo-reg-tree-knn-nn.pkl", "rb") as f:
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
    df = pd.DataFrame(np.zeros((1, len(variables))), columns=variables)
    df['Edad'] = edad
    if sexo == 'Masculino':
        df['Sexo_Masculino'] = 1
    if consumidor == 'Sí':
        df['Consumidor_habitual_Sí'] = 1
    if f'videojuego_{videojuego}' in df.columns:
        df[f'videojuego_{videojuego}'] = 1
    if f'Plataforma_{plataforma}' in df.columns:
        df[f'Plataforma_{plataforma}'] = 1
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

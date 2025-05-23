import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re

# Función para normalizar nombres a formato dummy variable
def normalize_dummy_name(name):
    name = name.replace("'", "")  # eliminar comillas
    name = re.sub(r'[^\w]', '_', name)  # reemplazar caracteres no alfanuméricos por _
    name = re.sub(r'_+', '_', name)  # evitar __ dobles
    name = name.strip('_')
    return name

# Cargar modelos entrenados y variables
with open("modelo-reg-tree-knn-nn.pkl", "rb") as f:
    model_Tree, model_Knn, model_NN, variables, min_max_scaler = pickle.load(f)

# Selección de modelo
st.title("🎮 Predicción de Presupuesto para Videojuegos")

modelo_nombre = st.selectbox("Selecciona el modelo a usar", ["Árbol de Decisión", "KNN", "Red Neuronal"])
modelo = {"Árbol de Decisión": model_Tree, "KNN": model_Knn, "Red Neuronal": model_NN}[modelo_nombre]

# Entradas del usuario
st.header("📋 Ingresa los datos del consumidor")

edad = st.slider("Edad", min_value=14, max_value=52, value=25)

# Detectar categorías desde variables del modelo
videojuegos_dummies = [v for v in variables if v.startswith("videojuego_")]
plataformas_dummies = [v for v in variables if v.startswith("Plataforma_")]

# Mostrar nombres bonitos en el selectbox
videojuegos_nombres = [v.replace("videojuego_", "").replace("_", " ") for v in videojuegos_dummies]
plataformas_nombres = [p.replace("Plataforma_", "").replace("_", " ") for p in plataformas_dummies]

videojuego = st.selectbox("¿Qué videojuego le interesa?", videojuegos_nombres)
plataforma = st.selectbox("Plataforma preferida", plataformas_nombres)
sexo = st.selectbox("Sexo", ['Hombre', 'Mujer'])
consumidor_habitual = st.checkbox("¿Es consumidor habitual?", value=True)

# Crear input
input_dict = {col: 0 for col in variables}
input_dict["Edad"] = edad

# Convertir entradas a nombres dummy válidos
vj_col = "videojuego_" + normalize_dummy_name(videojuego)
plataforma_col = "Plataforma_" + normalize_dummy_name(plataforma)

# Asignar valores en input_dict si existen
if vj_col in variables:
    input_dict[vj_col] = 1
else:
    st.warning(f"⚠️ La variable {vj_col} no existe en el modelo.")

if plataforma_col in variables:
    input_dict[plataforma_col] = 1
else:
    st.warning(f"⚠️ La variable {plataforma_col} no existe en el modelo.")

# Sexo y consumidor habitual
if "Sexo_Mujer" in variables:
    input_dict["Sexo_Mujer"] = 1 if sexo == "Mujer" else 0
if "Consumidor_habitual_True" in variables:
    input_dict["Consumidor_habitual_True"] = 1 if consumidor_habitual else 0

# Convertir a DataFrame
input_df = pd.DataFrame([input_dict])

# Normalizar Edad para KNN y Red Neuronal
if modelo_nombre in ["KNN", "Red Neuronal"]:
    input_df[["Edad"]] = min_max_scaler.transform(input_df[["Edad"]])

# Mostrar datos al usuario
st.subheader("🔍 Datos que se ingresan al modelo")
st.dataframe(input_df)

# Predicción
if st.button("📊 Predecir presupuesto"):
    pred = modelo.predict(input_df)[0]
    st.success(f"💰 Presupuesto estimado: ${pred:,.2f}")





import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re

# Función para normalizar nombres a formato dummy variable
def normalize_dummy_name(name):
    # Elimina comillas simples, reemplaza espacios, dos puntos y caracteres especiales por _
    name = name.replace("'", "")
    name = re.sub(r'[^a-zA-Z0-9]', '_', name)
    name = re.sub(r'__+', '_', name)  # Reemplaza dobles guiones bajos por uno solo
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

# Extraer nombres para mostrar en selectbox (normalizando)
videojuegos = []
for v in videojuegos_dummies:
    base = v.replace("videojuego_", "")
    base_norm = base.replace("_", " ")
    videojuegos.append(base_norm)

plataformas = []
for p in plataformas_dummies:
    base = p.replace("Plataforma_", "")
    base_norm = base.replace("_", " ")
    plataformas.append(base_norm)

videojuego = st.selectbox("¿Qué videojuego le interesa?", videojuegos)
plataforma = st.selectbox("Plataforma preferida", plataformas)
sexo = st.selectbox("Sexo", ['Hombre', 'Mujer'])
consumidor_habitual = st.checkbox("¿Es consumidor habitual?", value=True)

# Crear diccionario de input inicializado en 0
input_dict = {col: 0 for col in variables}
input_dict["Edad"] = edad

# Normalizar las opciones seleccionadas para que coincidan con nombres dummy
vj_dummy_name = "videojuego_" + normalize_dummy_name(videojuego)
plataforma_dummy_name = "Plataforma_" + normalize_dummy_name(plataforma)

if vj_dummy_name in variables:
    input_dict[vj_dummy_name] = 1
else:
    st.warning(f"⚠️ Variable {vj_dummy_name} no encontrada en el modelo.")

if plataforma_dummy_name in variables:
    input_dict[plataforma_dummy_name] = 1
else:
    st.warning(f"⚠️ Variable {plataforma_dummy_name} no encontrada en el modelo.")

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

# Mostrar el input
st.subheader("🔍 Datos que se ingresan al modelo")
st.dataframe(input_df)

# Botón de predicción
if st.button("📊 Predecir presupuesto"):
    pred = modelo.predict(input_df)[0]
    st.success(f"💰 Presupuesto estimado: ${pred:,.2f}")



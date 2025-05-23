import streamlit as st
import pandas as pd
import pickle

# Cargar modelos entrenados y variables
with open("modelo-reg-tree-knn-nn.pkl", "rb") as f:
    _, _, model_NN, variables, min_max_scaler = pickle.load(f)

# Título principal
st.title("🎮 Predicción de Presupuesto para Videojuegos")

# Sección de entrada de datos
st.header("📋 Ingresa los datos del consumidor")

edad = st.slider("Edad", min_value=14, max_value=52, value=25)

# Detectar automáticamente las categorías desde las variables del modelo
videojuegos_dummies = [v for v in variables if v.startswith("videojuego_")]
plataformas_dummies = [v for v in variables if v.startswith("Plataforma_")]

videojuegos = [v.replace("videojuego_", "").replace("_", " ") for v in videojuegos_dummies]
plataformas = [v.replace("Plataforma_", "").replace("_", " ") for v in plataformas_dummies]

videojuego = st.selectbox("¿Qué videojuego le interesa?", videojuegos)
plataforma = st.selectbox("Plataforma preferida", plataformas)
sexo = st.selectbox("Sexo", ['Hombre', 'Mujer'])
consumidor_habitual = st.checkbox("¿Es consumidor habitual?", value=True)

# Crear el input del modelo
input_dict = {col: 0 for col in variables}
input_dict["Edad"] = edad

# Convertir selección del usuario a formato dummy (reemplaza espacios por guiones bajos)
vj_dummy = f"videojuego_{videojuego.replace(' ', '_')}"
plataforma_dummy = f"Plataforma_{plataforma.replace(' ', '_')}"

# Agregar al diccionario si existen en el modelo
if vj_dummy in variables:
    input_dict[vj_dummy] = 1
else:
    st.warning(f"⚠️ La variable {vj_dummy} no existe en el modelo.")

if plataforma_dummy in variables:
    input_dict[plataforma_dummy] = 1
else:
    st.warning(f"⚠️ La variable {plataforma_dummy} no existe en el modelo.")

# Sexo y consumidor habitual
if "Sexo_Mujer" in variables:
    input_dict["Sexo_Mujer"] = 1 if sexo == "Mujer" else 0
if "Consumidor_habitual_True" in variables:
    input_dict["Consumidor_habitual_True"] = 1 if consumidor_habitual else 0

# Convertir a DataFrame
input_df = pd.DataFrame([input_dict])

# Normalizar Edad
input_df[["Edad"]] = min_max_scaler.transform(input_df[["Edad"]])

# Mostrar los datos ingresados
st.subheader("🔍 Datos que se ingresan al modelo")
st.dataframe(input_df)

# Botón de predicción
if st.button("📊 Predecir presupuesto"):
    pred = model_NN.predict(input_df)[0]
    st.success(f"💰 Presupuesto estimado: ${pred:,.2f}")






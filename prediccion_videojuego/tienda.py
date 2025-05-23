import streamlit as st
import pandas as pd
import pickle

# Cargar el modelo y variables
filename = 'modelo-reg-tree-knn-nn.pkl'
_, _, model_NN, variables, min_max_scaler = pickle.load(open(filename, 'rb'))

# Extraer columnas dummy desde `variables`
videojuegos_dummies = [v for v in variables if v.startswith("videojuego_")]
plataformas_dummies = [v for v in variables if v.startswith("Plataforma_")]

# Crear diccionarios: mostrar valor limpio pero guardar el nombre dummy exacto
videojuego_map = {v.replace("videojuego_", "").replace("_", " "): v for v in videojuegos_dummies}
plataforma_map = {v.replace("Plataforma_", "").replace("_", " "): v for v in plataformas_dummies}

# Convertir a listas ordenadas sin comillas innecesarias
videojuego_opciones = list(videojuego_map.keys())
plataforma_opciones = list(plataforma_map.keys())

# Título y formulario
st.title("🎮 Predicción de Presupuesto para Videojuegos")
st.header("📋 Ingresa los datos del consumidor")

edad = st.slider("Edad", min_value=14, max_value=52, value=25)
videojuego = st.selectbox("¿Qué videojuego le interesa?", list(videojuego_map.keys()))
plataforma = st.selectbox("Plataforma preferida", list(plataforma_map.keys()))
sexo = st.selectbox("Sexo", ["Hombre", "Mujer"])
consumidor = st.checkbox("¿Es consumidor habitual?", value=True)

# Crear DataFrame base con todas las variables inicializadas en 0
input_data = pd.DataFrame(columns=variables)
input_data.loc[0] = 0

# Normalizar edad
input_data["Edad"] = min_max_scaler.transform([[edad]])[0][0]

# Activar las dummies correspondientes
input_data[videojuego_map[videojuego]] = 1
input_data[plataforma_map[plataforma]] = 1
if sexo == "Mujer":
    input_data["Sexo_Mujer"] = 1
if consumidor:
    input_data["Consumidor_habitual_True"] = 1

# Mostrar datos ingresados
st.subheader("🔍 Datos que se ingresan al modelo")
st.dataframe(input_data)

# Botón para predecir
if st.button("📊 Predecir presupuesto"):
    pred = model_NN.predict(input_data)[0]
    st.success(f"💰 Presupuesto estimado: ${pred:,.2f}")







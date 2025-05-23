import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Predicción de Riesgo Vehicular", page_icon="🚗", layout="centered")

st.markdown("""
    <h1 style='text-align: center;'>🚗 Predicción de Riesgo Vehicular</h1>
    <p style='text-align: center;'>Selecciona las características del conductor para predecir el riesgo de seguro</p>
""", unsafe_allow_html=True)

# Entradas del usuario
edad = st.slider("Edad del conductor", 18, 70, 35)
cartype = st.selectbox("Tipo de vehículo", ["combi", "sport", "family", "minivan"])
modelo_nombre = st.selectbox("Modelo a usar", ["DT (Árbol)", "KNN", "NN (Red Neuronal)"])

modelos_dict = {
    "DT (Árbol)": "modelo-clas-tree.pkl",
    "KNN": "modelo-clas-tree-knn-nn.pkl",
    "NN (Red Neuronal)": "modelo-clas-tree-knn-nn.pkl",
}

# Al presionar el botón, se carga el modelo y se predice
if st.button("🔍 Predecir riesgo"):
    try:
        # Cargar el archivo pickle
        modelo_path = modelos_dict[modelo_nombre]
        modelo_data = joblib.load(modelo_path)

        if modelo_path == "modelo-clas-tree.pkl":
            modelo = modelo_data[0]
            labelencoder = modelo_data[1]
            columnas = modelo_data[2]
            normalizador = None
        else:
            if modelo_nombre.startswith("KNN"):
                modelo = modelo_data[1]
            else:
                modelo = modelo_data[2]
            labelencoder = modelo_data[3]
            columnas = modelo_data[4]
            normalizador = modelo_data[5]

        # Crear DataFrame de entrada
        input_df = pd.DataFrame({"age": [edad], "cartype": [cartype]})

        # Dummies igual que entrenamiento
        input_dummies = pd.get_dummies(input_df, columns=["cartype"])

        # Agregar columnas faltantes
        for col in columnas:
            if col not in input_dummies.columns:
                input_dummies[col] = 0

        # Ordenar columnas igual que el entrenamiento
        input_dummies = input_dummies[columnas]

        # Normalizar si es necesario
        if normalizador:
            input_dummies[["age"]] = normalizador.transform(input_dummies[["age"]])

        # Predecir
        pred = modelo.predict(input_dummies)[0]
        clase = labelencoder.inverse_transform([pred])[0]

        st.success(f"🔎 Riesgo estimado: **{clase.upper()}**")

    except Exception as e:
        st.error(f"Ocurrió un error al predecir: {e}")

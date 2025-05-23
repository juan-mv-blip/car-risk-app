import streamlit as st
import pandas as pd
import joblib

st.title("Predicción de Riesgo en Ventas de Seguros Vehiculares")

# Entradas del usuario
edad = st.slider("Edad del conductor", 18, 60, 30)
cartype = st.selectbox("Tipo de vehículo", ["combi", "sport", "family", "minivan"])
modelo_nombre = st.selectbox("Modelo a usar", ["DT", "KNN", "NN"])

# Mapeo de modelos a archivos
modelos_dict = {
    "DT": "modelo-clas-tree.pkl",
    "KNN": "modelo-clas-tree-knn-nn.pkl",
    "NN": "modelo-clas-tree-RL.pkl"
}

# Crear DataFrame con entrada
input_data = pd.DataFrame({"age": [edad], "cartype": [cartype]})

# Al presionar el botón, se carga el modelo y se predice
if st.button("Predecir riesgo"):
    modelo_path = modelos_dict[modelo_nombre]
    modelo = joblib.load(modelo_path)
    prediccion = modelo.predict(input_data)[0]
    st.success(f"⚠️ Riesgo estimado: {prediccion.upper()}")

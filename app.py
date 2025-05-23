import streamlit as st
import pandas as pd
import joblib

# Configuración de página
st.set_page_config(page_title="Riesgo Vehicular", page_icon="🚗", layout="centered")

# Imagen decorativa (reemplaza esta URL si tienes una imagen propia en GitHub)
st.image("https://github.com/juan-mv-blip/car-risk-app/blob/main/assets/auto.jpg", width=120)

# Título estilizado
st.markdown("""
    <h1 style='text-align: center; color: #00c4b3;'>🚗 Riesgo en Seguros Vehiculares</h1>
    <p style='text-align: center; font-size: 16px;'>Estima el nivel de riesgo de un conductor según su edad y tipo de vehículo</p>
""", unsafe_allow_html=True)

# Entradas del usuario
edad = st.slider("🧍 Edad del conductor", 18, 70, 35)
cartype = st.selectbox("🚙 Tipo de vehículo", ["combi", "sport", "family", "minivan"])
modelo_nombre = st.selectbox("🧠 Modelo a usar", ["DT (Árbol)", "KNN", "NN (Red Neuronal)"])

# Mapeo de nombres del modelo
modelos_dict = {
    "DT (Árbol)": "modelo-clas-tree.pkl",
    "KNN": "modelo-clas-tree-knn-nn.pkl",
    "NN (Red Neuronal)": "modelo-clas-tree-knn-nn.pkl"
}

# Formateo de entrada
input_df = pd.DataFrame({"age": [edad], "cartype": [cartype]})
input_df = pd.get_dummies(input_df)

# Aseguramos que tenga las columnas correctas en el mismo orden
modelo_path = modelos_dict[modelo_nombre]
modelo_cargado = joblib.load(modelo_path)

# Cargamos adecuadamente los objetos según el modelo
if modelo_nombre == "DT (Árbol)":
    model, labelencoder, columnas = modelo_cargado
elif modelo_nombre == "KNN":
    _, model, _, labelencoder, columnas, scaler = modelo_cargado
    input_df[['age']] = scaler.transform(input_df[['age']])
elif modelo_nombre == "NN (Red Neuronal)":
    _, _, model, labelencoder, columnas, scaler = modelo_cargado
    input_df[['age']] = scaler.transform(input_df[['age']])

# Agregamos columnas faltantes
for col in columnas:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[columnas]

# Predicción
if st.button("🔍 Predecir riesgo"):
    pred = model.predict(input_df)[0]
    resultado = labelencoder.inverse_transform([pred])[0]

    icono = "✅" if resultado == "low" else "⚠️" if resultado == "medium" else "🚨"
    st.success(f"{icono} Riesgo estimado: **{resultado.upper()}**")

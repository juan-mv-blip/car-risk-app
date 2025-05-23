import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Cargar modelo y columnas utilizadas en el entrenamiento
filename = 'modelo-reg-tree-knn-nn.pkl'
model_tree, model_knn, model_nn, variables, min_max_scaler = pickle.load(open(filename, 'rb'))

# Función para generar el input del usuario
def generar_input(edad, sexo, consumidor_habitual, videojuego, plataforma):
    # Crear DataFrame base
    data = pd.DataFrame([{
        'Edad': edad,
        'Sexo': sexo,
        'Consumidor_habitual': consumidor_habitual,
        'videojuego': videojuego,
        'Plataforma': plataforma
    }])

    # Aplicar get_dummies igual que en entrenamiento
    data = pd.get_dummies(data, columns=['videojuego', 'Plataforma'], drop_first=False)
    data = pd.get_dummies(data, columns=['Sexo', 'Consumidor_habitual'], drop_first=True)

    # Escalar edad
    data[['Edad']] = min_max_scaler.transform(data[['Edad']])

    # Asegurar que todas las columnas necesarias están presentes
    for col in variables:
        if col not in data.columns:
            data[col] = 0

    # Asegurar el orden de columnas
    data = data[variables]

    return data

# Título
st.title("🎮 Predicción de gasto promedio en videojuegos")

# Entradas del usuario
edad = st.slider("Edad del jugador", min_value=10, max_value=70, value=25)
sexo = st.selectbox("Sexo", ["Hombre", "Mujer"])
consumidor = st.selectbox("¿Es consumidor habitual?", ["Sí", "No"]) == "Sí"
videojuego = st.selectbox("Videojuego favorito", ["Fifa", "GTA", "Minecraft", "Call of Duty", "Fortnite"])
plataforma = st.selectbox("Plataforma preferida", ["PC", "PlayStation", "Xbox", "Switch", "Móvil"])

# Generar entrada del usuario
X_input = generar_input(edad, sexo, consumidor, videojuego, plataforma)

# Mostrar input procesado para depuración
st.subheader("🔍 Entrada al modelo")
st.dataframe(X_input)

# Realizar predicciones
pred_tree = model_tree.predict(X_input)[0]
pred_knn = model_knn.predict(X_input)[0]
pred_nn = model_nn.predict(X_input)[0]

# Mostrar resultados
st.subheader("💡 Predicción del gasto promedio mensual")
st.write(f"🌳 Árbol de decisión: **${pred_tree:.2f}**")
st.write(f"🤖 K-Nearest Neighbors: **${pred_knn:.2f}**")
st.write(f"🧠 Red neuronal: **${pred_nn:.2f}**")

# Diagnóstico rápido
if X_input.sum().sum() == 0:
    st.warning("⚠️ Entrada completamente vacía. Verifica si las categorías coinciden con las del entrenamiento.")

# Prueba de múltiples edades para diagnóstico
st.subheader("🧪 Diagnóstico: efecto de la edad")
for edad_test in [18, 25, 35, 45, 60]:
    X_test = generar_input(edad_test, sexo, consumidor, videojuego, plataforma)
    pred_test = model_tree.predict(X_test)[0]
    st.write(f"Edad: {edad_test} → Árbol: ${pred_test:.2f}")




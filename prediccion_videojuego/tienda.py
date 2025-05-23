import streamlit as st
import pandas as pd
import pickle

# Cargar modelos y datos de entrenamiento
model_Tree, model_Knn, model_NN, variables, min_max_scaler = pickle.load(open('modelo-reg-tree-knn-nn.pkl', 'rb'))

# Título
st.title('Predicción de Tiempo de Juego por Videojuego')

# Entradas del usuario
videojuego = st.selectbox('Selecciona un videojuego:', [
    'Battlefield', 'Crysis', 'Dead Space', 'F1', 'Fifa',
    'KOA: Reckoning', 'Mass Effect', 'Sim City'
])

plataforma = st.selectbox('Selecciona la plataforma:', [
    'Play Station', 'Xbox', 'Otros', 'PC'
])

sexo = st.selectbox('Selecciona tu sexo:', ['Hombre', 'Mujer'])

edad = st.slider('Selecciona tu edad:', min_value=10, max_value=60, value=25)

consumidor_habitual = st.selectbox('¿Eres consumidor habitual de videojuegos?', ['Sí', 'No'])
consumidor_habitual = True if consumidor_habitual == 'Sí' else False

# Crear DataFrame con los datos ingresados
input_data = pd.DataFrame([{
    'Edad': edad,
    'videojuego': videojuego,
    'Plataforma': plataforma,
    'Sexo': sexo,
    'Consumidor_habitual': consumidor_habitual
}])

# One-hot encoding
input_data = pd.get_dummies(input_data, columns=['videojuego', 'Plataforma', 'Sexo', 'Consumidor_habitual'], drop_first=False)

# Alinear con columnas de entrenamiento
input_data = input_data.reindex(columns=variables, fill_value=0)

# Escalar edad
input_data[['Edad']] = min_max_scaler.transform(input_data[['Edad']])

# Mostrar entrada procesada (opcional para debug)
st.subheader("Entrada del modelo:")
st.dataframe(input_data)

# Predicciones
pred_tree = model_Tree.predict(input_data)[0]
pred_knn = model_Knn.predict(input_data)[0]
pred_nn = model_NN.predict(input_data)[0]

# Mostrar resultados
st.subheader("Resultados de la predicción:")
st.write(f"Árbol de decisión: **{round(pred_tree, 2)}** minutos")
st.write(f"KNN: **{round(pred_knn, 2)}** minutos")
st.write(f"Red neuronal: **{round(pred_nn, 2)}** minutos")




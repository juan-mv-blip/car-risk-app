# Cargamos librerías principales
import numpy as np
import pandas as pd
import pickle

# Cargamos el modelo
filename = 'modelo-reg-tree-knn-nn.pkl'
model_Tree, model_Knn, model_NN, variables, min_max_scaler = pickle.load(open(filename, 'rb'))

# Cargamos los datos futuros
data = pd.read_csv("videojuegos-datosFuturos.csv", sep=",")

# Limpiamos comillas simples en variables categóricas
data['videojuego'] = data['videojuego'].str.replace("'", "", regex=False)
data['Plataforma'] = data['Plataforma'].str.replace("'", "", regex=False)

# Preparamos los datos
data_preparada = data.copy()
data_preparada = pd.get_dummies(data_preparada, columns=['videojuego', 'Plataforma'], drop_first=False)
data_preparada = pd.get_dummies(data_preparada, columns=['Sexo', 'Consumidor_habitual'], drop_first=True)

# Agregamos columnas faltantes que espera el modelo
data_preparada = data_preparada.reindex(columns=variables, fill_value=0)

# Normalizamos la Edad
data_preparada[['Edad']] = min_max_scaler.transform(data_preparada[['Edad']])

# Realizamos predicciones
Y_Tree = model_Tree.predict(data_preparada)
Y_Knn = model_Knn.predict(data_preparada)
Y_NN = model_NN.predict(data_preparada)

# Guardamos resultados en el dataframe original
data['Prediccion_Tree'] = Y_Tree
data['Prediccion_Knn'] = Y_Knn
data['Prediccion_NN'] = Y_NN

# Mostramos el resultado final
print(data)



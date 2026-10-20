#Importar librerias
import pandas as pd
#Cargar el dataset
df = pd.read_csv("ENB2012_data.csv", sep=";")
#Visualizar la información
print(df.head())   # Muestra las primeras 5 filas
df.info()          # Muestra columnas, tipos de datos y valores no nulos
print(df.shape)    # Muestra (cantidad de filas, cantidad de columnas

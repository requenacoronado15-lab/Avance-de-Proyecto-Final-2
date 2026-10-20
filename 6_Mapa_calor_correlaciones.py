# Importar librerías
import pandas as pd
import plotly.express as px
 
# Nombres descriptivos para las columnas del dataset ENB2012
nombres = {
    "X1": "Compacidad relativa",
    "X2": "Área superficial",
    "X3": "Área de paredes",
    "X4": "Área de techo",
    "X5": "Altura total",
    "X6": "Orientación",
    "X7": "Área de vidrio",
    "X8": "Distribución del vidrio",
    "Y1": "Carga de calefacción",
    "Y2": "Carga de refrigeración",
}
 
# Cargar el dataset (separado por ';'), normalizar encabezados,
# quitar filas vacías y eliminar filas duplicadas
df = pd.read_csv("ENB2012_data.csv", sep=";")
df.columns = df.columns.str.strip()
df = df.dropna(how="all")
df = df.drop_duplicates()
df = df.rename(columns=nombres)
 
# Calcular correlaciones entre las columnas numéricas
corr = df.corr(numeric_only=True)
 
# Crear el mapa de calor
fig = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    aspect="auto",
    title="Mapa de correlaciones",
    labels={"color": "Correlación"},
)
fig.update_xaxes(tickangle=-45)
fig.show()      


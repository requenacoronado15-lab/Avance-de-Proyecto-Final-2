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
 
# Cargar el dataset (separado por ';'), quitar filas vacías y duplicadas
df = pd.read_csv("ENB2012_data.csv", sep=";")
df.columns = df.columns.str.strip()
df = df.dropna(how="all")
df = df.drop_duplicates()
df = df.rename(columns=nombres)
 
# Variable del eje X (ordenada) y variable objetivo
# La base no tiene fechas: se usa la compacidad relativa como "eje de avance"
eje_x = "Compacidad relativa"
objetivo = "Carga de calefacción"   # o "Carga de refrigeración"
 
# Promedio de la variable objetivo y del diseño del edificio para cada valor del eje X
resumen = (
    df.groupby(eje_x)
    .agg(
        **{
            f"{objetivo} promedio": (objetivo, "mean"),
            "Área superficial promedio": ("Área superficial", "mean"),
            "Área de techo promedio": ("Área de techo", "mean"),
            "Altura total promedio": ("Altura total", "mean"),
            "Número de registros": (objetivo, "size"),
        }
    )
    .reset_index()
    .sort_values(eje_x)
)
 
# Crear la serie (línea con marcadores)
fig = px.line(
    resumen,
    x=eje_x,
    y=f"{objetivo} promedio",
    markers=True,
    hover_data={
        "Área superficial promedio": ":.1f",
        "Área de techo promedio": ":.1f",
        "Altura total promedio": ":.2f",
        "Número de registros": True,
    },
    title=f"{objetivo} promedio y diseño del edificio por {eje_x.lower()}",
    labels={
        eje_x: eje_x,
        f"{objetivo} promedio": f"{objetivo} promedio (kWh/m²)",
    },
)
fig.show()  

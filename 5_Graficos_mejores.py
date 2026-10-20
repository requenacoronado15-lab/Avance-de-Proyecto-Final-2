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
 
# Variables del gráfico
objetivo = "Carga de calefacción"      # eje Z (o "Carga de refrigeración")
eje_x = "Compacidad relativa"          # eje X
eje_y = "Área de vidrio"               # eje Y
color_var = "Área de techo"            # color de los puntos
 
# Crear un diagrama 3D interactivo: permite girar y acercar la vista
fig = px.scatter_3d(
    df,
    x=eje_x,
    y=eje_y,
    z=objetivo,
    color=color_var,
    color_continuous_scale="Viridis",
    opacity=0.75,
    hover_data=["Altura total", "Orientación", "Área superficial"],
    title=f"{eje_x}, {eje_y.lower()} y {objetivo.lower()} del edificio",
    labels={
        eje_x: "Compacidad relativa",
        eje_y: "Área de vidrio (m²)",
        objetivo: f"{objetivo} (kWh/m²)",
        color_var: "Área de techo (m²)",
    },
)
 
fig.update_traces(marker={"size": 4})
fig.update_layout(
    scene={
        "xaxis_title": "Compacidad relativa",
        "yaxis_title": "Área de vidrio (m²)",
        "zaxis_title": f"{objetivo} (kWh/m²)",
        "camera": {"eye": {"x": 1.6, "y": 1.6, "z": 1.2}},
    },
    coloraxis_colorbar={"title": "Área de techo (m²)"},
)
fig.show()

    
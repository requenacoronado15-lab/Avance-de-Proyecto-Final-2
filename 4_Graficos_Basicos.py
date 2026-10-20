# Importar librerías
import pandas as pd
import plotly.express as px
 
# Cargar el dataset (separado por ';') y preparar la variable objetivo
df = pd.read_csv("ENB2012_data.csv", sep=";")
df.columns = df.columns.str.strip()
 
# Quitar filas completamente vacías y duplicadas
df = df.dropna(how="all")
df = df.drop_duplicates()
 
# Variable objetivo: "Y1" = carga de calefacción, "Y2" = carga de refrigeración
objetivo = "Y1"
nombres = {
    "Y1": "Carga de calefacción",
    "Y2": "Carga de refrigeración",
}
nombre = nombres[objetivo]
 
# Estadísticos descriptivos
media = df[objetivo].mean()
moda = df[objetivo].mode().iloc[0]
mediana = df[objetivo].median()
desviacion_estandar = df[objetivo].std()
 
# Crear el histograma de la variable objetivo
fig = px.histogram(
    df,
    x=objetivo,
    nbins=30,
    title=f"Distribución de la {nombre.lower()} (sin filas vacías ni duplicadas)",
    labels={objetivo: nombre},
)
 
# Líneas verticales: media, moda y mediana
fig.add_vline(
    x=media,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Media: {media:.2f}",
)
fig.add_vline(
    x=moda,
    line_dash="dot",
    line_color="green",
    annotation_text=f"Moda: {moda:.2f}",
    annotation_position="top left",
)
fig.add_vline(
    x=mediana,
    line_dash="dashdot",
    line_color="blue",
    annotation_text=f"Mediana: {mediana:.2f}",
    annotation_position="bottom right",
)
 
# Cuadro con la desviación estándar
fig.add_annotation(
    xref="paper",
    yref="paper",
    x=0.99,
    y=0.99,
    xanchor="right",
    yanchor="top",
    text=f"Desviación estándar: {desviacion_estandar:.2f}",
    showarrow=False,
    bgcolor="white",
)
 
fig.show()  

 

import pandas as pd

# 1. Cargar (el CSV usa ';' como separador) y limpiar
df = pd.read_csv("ENB2012_data.csv", sep=";")
df.columns = df.columns.str.strip()
df = df.dropna(how="all")          # elimina las filas vacías del final
df = df.drop_duplicates()

# 2. Nombres descriptivos
df = df.rename(columns={
    "X1": "Compacidad_Relativa", "X2": "Area_Superficial_m2", "X3": "Area_Pared_m2",
    "X4": "Area_Techo_m2", "X5": "Altura_Total_m", "X6": "Orientacion",
    "X7": "Area_Acristalada_frac", "X8": "Distribucion_Acristalamiento",
    "Y1": "Carga_Calefaccion_kWh_m2", "Y2": "Carga_Refrigeracion_kWh_m2"})

# 3. Variables objetivo
y1 = "Carga_Calefaccion_kWh_m2"
y2 = "Carga_Refrigeracion_kWh_m2"

# 4. KPIs
calef_prom, calef_max, calef_min = df[y1].mean(), df[y1].max(), df[y1].min()
refri_prom, refri_max, refri_min = df[y2].mean(), df[y2].max(), df[y2].min()
n = df.shape[0]
altura_prom = df["Altura_Total_m"].mean()
vidrio_prom = df["Area_Acristalada_frac"].mean()
pct_eficientes = (df[y1] < 15).mean() * 100
corr_altura = df["Altura_Total_m"].corr(df[y1])

# 5. Mostrar resultados
print(f"Calefacción promedio: {calef_prom:.2f} kWh/m²")
print(f"Calefacción máxima: {calef_max:.2f} kWh/m²")
print(f"Calefacción mínima: {calef_min:.2f} kWh/m²")
print(f"Refrigeración promedio: {refri_prom:.2f} kWh/m²")
print(f"Refrigeración máxima: {refri_max:.2f} kWh/m²")
print(f"Refrigeración mínima: {refri_min:.2f} kWh/m²")
print(f"Número de registros: {n}")
print(f"Altura promedio: {altura_prom:.2f} m")
print(f"Área acristalada promedio: {vidrio_prom*100:.1f} % del piso")
print(f"Edificios muy eficientes (< 15 kWh/m²): {pct_eficientes:.1f} %")
print(f"Correlación altura vs calefacción: {corr_altura:.2f}")  

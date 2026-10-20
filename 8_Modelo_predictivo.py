import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Cargar y limpiar
df = pd.read_csv("ENB2012_data.csv", sep=";")
df.columns = df.columns.str.strip()
df = df.dropna(how="all").drop_duplicates()
df = df.rename(columns={
    "X1": "Compacidad_Relativa", "X2": "Area_Superficial_m2", "X3": "Area_Pared_m2",
    "X4": "Area_Techo_m2", "X5": "Altura_Total_m", "X6": "Orientacion",
    "X7": "Area_Acristalada_frac", "X8": "Distribucion_Acristalamiento",
    "Y1": "Carga_Calefaccion_kWh_m2", "Y2": "Carga_Refrigeracion_kWh_m2"})

# 2. Variables
objetivo = "Carga_Calefaccion_kWh_m2"          # cambia a Carga_Refrigeracion_kWh_m2 para Y2
X = df.drop(columns=["Carga_Calefaccion_kWh_m2", "Carga_Refrigeracion_kWh_m2"])
y = df[objetivo]

# 3. Entrenamiento / prueba (80 % / 20 %)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Modelos
modelos = {
    "Regresión lineal": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

resultados = []
for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    resultados.append({
        "Modelo": nombre,
        "R2": r2_score(y_test, pred),
        "MAE (kWh/m2)": mean_absolute_error(y_test, pred),
        "RMSE (kWh/m2)": np.sqrt(mean_squared_error(y_test, pred)),
    })
tabla = pd.DataFrame(resultados).set_index("Modelo").round(4)
print(tabla)

# 5. Mejor modelo
mejor_nombre = tabla["R2"].idxmax()
mejor = modelos[mejor_nombre]
pred = mejor.predict(X_test)
print("\nMejor modelo:", mejor_nombre)

# 6. Gráfica real vs predicho
plt.figure(figsize=(6, 5))
plt.scatter(y_test, pred, alpha=0.7)
lim = [y_test.min() - 1, y_test.max() + 1]
plt.plot(lim, lim, "r--")
plt.title(f"Real vs. predicho ({mejor_nombre})")
plt.xlabel("Carga real (kWh/m²)")
plt.ylabel("Carga predicha (kWh/m²)")
plt.tight_layout()
plt.savefig("real_vs_predicho.png", dpi=120)
plt.show()

# 7. Importancia de variables (solo modelos de árboles)
if hasattr(mejor, "feature_importances_"):
    imp = pd.Series(mejor.feature_importances_, index=X.columns).sort_values()
    plt.figure(figsize=(7, 4))
    imp.plot(kind="barh")
    plt.title("Importancia de variables")
    plt.xlabel("Importancia (adimensional)")
    plt.tight_layout()
    plt.savefig("importancia.png", dpi=120)
    plt.show()
    print(imp.sort_values(ascending=False).round(3))

# 8. Validación exigente: predecir formas de edificio que el modelo NO vio
grupos = df["Area_Superficial_m2"].astype(str) + "_" + df["Altura_Total_m"].astype(str)
r2_formas = cross_val_score(mejor.__class__(**mejor.get_params()),
                            X, y, cv=GroupKFold(n_splits=4), groups=grupos, scoring="r2").mean()
print(f"\nR2 en formas NO vistas ({mejor_nombre}): {r2_formas:.3f}")

# 9. Guardar y usar el modelo
joblib.dump(mejor, "modelo_calefaccion.joblib")
nuevo = pd.DataFrame([{
    "Compacidad_Relativa": 0.90, "Area_Superficial_m2": 563.5, "Area_Pared_m2": 318.5,
    "Area_Techo_m2": 122.5, "Altura_Total_m": 7.0, "Orientacion": 2,
    "Area_Acristalada_frac": 0.25, "Distribucion_Acristalamiento": 3}])
print(f"\nCalefacción prevista: {mejor.predict(nuevo)[0]:.2f} kWh/m²")  

    
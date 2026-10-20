"""
Dashboard integral con Dash - Eficiencia energética de edificios (ENB2012)
 
Uso (terminal de VS Code):
    pip install -r requirements.txt
    python 9_Dashboard_Integral.py
Luego abre http://127.0.0.1:8050 en el navegador.
Necesita ENB2012_data.csv en la misma carpeta que este script.
"""
from pathlib import Path
 
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, dash_table, Input, Output
from dash.exceptions import PreventUpdate
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, KFold, cross_validate, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
 
# ---------- Datos ----------
CSV = Path(__file__).resolve().parent / "ENB2012_data.csv"
if not CSV.exists():
    CSV = Path("ENB2012_data.csv")
 
df = pd.read_csv(CSV, sep=";")                      # el CSV usa ';'
df.columns = df.columns.str.strip()
df = df.dropna(how="all").drop_duplicates().reset_index(drop=True)   # quita filas vacías y duplicadas
df = df.rename(columns={
    "X1": "Compacidad relativa", "X2": "Área superficial (m²)", "X3": "Área de paredes (m²)",
    "X4": "Área de techo (m²)", "X5": "Altura total (m)", "X6": "Orientación",
    "X7": "Área acristalada (fracción)", "X8": "Distribución del acristalamiento",
    "Y1": "Carga de calefacción (kWh/m²)", "Y2": "Carga de refrigeración (kWh/m²)"})
 
calef = "Carga de calefacción (kWh/m²)"
refri = "Carga de refrigeración (kWh/m²)"
altura = "Altura total (m)"
vidrio = "Área acristalada (fracción)"
FEATS = [c for c in df.columns if c not in (calef, refri)]
X = df[FEATS]
OBJETIVOS = {calef: "calefacción", refri: "refrigeración"}
 
# Una "forma" de edificio = combinación de superficie y altura (12 formas en la base)
grupos = df["Área superficial (m²)"].astype(str) + "_" + df[altura].astype(str)
 
# ---------- Modelos (uno por cada carga) ----------
modelos = {
    "Regresión lineal": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
}
kf = KFold(n_splits=5, shuffle=True, random_state=42)
gkf = GroupKFold(n_splits=4)
 
 
def evaluar(objetivo):
    y = df[objetivo]
    filas = []
    for nombre, m in modelos.items():
        cv = cross_validate(clone(m), X, y, cv=kf,
                            scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"])
        cvg = cross_validate(clone(m), X, y, cv=gkf, groups=grupos, scoring="r2")
        filas.append({
            "Modelo": nombre,
            "R2": cv["test_r2"].mean(),
            "MAE (kWh/m²)": -cv["test_neg_mean_absolute_error"].mean(),
            "RMSE (kWh/m²)": -cv["test_neg_root_mean_squared_error"].mean(),
            "R2 formas no vistas": cvg["test_score"].mean(),
        })
    tabla = pd.DataFrame(filas).round(3)
    mejor = tabla.loc[tabla["R2"].idxmax(), "Modelo"]
    pred = pd.Series(cross_val_predict(clone(modelos[mejor]), X, y, cv=kf), index=df.index)
    final = clone(modelos[mejor]).fit(X, y)
    lineal = clone(modelos["Regresión lineal"]).fit(X, y)
    coef = pd.DataFrame({"Variable": FEATS, "Coeficiente": lineal[-1].coef_}).sort_values("Coeficiente")
    return {"tabla": tabla, "mejor": mejor, "pred": pred, "final": final, "coef": coef}
 
 
RES = {obj: evaluar(obj) for obj in OBJETIVOS}
 
# ---------- Utilidades ----------
def kpi(titulo, valor):
    return html.Div([html.Span(titulo), html.H3(valor)], className="kpi")
 
 
def fig_hist(d, obj):
    nombre = OBJETIVOS[obj]
    media, mediana, desv = d[obj].mean(), d[obj].median(), d[obj].std()
    moda = d[obj].mode().iloc[0]
    f = px.histogram(d, x=obj, nbins=30, title=f"Distribución de la carga de {nombre}")
    f.add_vline(x=media, line_dash="dash", line_color="red", annotation_text=f"Media: {media:.2f}")
    f.add_vline(x=mediana, line_dash="dashdot", line_color="blue",
                annotation_text=f"Mediana: {mediana:.2f}", annotation_position="bottom right")
    f.add_vline(x=moda, line_dash="dot", line_color="green",
                annotation_text=f"Moda: {moda:.2f}", annotation_position="top left")
    f.add_annotation(text=f"Desv. estándar: {desv:.2f} kWh/m²", xref="paper", yref="paper",
                     x=0.98, y=0.98, showarrow=False, bgcolor="white")
    f.update_layout(xaxis_title=f"Carga de {nombre} (kWh/m²)", yaxis_title="Cantidad de edificios")
    return f
 
 
def con_tendencia(f, x, y):
    """Agrega una recta de mínimos cuadrados (sin necesitar statsmodels)."""
    if x.nunique() > 1:
        b, a = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        f.add_scatter(x=xs, y=a + b * xs, mode="lines", line=dict(color="red", dash="dash"),
                      name="Tendencia", showlegend=False)
    return f
 
 
def fig_superficie(d, obj):
    d = d.assign(**{"Altura (m)": d[altura].astype(str)})
    f = px.scatter(d, x="Área superficial (m²)", y=obj, color="Altura (m)",
                   title="Área superficial vs. carga (color = altura)")
    f.update_layout(yaxis_title=f"Carga de {OBJETIVOS[obj]} (kWh/m²)")
    return con_tendencia(f, d["Área superficial (m²)"], d[obj])
 
 
def fig_calef_refri(d):
    d = d.assign(**{"Altura (m)": d[altura].astype(str)})
    f = px.scatter(d, x=calef, y=refri, color="Altura (m)", title="Calefacción vs. refrigeración")
    return con_tendencia(f, d[calef], d[refri])
 
 
def fig_corr(d):
    c = d.corr(numeric_only=True)
    f = px.imshow(c, text_auto=".2f", color_continuous_scale="RdBu_r",
                  zmin=-1, zmax=1, title="Matriz de correlación (celdas vacías = variable constante)")
    f.update_xaxes(tickangle=-45)
    return f
 
 
def fig_3d(d, obj):
    f = px.scatter_3d(d, x="Compacidad relativa", y=vidrio, z=obj, color=obj,
                      color_continuous_scale="Viridis",
                      title="Compacidad, área acristalada y carga")
    f.update_traces(marker=dict(size=4))
    f.update_layout(coloraxis_colorbar_title="kWh/m²")
    return f
 
 
def fig_serie(d, obj, eje):
    g = d.groupby(eje, as_index=False)[obj].mean().sort_values(eje)
    f = px.line(g, x=eje, y=obj, markers=True,
                title=f"Carga de {OBJETIVOS[obj]} promedio según {eje.lower()}")
    f.update_layout(yaxis_title="Promedio (kWh/m²)")
    return f
 
 
def fig_real_pred(d, obj):
    r = RES[obj]
    y = df[obj]
    f = px.scatter(x=d[obj], y=r["pred"].loc[d.index],
                   title=f"Carga real vs. predicha ({r['mejor']}, validación cruzada)",
                   labels={"x": "Real (kWh/m²)", "y": "Predicha (kWh/m²)"})
    f.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(),
                line=dict(color="red", dash="dash"))
    return f
 
 
def fig_coef(obj):
    f = px.bar(RES[obj]["coef"], x="Coeficiente", y="Variable", orientation="h",
               color="Coeficiente", color_continuous_scale="RdBu",
               title="Influencia de cada variable (regresión lineal estandarizada)")
    f.update_layout(coloraxis_showscale=False)
    return f
 
 
def conclusiones(d, obj):
    nombre = OBJETIVOS[obj]
    otro = refri if obj == calef else calef
    r = RES[obj]
    fila = r["tabla"].set_index("Modelo").loc[r["mejor"]]
    pts = [html.Li([f"La carga de {nombre} promedio es ", html.B(f"{d[obj].mean():.2f} kWh/m²"),
                    f" (mediana {d[obj].median():.2f}, desv. estándar {d[obj].std():.2f}) en {len(d)} edificios."])]
    alt = d.groupby(altura)[obj].mean()
    if len(alt) > 1:
        pts.append(html.Li(f"Altura: los edificios de {alt.index.max():g} m promedian {alt.max():.2f} kWh/m² "
                           f"frente a {alt.min():.2f} kWh/m² los de {alt.index.min():g} m."))
    else:
        pts.append(html.Li(f"El filtro actual solo incluye edificios de {alt.index[0]:g} m de altura."))
    vid = d.groupby(vidrio)[obj].mean()
    if len(vid) > 1:
        pts.append(html.Li(f"Acristalamiento: de {vid.index.min() * 100:.0f} % a {vid.index.max() * 100:.0f} % del piso, "
                           f"la carga promedio pasa de {vid.iloc[0]:.2f} a {vid.iloc[-1]:.2f} kWh/m²."))
    ori = d.groupby("Orientación")[obj].mean()
    if len(ori) > 1:
        pts.append(html.Li(f"Orientación: la diferencia máxima entre orientaciones es de solo {ori.max() - ori.min():.2f} kWh/m²."))
    if d[obj].std() > 0 and d[otro].std() > 0:
        pts.append(html.Li(f"Calefacción y refrigeración se mueven juntas (correlación {d[obj].corr(d[otro]):.2f})."))
    pts.append(html.Li([f"Modelo: {r['mejor']} con R² = {fila['R2']:.3f} en validación cruzada (toda la base). ",
                        "Al dejar fuera formas de edificio completas el R² baja a ",
                        html.B(f"{fila['R2 formas no vistas']:.2f}"),
                        ": úsalo con cautela fuera de las 12 formas de la base."]))
    mg = r["tabla"].loc[r["tabla"]["R2 formas no vistas"].idxmax()]
    if mg["Modelo"] != r["mejor"]:
        pts.append(html.Li(f"Para formas de edificio nuevas, {mg['Modelo']} generaliza mejor "
                           f"(R² = {mg['R2 formas no vistas']:.2f}) aunque tenga menor R² en validación aleatoria."))
    pts.append(html.Li("Limitación: son edificios simulados (según la documentación del dataset) y las relaciones "
                       "son asociaciones; compacidad, superficie, techo y altura están muy relacionadas entre sí."))
    return html.Ul(pts)
 
 
# ---------- Layout ----------
EJES_SERIE = ["Compacidad relativa", "Área superficial (m²)", "Área de paredes (m²)",
              "Área de techo (m²)", altura, vidrio, "Distribución del acristalamiento"]
DEF = dict(zip(FEATS, [0.90, 563.5, 318.5, 122.5, 7.0, 3, 0.25, 3]))
lo = float(np.floor(min(df[calef].min(), df[refri].min())))
hi = float(np.ceil(max(df[calef].max(), df[refri].max())))
 
app = Dash(__name__)
app.title = "Dashboard de eficiencia energética"
app.index_string = """<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
body{font-family:system-ui,sans-serif;background:#f4f6f8;color:#1f2933;margin:0;padding:24px}
h1{margin:0 0 16px}.card{background:#fff;border:1px solid #e3e8ee;border-radius:10px;padding:14px;margin-bottom:16px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:16px}
.kpi{background:#fff;border:1px solid #e3e8ee;border-radius:10px;padding:12px}
.kpi span{font-size:12px;color:#6b7785}.kpi h3{margin:4px 0 0;font-size:22px;color:#2f6fed}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:16px}
.filtros{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px;align-items:start}
.filtros label{font-size:13px;display:block;margin-bottom:6px}
.form{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}
.form label{display:flex;flex-direction:column;font-size:13px;gap:4px}
#pred{font-size:22px;font-weight:700;color:#2f6fed;margin-top:12px;line-height:1.6}
#concl li{margin-bottom:6px}
</style></head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>"""
 
app.layout = html.Div([
    html.H1("Dashboard de eficiencia energética de edificios"),
    html.Div([
        html.Div([html.Label("Variable objetivo"),
                  dcc.RadioItems(id="objetivo", value=calef, inline=True,
                                 options=[{"label": " Calefacción  ", "value": calef},
                                          {"label": " Refrigeración", "value": refri}])]),
        html.Div([html.Label("Filtrar por carga (kWh/m²)"),
                  dcc.RangeSlider(id="rango", min=lo, max=hi, step=0.5, value=[lo, hi],
                                  tooltip={"placement": "bottom"})]),
        html.Div([html.Label("Altura del edificio (m)"),
                  dcc.Dropdown(id="altura", multi=True, clearable=False,
                               options=[{"label": f"{a:g} m", "value": a} for a in sorted(df[altura].unique())],
                               value=sorted(df[altura].unique().tolist()))]),
        html.Div([html.Label("Eje de la serie"),
                  dcc.Dropdown(id="eje-serie", options=EJES_SERIE, value="Compacidad relativa", clearable=False)]),
    ], className="card filtros"),
    html.Div(id="kpis", className="kpis"),
    html.Div([dcc.Graph(id=i) for i in
              ["g-hist", "g-sup", "g-cr", "g-corr", "g-3d", "g-serie", "g-real", "g-coef"]], className="grid"),
    html.Div([
        html.H3("Comparación de modelos (validación cruzada)"),
        html.P("R2, MAE y RMSE: 5 pliegues aleatorios. «R2 formas no vistas»: 4 pliegues que dejan fuera "
               "formas de edificio completas (más exigente).", style={"fontSize": "13px", "color": "#6b7785"}),
        dash_table.DataTable(id="tabla", columns=[{"name": c, "id": c} for c in RES[calef]["tabla"].columns],
                             data=RES[calef]["tabla"].to_dict("records"),
                             style_cell={"textAlign": "left", "padding": "6px"}),
    ], className="card", style={"marginTop": "16px"}),
    html.Div([
        html.H3("Predicción de la carga de un edificio"),
        html.Div([html.Label([f, dcc.Input(id=f"in-{i}", type="number", value=DEF[f], debounce=True,
                                           min=float(df[f].min()), max=float(df[f].max()), step="any")])
                  for i, f in enumerate(FEATS)], className="form"),
        html.Div(id="pred"),
    ], className="card"),
    html.Div([html.H3("Conclusiones"), html.Div(id="concl")], className="card"),
])
 
 
# ---------- Callbacks ----------
@app.callback(
    Output("kpis", "children"), Output("g-hist", "figure"), Output("g-sup", "figure"),
    Output("g-cr", "figure"), Output("g-corr", "figure"), Output("g-3d", "figure"),
    Output("g-serie", "figure"), Output("g-real", "figure"), Output("g-coef", "figure"),
    Output("tabla", "data"), Output("concl", "children"),
    Input("objetivo", "value"), Input("rango", "value"), Input("altura", "value"), Input("eje-serie", "value"),
)
def actualizar(obj, rango, alturas, eje):
    if not alturas:
        raise PreventUpdate
    d = df[df[obj].between(rango[0], rango[1]) & df[altura].isin(alturas)]
    if len(d) < 3:
        raise PreventUpdate
    n = OBJETIVOS[obj]
    kpis = [
        kpi("Registros", f"{len(d)}"),
        kpi("Calefacción promedio", f"{d[calef].mean():.2f} kWh/m²"),
        kpi("Refrigeración promedio", f"{d[refri].mean():.2f} kWh/m²"),
        kpi(f"Carga de {n} máxima", f"{d[obj].max():.2f} kWh/m²"),
        kpi(f"Carga de {n} mínima", f"{d[obj].min():.2f} kWh/m²"),
        kpi("Muy eficientes (< 15 kWh/m²)", f"{(d[obj] < 15).mean() * 100:.1f} %"),
        kpi("Altura promedio", f"{d[altura].mean():.2f} m"),
        kpi("Área acristalada promedio", f"{d[vidrio].mean() * 100:.1f} % del piso"),
        kpi(f"Correlación altura–{n}", "—" if d[altura].nunique() < 2 else f"{d[altura].corr(d[obj]):.2f}"),
    ]
    return (kpis, fig_hist(d, obj), fig_superficie(d, obj), fig_calef_refri(d), fig_corr(d),
            fig_3d(d, obj), fig_serie(d, obj, eje), fig_real_pred(d, obj), fig_coef(obj),
            RES[obj]["tabla"].to_dict("records"), conclusiones(d, obj))
 
 
@app.callback(Output("pred", "children"), [Input(f"in-{i}", "value") for i in range(len(FEATS))])
def predecir(*valores):
    if None in valores:
        return "Completa todos los campos"
    fila = pd.DataFrame([valores], columns=FEATS)
    out = [html.Div(f"Carga de {n}: {RES[obj]['final'].predict(fila)[0]:.2f} kWh/m²") for obj, n in OBJETIVOS.items()]
    fuera = [f for f, v in zip(FEATS, valores) if not (df[f].min() <= v <= df[f].max())]
    if fuera:
        out.append(html.Div("Aviso: fuera del rango de la base en " + ", ".join(fuera) + "; la predicción puede ser poco fiable.",
                            style={"fontSize": "13px", "fontWeight": "400", "color": "#b54708"}))
    return out
 
 
if __name__ == "__main__":
    app.run(debug=True)

    
# %% 
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score
from IPython.display import display

# %%
ruta_datos = Path("../../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = pd.read_csv(ruta_datos)

display(df.head())
# %%
variables_modelo = [
    "tenure",
    "MonthlyCharges"
]

datos_modelo = df[
    variables_modelo
].copy()

for columna in variables_modelo:

    datos_modelo[columna] = pd.to_numeric(
        datos_modelo[columna],
        errors="coerce"
    )

filas_originales = len(datos_modelo)

datos_modelo = datos_modelo.dropna(
    subset=variables_modelo
).copy()

filas_validas = len(datos_modelo)

print("Variables seleccionadas:")
print(variables_modelo)

print("\nFilas originales:", filas_originales)
print("Filas válidas:", filas_validas)
print("Filas excluidas:", filas_originales - filas_validas)

print("\nEstadísticas descriptivas:")

display(
    datos_modelo.describe().round(2)
)
# %%
plt.figure(figsize=(10, 6))

plt.scatter(
    datos_modelo["tenure"],
    datos_modelo["MonthlyCharges"],
    alpha=0.5
)

plt.title("Clientes según antigüedad y cargos mensuales")
plt.xlabel("Antigüedad (meses)")
plt.ylabel("Cargos mensuales")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# %%
scaler = StandardScaler()

datos_escalados = scaler.fit_transform(
    datos_modelo[
        ["tenure", "MonthlyCharges"]
    ]
)

datos_escalados_df = pd.DataFrame(
    datos_escalados,
    columns=["tenure", "MonthlyCharges"],
    index=datos_modelo.index
)

print("Datos originales:")

display(
    datos_modelo.head()
)

print("Datos estandarizados:")

display(
    datos_escalados_df.head()
)

print("Medias después del escalado:")

print(
    datos_escalados_df.mean().round(4)
)

print("Desviaciones estándar:")

print(
    datos_escalados_df.std(ddof=0).round(4)
)
# %%
inercias = []

valores_k = range(1, 9)

for k in valores_k:

    modelo_k = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    modelo_k.fit(
        datos_escalados
    )

    inercias.append(
        modelo_k.inertia_
    )


plt.figure(figsize=(9, 5))

plt.plot(
    list(valores_k),
    inercias,
    marker="o"
)

plt.title("Método del codo: Telco Customer Churn")
plt.xlabel("Cantidad de clústeres (K)")
plt.ylabel("Inercia")

plt.xticks(
    list(valores_k)
)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# %%
K_FINAL = 3

modelo = KMeans(
    n_clusters=K_FINAL,
    random_state=42,
    n_init=10
)

etiquetas = modelo.fit_predict(
    datos_escalados
)

resultado = df.loc[
    datos_modelo.index,
    [
        "customerID",
        "tenure",
        "MonthlyCharges",
        "Churn",
        "Contract",
        "InternetService"
    ]
].copy()

resultado["Cluster"] = etiquetas

print("Primeros clientes con su clúster:")

display(
    resultado.head(10)
)

print("Cantidad de clientes por clúster:")

display(
    resultado["Cluster"].value_counts().sort_index()
)
# %%
resumen_clusters = (
    resultado.groupby("Cluster")
    .agg(
        Clientes=("customerID", "count"),
        Antiguedad_promedio=("tenure", "mean"),
        Antiguedad_mediana=("tenure", "median"),
        Cargo_mensual_promedio=("MonthlyCharges", "mean"),
        Cargo_mensual_mediano=("MonthlyCharges", "median")
    )
    .round(2)
)

display(
    resumen_clusters
)
# %%
plt.figure(figsize=(10, 6))

grafico = plt.scatter(
    resultado["tenure"],
    resultado["MonthlyCharges"],
    c=resultado["Cluster"],
    cmap="viridis",
    alpha=0.45,
    s=22
)

plt.title("Segmentación de clientes por antigüedad y cargos")
plt.xlabel("Antigüedad (meses)")
plt.ylabel("Cargos mensuales")

plt.colorbar(
    grafico,
    label="Clúster"
)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# %%
centroides_originales = scaler.inverse_transform(
    modelo.cluster_centers_
)

centroides = pd.DataFrame(
    centroides_originales,
    columns=["tenure", "MonthlyCharges"]
)

centroides.insert(
    0,
    "Cluster",
    range(K_FINAL)
)

display(
    centroides.round(2)
)
# %%
resumen_churn = (
    resultado.groupby("Cluster")
    .agg(
        Clientes=("customerID", "count"),
        Abandonaron=(
            "Churn",
            lambda valores: (valores == "Yes").sum()
        ),
        Continuan=(
            "Churn",
            lambda valores: (valores == "No").sum()
        ),
        Tasa_abandono_pct=(
            "Churn",
            lambda valores: (valores == "Yes").mean() * 100
        )
    )
    .round(2)
)

display(
    resumen_churn
)
# %%
distribucion_contrato = pd.crosstab(
    resultado["Cluster"],
    resultado["Contract"],
    normalize="index"
) * 100

display(
    distribucion_contrato.round(2)
)
# %%
for k in [2, 3, 4]:

    modelo_experimento = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    etiquetas_experimento = (
        modelo_experimento.fit_predict(
            datos_escalados
        )
    )

    tabla_experimento = datos_modelo.copy()

    tabla_experimento["Cluster"] = etiquetas_experimento

    print(f"\nResultados para K = {k}")
    print(f"Inercia: {modelo_experimento.inertia_:.2f}")

    resumen_experimento = (
        tabla_experimento.groupby("Cluster")
        .agg(
            Clientes=("Cluster", "size"),
            Antiguedad_promedio=("tenure", "mean"),
            Cargo_mensual_promedio=("MonthlyCharges", "mean")
        )
        .round(2)
    )

    display(
        resumen_experimento
    )
# %%

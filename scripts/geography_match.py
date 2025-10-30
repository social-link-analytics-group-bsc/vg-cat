# scripts/process_heat_2024.py
import pandas as pd
from pathlib import Path

RAW_PATH = Path("../data/raw/df_heat_wave_indices.csv")          
OUT_PATH = Path("../data/processed/df_heat_2024.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 1) Cargar
if not RAW_PATH.exists():
    raise FileNotFoundError(f"No encuentro {RAW_PATH}. Ajusta la ruta en el script.")
df = pd.read_csv(RAW_PATH)

# 2) Filtrar 2024
df_2024 = df[df["int_year"] == 2024].copy()
if df_2024.empty:
    raise ValueError("El filtro int_year == 2024 no devolvió filas. Revisa los datos de entrada.")

# 3) Seleccionar columnas clave (sumas/promedios por cluster)
cols_exist = set(df_2024.columns)
cols_needed = {
    "cluster_id",
    "int_n_heat_waves",
    "int_days_in_heat_wave",
    "f_year_hw_tmax_vl",
    "f_year_hw_tmin_vl",
}
missing = cols_needed - cols_exist
if missing:
    raise KeyError(f"Faltan columnas en el CSV para 2024: {missing}")

df_2024 = df_2024[list(cols_needed)].copy()

# 4) Agregar por cluster (por si hubiera más de una fila por cluster)
agg = (
    df_2024.groupby("cluster_id", as_index=False)
    .agg(
        int_n_heat_waves=("int_n_heat_waves", "sum"),
        int_days_in_heat_wave=("int_days_in_heat_wave", "sum"),
        f_year_hw_tmax_vl=("f_year_hw_tmax_vl", "mean"),
        f_year_hw_tmin_vl=("f_year_hw_tmin_vl", "mean"),
    )
)

# 5) Guardar
agg.to_csv(OUT_PATH, index=False)
print(f"[OK] Guardado: {OUT_PATH}")
print("Filas:", len(agg))
print(agg.head(8).to_string(index=False))


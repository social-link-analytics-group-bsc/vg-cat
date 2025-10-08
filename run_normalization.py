from pathlib import Path
import pandas as pd
from preprocessor import DataPreprocessor  

CASE_RECORDS = Path("data/raw/siad_raw.csv")

# 2) Directorio de centros (SIAD/PIAD/SIE). Si ya tienes uno en Excel/CSV, pon la ruta aquí:
#    Si no, usa alguno de los ficheros que ya ves en tu carpeta (p. ej. los 'fqcc-7vme_*.xlsx').
#    Debe contener, al menos, nombre del centro y municipio/comarca.
SIAD_CENTERS = Path("fqcc-7vme_SIE_metadata.csv")  # <-- cambia esta ruta si tu fichero se llama distinto

# --- Salidas ---
OUTDIR = Path("data/processed")
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT_CASE = OUTDIR / "siad_normalized.csv"
OUT_CENTERS = OUTDIR / "centers_normalized.csv"

def read_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path.resolve()}")
    if path.suffix.lower() in [".csv", ".txt"]:
        return pd.read_csv(path, dtype=str, low_memory=False)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path, dtype=str)
    raise ValueError(f"Formato no soportado: {path.suffix}")

def main():
    df_cases = read_any(CASE_RECORDS)
    df_centers = read_any(SIAD_CENTERS)

    proc = DataPreprocessor()

    df_cases = proc.remove_blank_spaces(df_cases).fillna("No consta")
    df_centers = proc.remove_blank_spaces(df_centers).fillna("No consta")

    df_cases = proc.normalize_typos(df_cases, threshold=96)
    df_centers = proc.normalize_typos(df_centers, threshold=96)

    if "NOM DEL CENTRE" in df_centers.columns:
        df_centers["NOM DEL CENTRE"] = df_centers["NOM DEL CENTRE"].replace(
            {"InformaciÃ³ i atenciÃ³ a les dones deÂ Badia del VallÃ¨s":
             "InformaciÃ³ i atenciÃ³ a les dones (SIAD) de Badia del VallÃ¨s"}
        )

    df_cases.to_csv(OUT_CASE, index=False, encoding="utf-8-sig")
    df_centers.to_csv(OUT_CENTERS, index=False, encoding="utf-8-sig")
    print(f"OK → {OUT_CASE} ({len(df_cases):,} filas)")
    print(f"OK → {OUT_CENTERS} ({len(df_centers):,} filas)")

if __name__ == "__main__":
    main()
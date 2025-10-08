from pathlib import Path
import unicodedata as ud
import re
import pandas as pd
import matplotlib.pyplot as plt

CASES = Path("data/raw/siad_raw.csv")           
CENTERS = Path("data/raw/centers_raw.csv")      
OUTDIR = Path("outputs"); OUTDIR.mkdir(exist_ok=True)

def strip_accents(s: str) -> str:
    return "".join(c for c in ud.normalize("NFD", s) if ud.category(c) != "Mn")

def norm(s: str) -> str:
    if pd.isna(s): return ""
    s = strip_accents(str(s)).lower()
    s = re.sub(r"[\W_]+", " ", s, flags=re.U).strip()
    return s

def guess_municipi_from_siad(siad_text: str) -> str:
    if pd.isna(siad_text): 
        return None
    t = strip_accents(str(siad_text))

    # Patrones para extraer municipio desde el nombre del servicio
    patts = [
        r"Ajuntament d[e']\s+(.+)",
        r"Servei d'informacio i atencio a les dones.* de ([^-\(\)]+)",
        r"Servei d'Intervencio Especialitzada.* de ([^-\(\)]+)",
        r"Informacio i atencio a les dones de\s+(.+)",
        r"Punt d'informacio i atencio a les dones.* de ([^-\(\)]+)",
        # ➕ nuevos patrones:
        r"Oficina ICD de\s+(.+)",          # p.ej. "Oficina ICD de Barcelona"
        r"PIAD de\s+([^-–]+)",             # p.ej. "PIAD de Barcelona – Eixample" → Barcelona
        r".*-\s*([^-–]+)$",                # p.ej. "SIE del Baix Llobregat - Sant Feliu de Llobregat" → Sant Feliu de Llobregat
    ]

    for p in patts:
        m = re.search(p, t, flags=re.I)
        if m:
            return m.group(1).strip()
    return None

def read_cases_and_centers():
    df = pd.read_csv(CASES, dtype=str, low_memory=False)

    df.columns = [c.strip().lower() for c in df.columns]
    if "siad" not in df.columns:
        raise SystemExit("No encuentro la columna 'siad' en tus casos.")

    if not CENTERS.exists():
        raise SystemExit("No encuentro data/raw/centers_raw.csv. Descárgalo primero.")
    ct = pd.read_csv(CENTERS, dtype=str, low_memory=False)
    ct.columns = [c.strip().lower() for c in ct.columns]

    name_col = "nom_del_centre" if "nom_del_centre" in ct.columns else ("nom" if "nom" in ct.columns else None)
    municipi_col = "poblaci" if "poblaci" in ct.columns else ("municipi" if "municipi" in ct.columns else None)
    comarca_col = "comarca" if "comarca" in ct.columns else None
    if not all([name_col, municipi_col, comarca_col]):
        raise SystemExit("El directorio no tiene columnas esperadas (nom_del_centre/municipi/comarca). Revisa centers_raw.csv.")


    df["_key"] = df["siad"].map(norm)
    ct["_key"] = ct[name_col].map(norm)

    merged = df.merge(
        ct[["_key", municipi_col, comarca_col]],
        on="_key",
        how="left",
        suffixes=("", "_centers"),
    )

    
    miss = merged[municipi_col].isna()
    if miss.any():
        merged.loc[miss, municipi_col] = merged.loc[miss, "siad"].map(guess_municipi_from_siad)

    
    for col in [municipi_col, comarca_col]:
        merged[col] = merged[col].fillna("No consta").str.title()
    merged.rename(columns={municipi_col: "municipi", comarca_col: "comarca"}, inplace=True)
    return merged

def barh_top(df_counts, title, outpath, top=20):
    sub = df_counts.head(top).iloc[::-1]  # para que el top quede arriba
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(sub.index, sub.values)
    ax.set_title(title)
    ax.set_xlabel("Número de incidències")
    ax.set_ylabel("")
    
    for i, v in enumerate(sub.values):
        ax.text(v, i, f" {int(v):,}".replace(",", "."), va="center")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def main():
    df = read_cases_and_centers()

    
    por_muni = (
    df.loc[
        df["municipi"].notna() &
        (df["municipi"].str.strip().str.lower() != "no consta")
    ]
    .groupby("municipi")
    .size()
    .sort_values(ascending=False)
    )

    barh_top(
        por_muni,
        "Incidències per municipi (Top 20)",
        OUTDIR / "incidencies_por_municipio.png",
        top=20,
    )

    
    por_comarca = df.groupby("comarca", dropna=False).size().sort_values(ascending=False)
    barh_top(
        por_comarca,
        "Incidències per comarca",
        OUTDIR / "incidencies_por_comarca.png",
        top=len(por_comarca),  
    )

    print(f"✓ Guardado: {OUTDIR/'incidencies_por_municipio.png'}")
    print(f"✓ Guardado: {OUTDIR/'incidencies_por_comarca.png'}")

if __name__ == "__main__":
    main()
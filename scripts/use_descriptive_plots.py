from pathlib import Path
import sys, types, importlib.util
import pandas as pd
import re, unicodedata as ud

REPO = Path(__file__).parent

pkg = types.ModuleType("vgcat"); pkg.__path__ = [str(REPO)]
sys.modules["vgcat"] = pkg

def load_as(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod

load_as("vgcat.utils", REPO / "utils.py")            
desc = load_as("vgcat.descriptive", REPO / "descriptive.py")
DescriptiveAnalysis = desc.DescriptiveAnalysis

def strip_accents(s: str) -> str:
    return "".join(c for c in ud.normalize("NFD", str(s)) if ud.category(c) != "Mn")

def norm(s: str) -> str:
    s = strip_accents(str(s)).lower()
    s = re.sub(r"[\W_]+", " ", s).strip()
    return s

CASES = Path("data/processed/siad_normalized.csv")   
if not CASES.exists():
    CASES = Path("data/raw/siad_raw.csv")

CENTERS = Path("data/raw/centers_raw.csv")

cases = pd.read_csv(CASES, dtype=str, low_memory=False)
centers = pd.read_csv(CENTERS, dtype=str, low_memory=False)

cases.columns   = [c.strip() for c in cases.columns]
centers.columns = [c.strip().lower() for c in centers.columns]

rename_cases = {
    "siad": "SIAD", "comarca": "Comarca", "ambit": "Àmbit",
    "any": "Any", "sexe": "Sexe"
}
for k,v in rename_cases.items():
    if k in cases.columns and v not in cases.columns:
        cases.rename(columns={k:v}, inplace=True)

name_col   = "nom_del_centre" if "nom_del_centre" in centers.columns else ("nom" if "nom" in centers.columns else None)
municipi_c = "poblaci" if "poblaci" in centers.columns else ("municipi" if "municipi" in centers.columns else None)

if not (name_col and municipi_c and "Comarca" in cases.columns and "SIAD" in cases.columns):
    raise SystemExit("Faltan columnas esperadas: SIAD/Comarca en casos y nom_del_centre + municipi/poblaci en centers_raw.csv")

cases["_key"]   = cases["SIAD"].map(norm)
centers["_key"] = centers[name_col].map(norm)

m = cases.merge(centers[["_key", municipi_c]], on="_key", how="left")
m.rename(columns={municipi_c: "Municipi"}, inplace=True)
m["Municipi"] = m["Municipi"].fillna("No consta").str.title()
m["Comarca"]  = m["Comarca"].fillna("No consta").str.title()


eda = DescriptiveAnalysis()


municipi_counts = m["Municipi"].value_counts().head(20)
eda.plot_hist(municipi_counts, "Incidències per municipi (Top 20)", "./vg-cat/eda/hist")


comarca_counts = m["Comarca"].value_counts()
eda.plot_hist(comarca_counts, "Incidències per comarca", "./vg-cat/eda/hist")

print("✓ Plots guardados en ./vg-cat/eda/hist :")
print("  - hist_Incidències per municipi (Top 20).png")
print("  - hist_Incidències per comarca.png")
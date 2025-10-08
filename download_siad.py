from pathlib import Path
import os, time, requests, pandas as pd

DOMAIN = "https://analisi.transparenciacatalunya.cat"
DATASET_ID = "imk8-b6zj"
OUT_PATH = Path("data/raw/siad_raw.csv")

def fetch_socrata(domain: str, dataset_id: str, limit: int = 50000, app_token: str | None = None) -> pd.DataFrame:
    headers = {"X-App-Token": app_token} if app_token else {}
    rows, offset = [], 0
    while True:
        url = f"{domain}/resource/{dataset_id}.json?$limit={limit}&$offset={offset}"
        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()
        chunk = r.json()
        if not chunk: break
        rows.extend(chunk)
        offset += limit
        time.sleep(0.2)
    return pd.DataFrame.from_records(rows)

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    token = os.getenv("SOCRATA_APP_TOKEN") or None
    df = fetch_socrata(DOMAIN, DATASET_ID, app_token=token)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Guardado {len(df):,} filas en {OUT_PATH}")

if __name__ == "__main__":
    main()
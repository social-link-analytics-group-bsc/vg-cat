from pathlib import Path
import os, time, requests, pandas as pd

DOMAIN = 'https://analisi.transparenciacatalunya.cat'
DATASET_ID = 'fqcc-7vme'  # Directori serveis d'atenció a les dones
OUT_PATH = Path('data/raw/centers_raw.csv')

def fetch(domain, dataset, limit=50000, token=None):
    headers = {'X-App-Token': token} if token else {}
    rows = []
    offset = 0
    base = f'{domain}/resource/{dataset}.json'
    while True:
        params = {'$limit': limit, '$offset': offset}
        r = requests.get(base, headers=headers, params=params, timeout=60)
        r.raise_for_status()
        chunk = r.json()
        if not chunk:
            break
        rows.extend(chunk)
        offset += limit
        time.sleep(0.2)
    return pd.DataFrame.from_records(rows)

if __name__ == '__main__':
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = fetch(DOMAIN, DATASET_ID)
    df.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
    print(f'Guardado {len(df):,} filas en {OUT_PATH}')
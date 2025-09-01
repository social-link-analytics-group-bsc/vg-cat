import json
import requests
import pandas as pd
from urllib.parse import urlparse

# Definición de servicios para cada view URL
URLS = [
    {
        'url': 'https://analisi.transparenciacatalunya.cat/api/views/imk8-b6zj',
        'serviceType': 'SIAD'
    },
    {
        'url': 'https://analisi.transparenciacatalunya.cat/api/views/fqcc-7vme',
        'serviceType': 'SIE'
    }
]


def is_valid_view_url(url: str) -> bool:
    """
    Verifica que la URL corresponde a la API de views de Transparència Catalunya.
    """
    try:
        parsed = urlparse(url)
        return (
            parsed.scheme in ('http', 'https') and
            parsed.netloc == 'analisi.transparenciacatalunya.cat' and
            parsed.path.startswith('/api/views/')
        )
    except Exception:
        return False


def fetch_columns(view_url: str) -> list:
    """
    Descarga la lista de columnas (metadatos) desde la API REST de la vista.
    """
    if not is_valid_view_url(view_url):
        raise ValueError(f"URL no válida o no corresponde a un view de Transparència Catalunya: {view_url}")

    # Asegurar endpoint JSON
    if not view_url.endswith('.json'):
        view_url = view_url.rstrip('/') + '.json'

    resp = requests.get(view_url)
    resp.raise_for_status()
    data = resp.json()
    return data.get('columns', [])


def metadata_summary_to_dataframe(columns: list, service_type: str) -> pd.DataFrame:
    """
    
    """
    if service_type == "SIAD":
        records = []
        for col in columns:
            cached = col.get('cachedContents', {})
            top = cached.get('top', [])
            records.append({
                'name': col.get('name'),
                'fieldName': col.get('fieldName') or col.get('fieldiame'),
                'dataTypeName': col.get('dataTypeName'),
                'description': col.get('description'),
                'items': [item.get('item') for item in top if 'item' in item],
                'counts': [(item.get('item'), item.get('count'))
                        for item in top if 'item' in item and 'count' in item],
                'serviceType': service_type,
                'sourceURL': columns[0].get('viewUrl') if columns and 'viewUrl' in columns[0] else None
            })
        return pd.DataFrame(records)
    
    if service_type == "SIE":

        records = []
        for col in columns:
            cached = col.get('cachedContents', {})
            top = cached.get('top', [])
            records.append({
                'name': col.get('name'),
                'fieldName': col.get('fieldName') or col.get('fieldiame'),
                'dataTypeName': col.get('dataTypeName'),
                'description': col.get('description'),
                'items': [item.get('item') for item in top if 'item' in item],
                'counts': [(item.get('item'), item.get('count'))
                        for item in top if 'item' in item and 'count' in item],
                'serviceType': service_type,
                'sourceURL': columns[0].get('viewUrl') if columns and 'viewUrl' in columns[0] else None
            })
        return pd.DataFrame(records)


def metadata_summary_to_json(columns: list, service_type: str) -> list:
    """
    
    """
    records = []
    for col in columns:
        cached = col.get('cachedContents', {})
        top = cached.get('top', [])
        first = top[0] if top else {}
        records.append({
            'name': col.get('name'),
            'fieldName': col.get('fieldName') or col.get('fieldiame'),
            'dataTypeName': col.get('dataTypeName'),
            'description': col.get('description'),
            'items': {
                'item_name': [entry['item'] for entry in top],
                'item_count': [entry['count'] for entry in top],
                'null_count': cached.get('null')
            },
            'items': first.get('item'),
            'counts': first.get('count'),
            'serviceType': service_type,
            'sourceURL': columns[0].get('viewUrl') if columns and 'viewUrl' in columns[0] else None
        })
        
    return records
    

def main():
    for entry in URLS:
        url = entry['url']
        service_type = entry['serviceType']
        view_id = url.rstrip('/').split('/')[-1]

        print(f"Procesando: {view_id} ({service_type})")
        try:
            columns = fetch_columns(url)
            metadata_df = metadata_summary_to_dataframe(columns, service_type)
            metadata_json = metadata_summary_to_json(columns, service_type)

            csv_path = f"{view_id}_{service_type}_metadata.csv"
            json_path = f"{view_id}_{service_type}_metadata.json"

            metadata_df.to_csv(csv_path, index=False)
            print(f"  • CSV generado: {csv_path}")

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_json, f, ensure_ascii=False, indent=2)
            print(f"  • JSON generado: {json_path}\n")

        except Exception as e:
            print(f"  ❌ Error: {e}\n")


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import re
import time
import typing 
import pandas as pd
from pathlib import Path
from .utils import setup_logging
from rapidfuzz import process, fuzz
from unidecode import unidecode

def clean(s):
    if pd.isna(s): return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^\w\s\-']+", " ", s)   # solo letras/números
    s = re.sub(r"\s+", " ", s)
    return s

# --- Funciones de limpieza y normalización ---
def drop_prefixes(text: str) -> str:
    """Elimina prefijos institucionales (Ajuntament, Consell Comarcal, Oficina ICD, etc.)."""
    pattern = (
        r"^(?:"
        r"Ajuntament\s+d(?:e|el|ella|els|')\s*|"
        r"Consell\s+Comarcal\s+d(?:e|el|ella|els|')\s*|"
        r"Conselh\s+Generau\s+d(?:e|el|ella|els|')\s*|"
        r"Consorci\s+Benestar\s+Social\s+del\s+|"
        r"Oficina\s+ICD\s+d(?:e|el|ella|els|')\s*"
        r")"
    )
    return re.sub(pattern, '', text or '', flags=re.IGNORECASE).strip()


def normalize(text: str) -> str:
    """
    - Minúsculas
    - Quita acentos
    - Elimina artículos sueltos y puntuación
    - Colapsa espacios
    """
    s = unidecode((text or '').lower())
    s = re.sub(r"\b(el|la|l|els|les|de|del|d')\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()



def run ( siad: str, sie: str, analysis: str, outdir:str, verbose: bool ):
    """
    
    --------------------------------------------------------------------
        :param siad:       
        :param sie:  
        :param analysis:     
        :param verbose:     

    """


    if (verbose): 
        print(f"")


    # Read 

    # Table with cosults to SIADs/SIEs/PIADs
    siad_df = pd.read_csv(siad) 

    # Table with location information of the centers (change name)
    sie_df = pd.read_csv(sie)


    #---------------------------------------
    # Normalization
    #---------------------------------------

    # Corregir anotaciones erroneas
    normalizations = {
        "Ajuntament de Badia del Vallés": "Ajuntament de Badia del Vallès",
        "Consell Comarcal de lAnoia": "Consell Comarcal de l'Anoia"
    }

    # Aplica la normalización
    siad_df["SIAD"] = siad_df["SIAD"].replace(normalizations)

    # Comprobar la normalización
    # print(print(siad_df[siad_df["SIAD"].str.contains("Vall[eé]s|lAnoia", regex=True)]))



    sie_df["NOM DEL CENTRE"] = sie_df["NOM DEL CENTRE"].replace(
        {"Informació i atenció a les dones de Badia del Vallès": 
        "Informació i atenció a les dones (SIAD) de Badia del Vallès"}
    )

    #-----------------------------------
    # Descriptive analysis
    #-----------------------------------

    # First line
    print(siad_df.iloc[0])
    print(sie_df.iloc[0])

    # Total numbers by table
    print(siad_df.shape) # (707453, 32)
    print(sie_df.shape)  # (151, 16)

    """
    Solamente hay expedientes de SIAD en la tabla siad_df. 
    La tabla SIE incluye información de todas la unidades que existen: siad, sie, piad, etc.
    """

    # Número de expedientes abiertos en los SIAD/SIE
    print(siad_df["SIAD"].value_counts())
    #siad_df["SIAD"].value_counts().to_csv("total_expedient_counts.csv", header=True, sep="\t")
    """
    Ajuntament de l'Hospitalet de Llobregat    28883
    Consell Comarcal del Vallès Oriental       24110
    Ajuntament de Castelldefels                20396
    Ajuntament de Vilafranca del Penedès       18112
    Consell Comarcal de la Selva               17599
                                            ...
    Ajuntament de Badia del Vallés               408
    Consell Comarcal de l'Alt Urgell             364
    Ajuntament de Banyoles                       249
    Consell Comarcal de lAnoia                   202
    Ajuntament de Badia del Vallès               132

    ¿Es posible pintar estas regiones en un mapa e identificar patrones
    geográficos que muestren las regioens afectadas?
    """

    # Tipo de asistencias y derivaciones de los expedientes
    print(siad_df["Derivació a serveis atenció a les dones"].value_counts())
 
    """
    Derivació a serveis atenció a les dones
    No consta                               591028
    SIAD                                     82083
    Altres                                   18365
    Altres serveis d'atenció a les dones      9944
    Acolliment i recuperació                  5628
    Acolliment d'urgències                     281
    Acolliment durgències                      113
    SIE                                          9
    Substitutori de la llar                      2
    """

    # Sexo de los asistentes
    print(siad_df["Sexe"].value_counts())

    """
    Sexe
    Dones     691222
    Homes      15980
    Altres       251

    ¿Son hombres afectados? ¿Son familiares que asisten en nombre
    de la víctima por discapacidad o situación de vulnerabilidas?
    ¿Son personas trans mal tagueadas?
    """

    males_filter = siad_df['Sexe'] == "Homes"
    males_violence = siad_df[males_filter]

    femmales_filter = siad_df['Sexe'] == "Dones"
    females_violence = siad_df[femmales_filter]

    others_filter = siad_df['Sexe'] == "Altres"
    others_violence = siad_df[others_filter]

    print("Homes\n-----------------")
    print(males_violence['Derivació a serveis atenció a les dones'].value_counts())
    print("Dones\n-----------------")
    print(females_violence['Derivació a serveis atenció a les dones'].value_counts())
    print("Altres\n-----------------")
    print(others_violence['Derivació a serveis atenció a les dones'].value_counts())
    print("\n")

    """
    Homes
    -----------------
    Derivació a serveis atenció a les dones
    No consta                               13259
    SIAD                                     2345
    Altres                                    183
    Altres serveis d'atenció a les dones      109
    Acolliment i recuperació                   82
    Acolliment d'urgències                      1
    Acolliment durgències                       1
    Name: count, dtype: int64
    Dones
    -----------------
    Derivació a serveis atenció a les dones
    No consta                               577544
    SIAD                                     79723
    Altres                                   18172
    Altres serveis d'atenció a les dones      9834
    Acolliment i recuperació                  5546
    Acolliment d'urgències                     280
    Acolliment durgències                      112
    SIE                                          9
    Substitutori de la llar                      2
    Name: count, dtype: int64
    Altres
    -----------------
    Derivació a serveis atenció a les dones
    No consta                               225
    SIAD                                     15
    Altres                                   10
    Altres serveis d'atenció a les dones      1

    """

    print("Homes\n-----------------")
    print(males_violence['Detecció violència masclista'].value_counts())
    print("Dones\n-----------------")
    print(females_violence['Detecció violència masclista'].value_counts())
    print("Altres\n-----------------")
    print(others_violence['Detecció violència masclista'].value_counts())
    print("\n")
    
    """
    Homes
    -----------------
    Detecció violència masclista
    No           6973
    No consta    6019
    Sí           2988
    Name: count, dtype: int64
    Dones
    -----------------
    Detecció violència masclista
    Sí           252328
    No           236770
    No consta    202124
    Name: count, dtype: int64
    Altres
    -----------------
    Detecció violència masclista
    No consta    212
    No            25
    Sí            14

    """

    print("Homes\n-----------------")
    print(males_violence['Violència física '].value_counts())
    print("Dones\n-----------------")
    print(females_violence['Violència física '].value_counts())
    print("Altres\n-----------------")
    print(others_violence['Violència física '].value_counts())
    print("\n")

    """
    Homes
    -----------------
    Violència física
    No consta    9104
    No           6370
    Sí            506
    Name: count, dtype: int64
    Dones
    -----------------
    Violència física
    No consta    457652
    No           189944
    Sí            43626
    Name: count, dtype: int64
    Altres
    -----------------
    Violència física
    No consta    232
    Sí            10
    No             9

    """

    print("Homes\n-----------------")
    print(males_violence['Violència psicològica '].value_counts())
    print("Dones\n-----------------")
    print(females_violence['Violència psicològica '].value_counts())
    print("Altres\n-----------------")
    print(others_violence['Violència psicològica '].value_counts())
    print("\n")

    """
    Homes
    -----------------
    Violència psicològica
    No consta    8591
    No           6260
    Sí           1129
    Name: count, dtype: int64
    Dones
    -----------------
    Violència psicològica
    No consta    420296
    No           177834
    Sí            93092
    Name: count, dtype: int64
    Altres
    -----------------
    Violència psicològica
    No consta    230
    Sí            17
    No             4

    """

    print("Homes\n-----------------")
    print(males_violence['Violència sexual i abusos sexuals'].value_counts())
    print("Dones\n-----------------")
    print(females_violence['Violència sexual i abusos sexuals'].value_counts())
    print("Altres\n-----------------")
    print(others_violence['Violència sexual i abusos sexuals'].value_counts())
    print("\n")

    """
    Homes
    -----------------
    Violència sexual i abusos sexuals
    No consta    9285
    No           6534
    Sí            161
    Name: count, dtype: int64
    Dones
    -----------------
    Violència sexual i abusos sexuals
    No consta    472236
    No           208668
    Sí            10318
    Name: count, dtype: int64
    Altres
    -----------------
    Violència sexual i abusos sexuals
    No consta    232
    No            19

    """

    print("Homes\n-----------------")
    print(males_violence['Violència econòmica'].value_counts())
    print("Dones\n-----------------")
    print(females_violence['Violència econòmica'].value_counts())
    print("Altres\n-----------------")
    print(others_violence['Violència econòmica'].value_counts())
    print("\n")

    """
    
    Homes
    -----------------
    Violència econòmica
    No consta    15876
    No             101
    Sí               3
    Name: count, dtype: int64
    Dones
    -----------------
    Violència econòmica
    No consta    660574
    No            26804
    Sí             3844
    Name: count, dtype: int64
    Altres
    -----------------
    Violència econòmica
    No consta    247
    No             4

    """

    print("Homes\n-----------------")
    print(males_violence['Violència masclista en la parella'].value_counts())
    print("Dones\n-----------------")
    print(females_violence['Violència masclista en la parella'].value_counts())
    print("Altres\n-----------------")
    print(others_violence['Violència masclista en la parella'].value_counts())
    print("\n")

    """
    Homes
    -----------------
    Violència masclista en la parella
    No consta    15835
    Sí              98
    No              47
    Name: count, dtype: int64
    Dones
    -----------------
    Violència masclista en la parella
    No consta    627708
    Sí            37224
    No            26290
    Name: count, dtype: int64
    Altres
    -----------------
    Violència masclista en la parella
    No consta    232
    Sí            10
    No             9
    
    """

    # Tarragona tiene dos SIEs, pero el resto de regiones solo tiene un centro
    center = sie_df['NOM DEL CENTRE'] == "Servei d'Intervenció Especialitzada en violència masclista (SIE) de Tarragona"
    sie_center = sie_df[center]
    print(sie_center)


    # 
    derivation = siad_df['Derivació a serveis atenció a les dones'] == "SIE"
    siad_derivation = siad_df[derivation].T
    print(siad_derivation)

    """
    ¿Por qué en algunos casos se derivan mujeres al SIE, pero muchas no recogen 
    "Detecció de Violencia Maclista" o no consta ningún tipo de violencia? 
    
    """

    #---------------------------------------
    # Clean Region information
    #---------------------------------------
    """
    No disponemos de manera directa de una forma de vincular los expedientes con 
    los distintos centros, así que vamos a tratar de hacer una aproximación regional
    que permita 
    """

    # Apply normalization regions:
    siad_df['region_raw'] = siad_df['SIAD'].astype(str)
    siad_df['region_clean'] = (
        siad_df['region_raw']
        .map(drop_prefixes)
        .map(normalize)
        .replace('', pd.NA)
    )

    # 1. Crear mapping manual (clave = valor normalizado -> canonical)
    mapping = {
        'vic': 'vicdones siad osona',
        'osona': 'vicdones siad osona'
    }

    # 2. Aplicar mapping conservando el original
    siad_df['centre_canonical'] = siad_df['region_clean'].map(mapping)
    siad_df['centre_canonical'] = siad_df['centre_canonical'].fillna(siad_df['region_clean'])
   
    reg1 = siad_df['centre_canonical'].dropna().unique().tolist()
    
    # # Two possible aproches for the centers
    # # Apply normalization regions:
    # sie_df['titularitat_raw'] = sie_df['Titularitat'].astype(str)
    # sie_df['titularitat_clean'] = (
    #     sie_df['titularitat_raw']
    #     .map(drop_prefixes)
    #     .map(normalize)
    #     .replace('', pd.NA)
    # )
    # reg2 = sie_df['titularitat_clean'].dropna().unique().tolist()


    sie_df['nom_del_centre_raw'] = sie_df['NOM DEL CENTRE'].astype(str)

    _patrones = [
        r"servei d'informaci[oó] i atenci[oó] a les dones\s*\(?siad\)?\s*",   # variants SIAD
        r"servei d'atenci[oó]\, recuperaci[oó] i acollida\s*\(?sara\)?\s*", # SARA
        r"informaci[oó] i atenci[oó] a les dones\s*(?:de|d')\s*",          # Informació i atenció a les dones de / d'
        r"punt dona\s*(?:de|d')\s*",                                      # Punt Dona de
        r"servei d'informaci[oó] i atenci[oó]\s*",                        # otras variantes cortas
        r"oficina\s+icd\s*(?:de)?\s*",                                    # Oficina ICD
        r"punt d'informaci[oó] i atenci[oó] a les dones\s*\(piad\)\s*de\s*",
        # añade aquí más patrones si aparecen otros prefijos (PIAD, PIAD, etc.)
    ]

    combined_pattern = r"(?i)" + r"(?:{})".format("|".join(_patrones))  # (?i) -> case-insensitive

    sie_df['nom_del_centre_raw'] = (
        sie_df['nom_del_centre_raw']
        .str.replace("\xa0", " ", regex=False)              # NBSP -> espacio
        .str.replace(combined_pattern, "", regex=True)      # quitar prefijos/comunes
        .str.strip()
    )

    sie_df['nom_del_centre_clean'] = (
        sie_df['nom_del_centre_raw']
        .map(drop_prefixes)   # elimina prefijos institucionales
        .map(normalize)       # normaliza (quita acentos, artículos, puntuación)
        .replace('', pd.NA)
    )

    # # 5) lista única de regiones
    reg2 = sie_df['nom_del_centre_clean'].dropna().unique().tolist()

    # # Apply normalization regions:
    # sie_df['nom_del_centre_raw'] = sie_df['NOM DEL CENTRE']
    # sie_df['nom_del_centre_raw'] = sie_df['nom_del_centre_raw'] \
    #     .str.replace(r"Servei d'informació i atenció a les dones \(SIAD\) ", '', regex=True)

    # sie_df['nom_del_centre_clean'] = (
    #     sie_df['nom_del_centre_raw']
    #     .map(drop_prefixes)
    #     .map(normalize)
    #     .replace('', pd.NA)
    # )
    # reg3 = sie_df['nom_del_centre_clean'].dropna().unique().tolist()

    #---------------------------------------
    # Offices Types
    #---------------------------------------

    siad_offices  = sie_df[sie_df['NOM DEL CENTRE'].str.contains('SIAD', case=False, na=False)]
    other_offices = sie_df[~sie_df['NOM DEL CENTRE'].str.contains('SIAD', case=False, na=False)]

    sie_offices   = other_offices[other_offices['NOM DEL CENTRE'].str.contains('SIE', case=False, na=False)]
    other_offices = other_offices[~other_offices['NOM DEL CENTRE'].str.contains('SIE', case=False, na=False)]

    piad_offices  = other_offices[other_offices['NOM DEL CENTRE'].str.contains('PIAD', case=False, na=False)]
    other_offices = other_offices[~other_offices['NOM DEL CENTRE'].str.contains('PIAD', case=False, na=False)]

    print(siad_offices)
    print(sie_offices)
    print(piad_offices)
    print(other_offices)
    
    
    #---------------------------------------
    # Offices Types
    #---------------------------------------

    # 5) lista única de regiones
    reg2 = siad_offices['nom_del_centre_clean'].dropna().unique().tolist()

    # Emparejamiento fuzzy 
    threshold = 80  
    results = []
    for r1 in reg1:
        best_match, score, _ = process.extractOne(
            r1, 
            reg2,
            scorer=fuzz.token_set_ratio
        )
        results.append({
            'SIAD_region': r1,
            'Matched_region': best_match if score >= threshold else None,
            'Score': score
        })

    matches_df = pd.DataFrame(results)

    # Check percentages
    print(matches_df.loc[matches_df['Score']<100])
    """
        SIAD_region  Matched_region      Score
    38   pla durgell     pla urgell  95.238095
    53        lanoia          anoia  90.909091
    62  esparraguera   esparreguera  91.666667

    No se han escrito bien estos nombres, pero se preservan. 


          SIAD_region Matched_region      Score
    35      barcelona           None  70.588235
    38    pla durgell     pla urgell  95.238095
    53         lanoia          anoia  90.909091 -> Consell Comarcal de lAnoia / Consell Comarcal de l'Anoia  corregido
    62   esparraguera   esparreguera  91.666667
    75    terres ebre           None  63.636364 -> Consell Comarcal del Baix Ebre No está / No identifico alternativas
    103           vic           None  46.153846 -> "vicdones siad osona" corregido
    108  badia valles           None  76.923077 -> Ajuntament de Badia del Vallés / Ajuntament de Badia del Vallès corregido
    
    """

    # Coincidences
    print(f"Total regiones SIAD únicas: {len(reg1)}")
    print(f"Total regiones SIAD_offices únicas: {len(reg2)}")
    print(f"Emparejamientos con score ≥ {threshold}: {matches_df['Matched_region'].notna().sum()}")

    # # Guardar resultados a CSV (opcional)
    # # matches_df.to_csv('region_matches.csv', index=False)

    
    #---------------------------------------
    # 
    #---------------------------------------

    normalized_ids = siad_offices[['IDSIAD','nom_del_centre_clean']].to_dict(orient='index')
    # print(normalized_ids, len(normalized_ids))




    info_offices = pd.concat([siad_offices, piad_offices], ignore_index=True)
    print(info_offices)

    print(siad_df)


    mapping_excepciones = {
        'barcelona': 'barcelona',  # Mapearemos todos los distritos de Barcelona a 'barcelona'
        'pla durgell': 'pla urgell',
        'esparraguera': 'esparreguera',
        'terres ebre': 'terres de lebre'  # Ejemplo de posible nombre alternativo
    }

    info_offices['merge_key'] = info_offices['nom_del_centre_clean'].apply(
     lambda x: 'barcelona' if 'barcelona' in x else x
    )

    # En la tabla SIAD, aplicar el mapeo de excepciones
    siad_df['merge_key'] = siad_df['region_clean'].map(mapping_excepciones).fillna(siad_df['region_clean'])

    merged_df = pd.merge(
        siad_df,
        info_offices,
        left_on='merge_key',
        right_on='merge_key',
        how='left'
    )

    # Revisar matches de Barcelona
    print(merged_df[merged_df['merge_key'] == 'barcelona']['nom_del_centre_clean'].unique())

    # Revisar casos sin match
    print(merged_df[merged_df['nom_del_centre_clean'].isna()]['merge_key'].unique())

    # 
    print(merged_df.loc[merged_df['region_clean'] == 'barcelona', ['merge_key', 'region_clean', 'nom_del_centre_clean']])
        
    print(merged_df.loc[merged_df['region_clean'] == 'terres ebre', ['merge_key', 'region_clean', 'nom_del_centre_clean']])

    print(merged_df.loc[merged_df['region_clean'] == 'pla durgell', ['merge_key', 'region_clean', 'nom_del_centre_clean']])

    print(merged_df.loc[merged_df['region_clean'] == 'esparraguera', ['merge_key', 'region_clean', 'nom_del_centre_clean']])
    
    
    """
    En Barcelona no soy capaz de deinifir cuáles son los centros asociados
    a los expedientes de siad. Sospecho que se han asociado de forma aleatoria. 
    Comprobar.
    """











    # # --- Diccionario de correcciones manuales ---
    # manual_map = {
    #     "pla durgell": "pla d'urgell",
    #     "esparraguera": "esparreguera",
    #     "terres ebre": "terres de l'ebre",
    # }

    # # --- Prepara datos ---
    # siad_df['siad_norm'] = siad_df['region_clean'].map(clean).replace(manual_map)
    # info_offices['region_norm'] = info_offices['nom_del_centre_raw'].map(clean).replace(manual_map)

    # candidates = info_offices['region_norm'].unique()

    # # --- Función de matching ---
    # def best_match(key, candidates):
    #     if key in candidates:
    #         return key, 100, "exacto"
    #     for c in candidates:         # substring
    #         if key and key in c:
    #             return c, 95, "substring"
    #     match, score, _ = process.extractOne(key, candidates, scorer=fuzz.token_sort_ratio)
    #     return match, score, "fuzzy"

    # # --- Buscar matches ---
    # matches = []
    # for k in siad_df['siad_norm'].unique():
    #     best, score, method = best_match(k, candidates)
    #     status = "auto" if score >= 85 else ("revisar" if score >= 60 else "no_match")
    #     matches.append([k, best, score, method, status])

    # matches_df = pd.DataFrame(matches, columns=["siad_norm","matched","score","method","status"])

    # # --- Aplicar solo los matches "auto" ---
    # map_auto = dict(matches_df[matches_df.status=="auto"][["siad_norm","matched"]].values)
    # siad_df['matched_region'] = siad_df['siad_norm'].map(map_auto)

    # # --- Resultado ---
    # df_merged = siad_df.merge(info_offices, left_on="matched_region", right_on="region_norm", how="left")


    # print(df_merged)






    # # Save SIAD IDs associated to regions
    # normalized_ids = siad_offices[['IDSIAD','region_clean']].to_dict(orient='index')

    # # Emparejamiento fuzzy 
    # threshold = 80  
    # results = []
    # for r1 in reg1:
    #     best_match, score, _ = process.extractOne(
    #         r1, 
    #         reg2,
    #         scorer=fuzz.token_set_ratio
    #     )
    #     results.append({
    #         'SIAD_region': r1,
    #         'Matched_region': best_match if score >= threshold else None,
    #         'Score': score
    #     })

    # matches_df = pd.DataFrame(results)

    # # Coincidences
    # print(f"Total regiones SIAD únicas: {len(reg1)}")
    # print(f"Total regiones SIAD_offices únicas: {len(reg2)}")
    # print(f"Emparejamientos con score ≥ {threshold}: {matches_df['Matched_region'].notna().sum()}")

    # # Mostrar tabla ordenada por score descendente
    # print(matches_df.sort_values('Score', ascending=False).head(20))

    # # Guardar resultados a CSV (opcional)
    # # matches_df.to_csv('region_matches.csv', index=False)

    # print(matches_df.loc[matches_df['Score']<100] )
    # print(len(siad_df['SIAD'].unique().tolist()))
    # print(len(reg1))
    # print(piad_offices['NOM DEL CENTRE'].unique().tolist())
    # print([ r1 for r1 in reg1 if r1 not in matches_df['SIAD_region'].tolist() ])
    # # print(sie_df[sie_df['NOM DEL CENTRE'].str.contains('Vic', case=False, na=False)])
    # # print(siad_df[siad_df['SIAD'].str.contains('Vic', case=False, na=False)].head(1))





























#     pat_prefijos = (
#     r"^(?:"
#     r"Ajuntament\s+d(?:e|el|ella|els|')\s*|"
#     r"Consell\s+Comarcal\s+d(?:e|el|ella|els|')\s*|"
#     r"Conselh\s+Generau\s+d(?:e|el|ella|els|')\s*|"
#     r"Consorci\s+Benestar\s+Social\s+del\s+|"
#     r"Oficina\s+ICD\s+d(?:e|el|ella|els|')\s*"
#     r")"
# )

#     def clean_region_name(region):
#         if pd.isna(region):
#             return region
#         region = re.sub(r'\s+', ' ', region)
#         region = re.sub(r"\b([dl])\s+", r"\1'", region)
#         corrections = {
#             "l Prat de Llobregat": "El Prat de Llobregat",
#             "l Vendrell": "El Vendrell",
#             "l Masnou": "El Masnou",
#             "l'Hospitalet de Llobregat": "L'Hospitalet de Llobregat",
#             "Badia del Vallés": "Badia del Vallès",
#             "Pla dUrgell": "Pla d'Urgell"
#         }
#         return corrections.get(region.strip(), region.strip())

#     # Aplicar limpieza
#     siad_df['region'] = (
#         siad_df['SIAD']
#         .str.replace(pat_prefijos, '', regex=True)
#         .str.strip()
#         .replace('', pd.NA)
#         .map(clean_region_name)
#     )

#     print(siad_df['region'].unique().tolist())
#     print(siad_offices['region'].unique().tolist())



























    # pat_prefijos = (
    #     r"^(?:"
    #     #r"Consell Comarcal\s+de(?:l|la|les|l')\s+|"
    #     r"Ajuntament\s+d(?:e|el|')\s+"
    #     r")"
    # )

    # siad_df['region'] = (
    #     siad_df['SIAD']
    #     .str.replace(pat_prefijos, '', regex=True)  
    #     .str.strip()                                
    #     .replace('', pd.NA)                         
    # )

    # print(siad_df['region'].unique().tolist())

    # print(siad_df.columns)
    # print(sie_df.columns)


    # print(sie_df['NOM DEL CENTRE'].value_counts())

    # print(siad_df['SIAD'].value_counts())


    # print(sie_df[sie_df['NOM DEL CENTRE'].str.contains('SIAD', case=False, na=False)])
    # print(sie_df[~sie_df['NOM DEL CENTRE'].str.contains('SIAD', case=False, na=False)])
    
    # print(sie_df[~sie_df['NOM DEL CENTRE'].str.contains('SIAD', case=False, na=False)]['NOM DEL CENTRE'].value_counts())


    # pd.set_option('display.max_columns', None)

    # print(sie_df[sie_df['NOM DEL CENTRE'].str.contains('Tarragona', case=False, na=False)])
    # print(siad_df[siad_df['SIAD'].str.contains('Tarragona', case=False, na=False)].head(1))

    # print(sie_df[sie_df['NOM DEL CENTRE'].str.contains('Hospitalet', case=False, na=False)])
    # print(siad_df[siad_df['SIAD'].str.contains('Hospitalet', case=False, na=False)].head(1))

    


    pd.reset_option('display.max_columns')


def main():

    # CMD Arguments
    parser = argparse.ArgumentParser(
        description='Pattern analysis in gender violence data in Catalonya')

    parser.add_argument('-siad', '--siad', type=str, required=True,
                        help='.')
    
    parser.add_argument('-sie', '--sie', type=str,
                        help='.')

    parser.add_argument('-a', '--analysis', choices=['summary','corr','find_corr',
                                                     'pca','cluster','map'],
                        required=True, help='.')
    
    parser.add_argument('-od', '--outdir', type=str, default=".",
                        help='.')
    

    parser.add_argument('-v', '--verbose', type=bool, default=True,
                        help='Verbose mode.')

    args = parser.parse_args()


    # Arguments validation
    if args.siad is None:
        print("[ Input Error ] Provide at least one of the following arguments: --siad or -siad")
        sys.exit()


    if args.sie is None:
        print("[ Input Error ] Provide at least one of the following arguments: --sie or -sie")
        sys.exit()
    else:
        pass

    if args.analysis is None:
        print("[ Input Error ] Provide at least one of the following arguments: --analysis or -a")
        sys.exit()


    if args.outdir is None:
        print("[ Input Error ] Provide at least one of the following arguments: --outdir or -od")
        sys.exit()
    else:
        date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
        outdir = Path(os.path.join(args.outdir, date))
        outdir.mkdir(parents=True,exist_ok=True)


    if args.verbose is None:
        print("[ Input Error ] Provide with the arguments --verbose or -v")
        sys.exit()


    if (args.verbose):
        print("vg-cat")
        print("------------------------")
        print(f"SIAD: {args.siad}")
        print(f"SIE: {args.sie}")
        print(f"analysis: {args.analysis}")
        print(f"Output path: {args.outdir}\n\n")

    # Execution
    run ( args.siad, args.sie, args.analysis, outdir, args.verbose )

if __name__ == "__main__":
    main()
    print("""\nWork completed!\n""")


#   python -m vg-cat -siad ../data/datos_abiertos_nacionales/Atencions_dels_Serveis_i_oficines_d_informaci__i_atenci__a_les_dones_i_Oficines_de_l_Institut_Catal__de_les_Dones.csv -sie ../data/dades_obertes_catalans/Directori_dels_serveis_d\'informació_i_atenció_a_les_dones_i_d\'abordatge_de_les_violències_masclistes_20250806.csv  -a summary

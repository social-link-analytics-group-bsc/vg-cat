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
from .preprocessor import DataPreprocessor
from .agent import AgentAnalysis
from .automatic import AutomaticAnalysis
from .exploratory import ExploratoryDataAnalysis
from .geography_match import GeographyMatch

CONFIG_OUTCOMES = "./vg-cat/config/outcomes.yml" 
CONFIG_PREDICTORS = "./vg-cat/config/predictors.yml"

# Display all columns
pd.reset_option('display.max_columns')

def run ( case_records: str, siad_centers: str, run_mode: str , analysis: str, outdir:str, verbose: bool ):
    """
    
    --------------------------------------------------------------------
        :param case_records:       
        :param siad_centers:  
        :param run_mode: 
        :param analysis:     
        :param verbose:     

    """
    ##############################
    # Read Data
    ##############################

    # Table with cosults to SIADs/SIEs/PIADs
    case_records = pd.read_csv(case_records, dtype=str)
    # Table with location information of the centers (change name)
    siad_centers = pd.read_csv(siad_centers, dtype=str)

    ##############################
    # Preprocessing Data
    ##############################

    processor= DataPreprocessor()
    
    # print(processor.summarize_missing(case_records))
    # print(processor.summarize_missing(siad_centers))

    # Remove blank spaces at the begining and the end of the text
    no_blank_case_records = processor.remove_blank_spaces(case_records).fillna("No consta")
    no_blank_siad_centers = processor.remove_blank_spaces(siad_centers).fillna("No consta")

    # Normalize entries with typos
    normaliced_case_records = processor.normalize_typos(no_blank_case_records, 96)
    normaliced_siad_centers = processor.normalize_typos(no_blank_siad_centers, 96)

    # Manual specific correction
    normaliced_siad_centers["NOM DEL CENTRE"] = normaliced_siad_centers["NOM DEL CENTRE"].replace(
        {"Informació i atenció a les dones de Badia del Vallès": "Informació i atenció a les dones (SIAD) de Badia del Vallès"}
    )

    # print(processor.count_duplicates(case_records))
    # print(processor.count_duplicates(siad_centers))
    
    ##############################
    # Analysis Mode Selection
    ##############################

    if run_mode == "automatic":

        analysis = AutomaticAnalysis()
        analysis.run(normaliced_case_records, normaliced_siad_centers)

    if run_mode == "exploratory":
        
        eda = ExploratoryDataAnalysis(CONFIG_OUTCOMES, CONFIG_PREDICTORS)
        eda.run(normaliced_case_records)

        # geo_analysis = GeographyMatch()
        # geo_analysis.run(normaliced_case_records, normaliced_siad_centers)


    if run_mode == "agent":

        analysis = AgentAnalysis()
        analysis.run(normaliced_case_records, normaliced_siad_centers)
    


def main():

    # CMD Arguments
    parser = argparse.ArgumentParser(
        description='Pattern analysis in gender violence data in Catalonya')

    parser.add_argument('-case_records', '--case_records', type=str, required=True,
                        help='.')
    
    parser.add_argument('-siad_centers', '--siad_centers', type=str,
                        help='.')
    
    parser.add_argument('-rm', '--run_mode', choices=['automatic','exploratory','agent'],
                        type=str, help='.')

    parser.add_argument('-a', '--analysis', choices=['summary','corr','find_corr',
                                                     'pca','cluster','map'],
                        required=True, help='.')
    
    parser.add_argument('-od', '--outdir', type=str, default=".",
                        help='.')

    parser.add_argument('-v', '--verbose', type=bool, default=True,
                        help='Verbose mode.')

    args = parser.parse_args()



    # Arguments validation
    if args.case_records is None:
        print("[ Input Error ] Provide at least one of the following arguments: --siad or -siad")
        sys.exit()


    if args.siad_centers is None:
        print("[ Input Error ] Provide at least one of the following arguments: --sie or -sie")
        sys.exit()
    else:
        pass

    if args.run_mode is None:
        print("[ Input Error ] Provide at least one of the following arguments: --run_mode or -rm")
        sys.exit()

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
        print(f"case_records: {args.case_records}")
        print(f"siad_centers: {args.siad_centers}")
        print(f"run_mode: {args.run_mode}")
        print(f"analysis: {args.analysis}")
        print(f"Output path: {args.outdir}\n\n")
        

    # Execution
    run ( args.case_records, args.siad_centers, args.run_mode, args.analysis, outdir, args.verbose )

if __name__ == "__main__":
    main()
    print("""\nWork completed!\n""")


#   python -m vg-cat -case_records ../data/datos_abiertos_nacionales/Atencions_dels_Serveis_i_oficines_d_informaci__i_atenci__a_les_dones_i_Oficines_de_l_Institut_Catal__de_les_Dones.csv -siad_centers ../data/dades_obertes_catalans/Directori_dels_serveis_d\'informació_i_atenció_a_les_dones_i_d\'abordatge_de_les_violències_masclistes_20250806.csv  -a summary -rm exploratory

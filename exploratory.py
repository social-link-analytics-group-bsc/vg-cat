import pandas as pd
from .utils import load_config 
from .descriptive import DescriptiveAnalysis
from .correlations import CorrelationAnalysis


class ExploratoryDataAnalysis:

    def __init__(self, config_violences_path, config_demographics_path):
        self.config_violences = load_config(config_violences_path)
        self.config_demographics = load_config(config_demographics_path)


    def get_variables(self, config_type:str, name_variable:str ):
        """
        """

        if config_type == "outcomes":
            return [var[name_variable] for var in self.config_violences[config_type]]

        if config_type == "predictors":
            return [var[name_variable] for var in self.config_demographics[config_type]]


    def run(self, case_records: pd.DataFrame, siad_centers: pd.DataFrame) -> pd.DataFrame:
        """
        """
            
        # Get choosen variables to analyze
        print("Get Variables...\n")
        violences_variables = self.get_variables("outcomes", "name")
        demographics_variables = self.get_variables("predictors", "name")

        # -----------------------
        # Subworkflow
        # -----------------------

        # 1. Descriptive Analysis (plots)
        descriptive = DescriptiveAnalysis()
        descriptive.get_plots(case_records, violences_variables, demographics_variables)

        # 2. Find correlations
        correlations = CorrelationAnalysis()
        correlations.find_correlations(case_records, violences_variables, demographics_variables)

        # # 3. Subsample cases
        # sampling = SamplingRecords() 
        # subset_records = sampling.apply_filter(case_records, violences_variables, demographics_variables)


        # # 4. Find correlations
        # correlations = CorrelationAnalysis()
        # correlations.find_correlations(subset_records, violences_variables, demographics_variables)
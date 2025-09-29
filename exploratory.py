import pandas as pd
from .utils import load_config, print_title
from .descriptive import DescriptiveAnalysis
from .correlations import CorrelationAnalysis
from .sampling import SamplingRecords


class ExploratoryDataAnalysis:

    def __init__(self, config_violences_path, config_demographics_path):
        self.config_violences = load_config(config_violences_path)
        self.config_demographics = load_config(config_demographics_path)


    def get_variables(self, config_type:str, name_variable:str ):
        """
        Define variables to contrast in the analysis.
        """

        if config_type == "outcomes":
            return [var[name_variable] for var in self.config_violences[config_type]]

        if config_type == "predictors":
            return [var[name_variable] for var in self.config_demographics[config_type]]


    def run(self, case_records: pd.DataFrame) -> pd.DataFrame:
        """
        Run Descriptive analysis workflow 
        """
            
        # Get choosen variables to analyze
        violences_variables = self.get_variables("outcomes", "name")
        demographics_variables = self.get_variables("predictors", "name")

        # -----------------------
        # Subworkflow
        # -----------------------

        # 1. Descriptive Analysis 
        print_title("DESCRIPTIVE ANALYSIS - COMPLETE DATASET", level=1, style='fixed', emoji='🔍',width=80)
        descriptive = DescriptiveAnalysis()
        descriptive.get_descriptive_data(case_records, violences_variables, demographics_variables)

        # 2. Find correlations
        print_title("CORRELATION ANALYSIS - COMPLETE DATASET", level=1, style='fixed', emoji='📈',width=80)
        correlations = CorrelationAnalysis()
        correlations.find_correlations(case_records, violences_variables, demographics_variables, 'all_records')

        # 3. Subsample cases
        print_title("REPRESENTATIVE SUBSET CREATION", level=1, style='fixed', emoji='📊',width=80)
        sampling = SamplingRecords() 
        subset_records = sampling.apply_filter(case_records, violences_variables, demographics_variables)

        # # 4. Find correlations
        print_title("CORRELATION ANALYSIS - REPRESENTATIVE SUBSET", level=1, style='fixed', emoji='📈',width=80)
        correlations = CorrelationAnalysis()
        correlations.find_correlations(subset_records, violences_variables, demographics_variables, 'subset_records')
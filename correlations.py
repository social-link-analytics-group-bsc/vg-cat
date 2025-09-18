import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency, pointbiserialr

class CorrelationAnalysis:

    def remove_nans(self, df):
        """
        Replace 'No consta' with NaN but don't drop rows yet to preserve data for pairwise analysis
        """
        df = df.replace("No consta", np.nan)
        return df


    def apply_chi2(self, df, vars_subset1, vars_subset2):
        """
        Perform chi-squared tests between all pairs of variables from two subsets
        """
        os.makedirs("./vg-cat/eda/chi2", exist_ok=True)
        
        results = pd.DataFrame(index=vars_subset1, columns=vars_subset2)
        
        for var1 in vars_subset1:
            for var2 in vars_subset2:
                if var1 != var2:
                    # Create contingency table with only complete cases for these two variables
                    temp_df = df[[var1, var2]].dropna()
                    
                    if len(temp_df) == 0:
                        results.loc[var1, var2] = np.nan
                        continue
                        
                    contingency_table = pd.crosstab(temp_df[var1], temp_df[var2])
                    
                    # Check if table has at least 2 rows and 2 columns
                    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                        results.loc[var1, var2] = np.nan
                        continue
                    
                    try:
                        chi2, p_value, _, _ = chi2_contingency(contingency_table)
                        results.loc[var1, var2] = p_value
                    except:
                        results.loc[var1, var2] = np.nan
        
        # Convert to float for plotting
        results_float = results.astype(float)
        
        # Plot heatmap of p-values if we have valid data
        if not results_float.isnull().all().all():
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(results_float, dtype=bool), k=1)
            sns.heatmap(results_float, annot=True, cmap='viridis_r', 
                       center=0.05, mask=mask, cbar_kws={'label': 'p-value'})
            plt.title('Chi-Squared Test p-values')
            plt.tight_layout()
            plt.savefig('./vg-cat/eda/chi2/chi2_pvalues.png')
            plt.close()
        
        return results


    def apply_cramers(self, df, vars_subset1, vars_subset2):
        """
        Calculate Cramér's V for all pairs of variables from two subsets
        """
        os.makedirs("./vg-cat/eda/cramer", exist_ok=True)
        
        def cramers_v(contingency_table):
            try:
                chi2 = chi2_contingency(contingency_table)[0]
                n = contingency_table.sum().sum()
                phi2 = chi2 / n
                r, k = contingency_table.shape
                phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
                rcorr = r - ((r-1)**2)/(n-1)
                kcorr = k - ((k-1)**2)/(n-1)
                return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
            except:
                return np.nan
        
        results = pd.DataFrame(index=vars_subset1, columns=vars_subset2)
        
        for var1 in vars_subset1:
            for var2 in vars_subset2:
                if var1 != var2:
                    # Create contingency table with only complete cases for these two variables
                    temp_df = df[[var1, var2]].dropna()
                    
                    if len(temp_df) == 0:
                        results.loc[var1, var2] = np.nan
                        continue
                        
                    contingency_table = pd.crosstab(temp_df[var1], temp_df[var2])
                    
                    # Check if table has at least 2 rows and 2 columns
                    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                        results.loc[var1, var2] = np.nan
                        continue
                    
                    results.loc[var1, var2] = cramers_v(contingency_table)
        
        # Convert to float for plotting
        results_float = results.astype(float)
        
        # Plot heatmap of Cramér's V values if we have valid data
        if not results_float.isnull().all().all():
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(results_float, dtype=bool), k=1)
            sns.heatmap(results_float, annot=True, cmap='viridis', 
                       center=0.5, mask=mask, cbar_kws={'label': "Cramér's V"})
            plt.title("Cramér's V Correlation Matrix")
            plt.tight_layout()
            plt.savefig('./vg-cat/eda/cramer/cramers_v.png')
            plt.close()
        
        return results


    def apply_standarized_residues(self, df, vars_subset1, vars_subset2):
        """
        Calculate standardized residuals for all pairs of variables
        """
        os.makedirs("./vg-cat/eda/residues", exist_ok=True)
        
        results = {}
        
        for var1 in vars_subset1:
            for var2 in vars_subset2:
                if var1 != var2:
                    # Create contingency table with only complete cases for these two variables
                    temp_df = df[[var1, var2]].dropna()
                    
                    if len(temp_df) == 0:
                        results[f'{var1}_{var2}'] = None
                        continue
                        
                    contingency_table = pd.crosstab(temp_df[var1], temp_df[var2])
                    
                    # Check if table has at least 2 rows and 2 columns
                    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                        results[f'{var1}_{var2}'] = None
                        continue
                    
                    try:
                        chi2, p, dof, expected = chi2_contingency(contingency_table)
                        standardized_residuals = (contingency_table - expected) / np.sqrt(expected)
                        
                        # Plot heatmap of standardized residuals
                        plt.figure(figsize=(10, 8))
                        sns.heatmap(standardized_residuals, annot=True, cmap='RdBu_r', 
                                   center=0, fmt='.2f',
                                   cbar_kws={'label': 'Standardized Residual'})
                        plt.title(f'Standardized Residuals: {var1} vs {var2}')
                        plt.tight_layout()
                        plt.savefig(f'./vg-cat/eda/residues/{var1}_{var2}_residuals.png')
                        plt.close()
                        
                        results[f'{var1}_{var2}'] = standardized_residuals
                    except:
                        results[f'{var1}_{var2}'] = None
        
        return results


    def summary_results(self, pvals_chi2, pvals_cramer, pvals_residues):
        """
        Print summary of the correlation analysis results
        """
        print("SUMMARY")
        print("------------------------")
        
        # Convert to float for easier handling
        chi2_float = pvals_chi2.astype(float)
        cramer_float = pvals_cramer.astype(float)
        
        # Find most significant correlations (lowest p-values)
        if not chi2_float.isnull().all().all():
            chi2_flat = chi2_float.stack()
            valid_chi2 = chi2_flat.dropna()
            
            if len(valid_chi2) > 0:
                print("Top 5 most significant associations (chi-squared):")
                for (var1, var2), p_value in valid_chi2.nsmallest(5).items():
                    # Handle extremely small p-values
                    if p_value < 1e-100:
                        # For very small values, use more precise formatting
                        p_str = f"{p_value:.3e}"
                        # Extract the exponent part
                        exponent = int(p_str.split('e')[-1])
                        p_str = f"< 1e{exponent}"  # Show as less than the next order of magnitude
                    else:
                        p_str = f"{p_value:.2e}"
                    print(f"{var1} - {var2}: p = {p_str}")
            else:
                print("No valid chi-squared results.")
        else:
            print("No valid chi-squared results.")
        
        # Find strongest associations (highest Cramér's V)
        if not cramer_float.isnull().all().all():
            cramer_flat = cramer_float.stack()
            valid_cramer = cramer_flat.dropna()
            
            if len(valid_cramer) > 0:
                print("\nTop 5 strongest associations (Cramér's V):")
                for (var1, var2), cramer_v in valid_cramer.nlargest(5).items():
                    print(f"{var1} - {var2}: V = {cramer_v:.4f}")
            else:
                print("\nNo valid Cramér's V results.")
        else:
            print("\nNo valid Cramér's V results.")


    def find_correlations(self, case_records: pd.DataFrame, vars_subset1: list, vars_subset2: list) -> pd.DataFrame:
        """
        Main method to find correlations between two sets of categorical variables
        """
        # Clean the data (don't drop rows yet)
        cleaned_df = self.remove_nans(case_records)
        
        # Perform different correlation analyses
        pvals_chi2 = self.apply_chi2(cleaned_df, vars_subset1, vars_subset2)
        pvals_cramer = self.apply_cramers(cleaned_df, vars_subset1, vars_subset2)
        pvals_residues = self.apply_standarized_residues(cleaned_df, vars_subset1, vars_subset2)
        
        # Generate summary
        self.summary_results(pvals_chi2, pvals_cramer, pvals_residues)
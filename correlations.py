import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .utils import print_title
from scipy.stats import chi2_contingency, pointbiserialr

class CorrelationAnalysis:

    def remove_nans(self, df):
        """
        Replace specific string placeholders with NaN for further processing.
        """
        df = df.replace("No consta", np.nan)
        return df

    def apply_chi2(self, df, vars_subset1, vars_subset2, execution_type):
        """
        Perform chi-squared tests for each pair of variables between two subsets.

        - Creates directories to save outputs.
        - Builds contingency tables using complete cases for each pair.
        - Skips pairs that do not have at least 2 rows and 2 columns.
        - Returns DataFrames with p-values, chi2 statistics and degrees of freedom.
        """
        os.makedirs("./vg-cat/eda/chi2", exist_ok=True)
        
        # DataFrames to store results
        pvalues_df = pd.DataFrame(index=vars_subset1, columns=vars_subset2)
        chi2_stats_df = pd.DataFrame(index=vars_subset1, columns=vars_subset2)
        dof_df = pd.DataFrame(index=vars_subset1, columns=vars_subset2)
        
        for var1 in vars_subset1:
            for var2 in vars_subset2:
                if var1 != var2:
                    # Create contingency table with only complete cases for these two variables
                    temp_df = df[[var1, var2]].dropna()
                    
                    if len(temp_df) == 0:
                        pvalues_df.loc[var1, var2] = np.nan
                        chi2_stats_df.loc[var1, var2] = np.nan
                        dof_df.loc[var1, var2] = np.nan
                        continue
                        
                    contingency_table = pd.crosstab(temp_df[var1], temp_df[var2])
                    
                    # Check if table has at least 2 rows and 2 columns
                    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                        pvalues_df.loc[var1, var2] = np.nan
                        chi2_stats_df.loc[var1, var2] = np.nan
                        dof_df.loc[var1, var2] = np.nan
                        continue
                    
                    try:
                        chi2, p_value, dof, _ = chi2_contingency(contingency_table)
                        pvalues_df.loc[var1, var2] = p_value
                        chi2_stats_df.loc[var1, var2] = chi2
                        dof_df.loc[var1, var2] = dof
                    except:
                        pvalues_df.loc[var1, var2] = np.nan
                        chi2_stats_df.loc[var1, var2] = np.nan
                        dof_df.loc[var1, var2] = np.nan
        
        # Convert to float for plotting
        pvalues_float = pvalues_df.astype(float)
        
        # Plot heatmap of p-values if we have valid data
        if not pvalues_float.isnull().all().all():
            # Create a larger figure
            plt.figure(figsize=(20, 18))
            
            # Use mask to show only the upper triangle
            mask = np.triu(np.ones_like(pvalues_float, dtype=bool), k=1)
            
            # Format annotations for very small values
            annot_matrix = pvalues_float.copy()
            for i in range(annot_matrix.shape[0]):
                for j in range(annot_matrix.shape[1]):
                    if pd.notna(annot_matrix.iloc[i, j]):
                        if annot_matrix.iloc[i, j] < 0.001:
                            annot_matrix.iloc[i, j] = f'{annot_matrix.iloc[i, j]:.2e}'
                        else:
                            annot_matrix.iloc[i, j] = f'{annot_matrix.iloc[i, j]:.4f}'
                    else:
                        annot_matrix.iloc[i, j] = ''
            
            # Create heatmap
            heatmap = sns.heatmap(pvalues_float, 
                                annot=annot_matrix, 
                                fmt='', 
                                cmap='viridis_r', 
                                center=0.05, 
                                mask=mask, 
                                cbar_kws={'label': 'p-value'},
                                annot_kws={'size': 8})
            
            # Rotate labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            plt.title('Chi-Squared Test p-values\n(Los valores muestran p-valores, valores < 0.05 indican significancia)', 
                     fontsize=14, pad=20)
            plt.tight_layout()
            plt.savefig(f'./vg-cat/eda/chi2/chi2_pvalues_{execution_type}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        return pvalues_df, chi2_stats_df, dof_df

    def apply_cramers(self, df, vars_subset1, vars_subset2, execution_type):
        """
        Calculate Cramér's V for all pairs of variables from two subsets.

        - Uses bias correction for small samples.
        - Returns a DataFrame with Cramér's V values.
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
            # Create a larger figure
            plt.figure(figsize=(20, 18))
            
            # Use mask to show only the upper triangle
            mask = np.triu(np.ones_like(results_float, dtype=bool), k=1)
            
            # Format annotations with 2 decimals
            annot_matrix = results_float.round(2).astype(str)
            annot_matrix[results_float.isnull()] = ''
            
            # Create heatmap
            heatmap = sns.heatmap(results_float, 
                                annot=annot_matrix, 
                                fmt='', 
                                cmap='viridis', 
                                center=0.5, 
                                mask=mask, 
                                cbar_kws={'label': "Cramér's V"},
                                annot_kws={'size': 8})
            
            # Rotate labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            plt.title("Cramér's V Correlation Matrix\n(Valores más altos indican asociación más fuerte)", 
                     fontsize=14, pad=20)
            plt.tight_layout()
            plt.savefig(f'./vg-cat/eda/cramer/cramers_v_{execution_type}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        return results

    def interpret_cramers_v(self, v_value):
        """
        Provide a textual interpretation for a given Cramér's V value.
        """
        if pd.isna(v_value):
            return "No calculable"
        elif v_value < 0.1:
            return "Asociación muy débil o nula"
        elif v_value < 0.3:
            return "Asociación débil"
        elif v_value < 0.5:
            return "Asociación moderada"
        else:
            return "Asociación fuerte"

    def format_p_value(self, p_value):
        """
        Format p-value for better readability and add significance markers.
        """
        if pd.isna(p_value):
            return "No calculable"
        elif p_value < 0.001:
            return f"{p_value:.2e} ***"
        elif p_value < 0.01:
            return f"{p_value:.4f} **"
        elif p_value < 0.05:
            return f"{p_value:.4f} *"
        else:
            return f"{p_value:.4f} (no significativo)"

    def create_correlation_tables(self, pvals_chi2, chi2_stats, dof_chi2, pvals_cramer, execution_type):
        """
        Build comprehensive tables from chi2 and Cramér's V results and save CSV outputs.

        - Produces detailed long-format table with p-values, chi2 stats, dof and Cramér's V.
        - Generates filtered tables for significant and strong correlations.
        - Calls create_variable_summary_table to build per-variable summaries.
        """
        os.makedirs("./vg-cat/eda/tablas", exist_ok=True)
        
        # Convert to float for easier handling
        chi2_float = pvals_chi2.astype(float)
        chi2_stats_float = chi2_stats.astype(float)
        dof_float = dof_chi2.astype(float)
        cramer_float = pvals_cramer.astype(float)
        
        # Create long format table with all pairs
        correlation_data = []
        
        for var1 in chi2_float.index:
            for var2 in chi2_float.columns:
                if var1 != var2:
                    p_value = chi2_float.loc[var1, var2]
                    chi2_stat = chi2_stats_float.loc[var1, var2]
                    dof = dof_float.loc[var1, var2]
                    cramer_v = cramer_float.loc[var1, var2] if var1 in cramer_float.index and var2 in cramer_float.columns else np.nan
                    
                    # Determine significance and strength
                    significativo = "Sí" if pd.notna(p_value) and p_value < 0.05 else "No"
                    fuerza = self.interpret_cramers_v(cramer_v)
                    
                    # Categorize relationship strength
                    if pd.isna(cramer_v):
                        categoria_fuerza = "No calculable"
                    elif cramer_v < 0.1:
                        categoria_fuerza = "Muy débil/Nula"
                    elif cramer_v < 0.3:
                        categoria_fuerza = "Débil"
                    elif cramer_v < 0.5:
                        categoria_fuerza = "Moderada"
                    else:
                        categoria_fuerza = "Fuerte"
                    
                    # Detailed significance level
                    if pd.isna(p_value):
                        nivel_significancia = "No calculable"
                    elif p_value < 0.001:
                        nivel_significancia = "Muy significativo (p < 0.001)"
                    elif p_value < 0.01:
                        nivel_significancia = "Altamente significativo (p < 0.01)"
                    elif p_value < 0.05:
                        nivel_significancia = "Significativo (p < 0.05)"
                    else:
                        nivel_significancia = "No significativo"
                    
                    correlation_data.append({
                        'Variable_1': var1,
                        'Variable_2': var2,
                        'chi2_estadistico': chi2_stat,
                        'chi2_grados_libertad': dof,
                        'p_value': p_value,
                        'cramers_v': cramer_v,
                        'significativo': significativo,
                        'nivel_significancia': nivel_significancia,
                        'fuerza_asociacion': fuerza,
                        'categoria_fuerza': categoria_fuerza
                    })
        
        # Create main correlation table
        correlation_df = pd.DataFrame(correlation_data)
        
        # Sort by significance and strength
        correlation_df = correlation_df.sort_values(by=['p_value', 'cramers_v'], 
                                                  ascending=[True, False])
        
        # Save main table
        correlation_df.to_csv(f'./vg-cat/eda/tablas/correlaciones_completas_{execution_type}.csv', 
                             index=False, encoding='utf-8-sig')
        
        # Create significant correlations table
        significant_df = correlation_df[correlation_df['significativo'] == 'Sí'].copy()
        significant_df.to_csv(f'./vg-cat/eda/tablas/correlaciones_significativas_{execution_type}.csv', 
                             index=False, encoding='utf-8-sig')
        
        # Create strong correlations table (moderate or strong)
        strong_df = correlation_df[correlation_df['categoria_fuerza'].isin(['Moderada', 'Fuerte'])].copy()
        strong_df.to_csv(f'./vg-cat/eda/tablas/correlaciones_fuertes_{execution_type}.csv', 
                        index=False, encoding='utf-8-sig')
        
        # TABLA: Chi2 y p-values
        chi2_pvalues_df = correlation_df[['Variable_1', 'Variable_2', 'chi2_estadistico', 
                                        'chi2_grados_libertad', 'p_value', 'significativo', 
                                        'nivel_significancia']].copy()
        chi2_pvalues_df = chi2_pvalues_df.sort_values('p_value', ascending=True)
        chi2_pvalues_df.to_csv(f'./vg-cat/eda/tablas/chi2_pvalues_significancia_{execution_type}.csv', 
                             index=False, encoding='utf-8-sig')
        
        # TABLA: Cramér's V
        cramer_df = correlation_df[['Variable_1', 'Variable_2', 'cramers_v', 'fuerza_asociacion', 
                                  'categoria_fuerza']].copy()
        cramer_df = cramer_df.sort_values('cramers_v', ascending=False)
        cramer_df.to_csv(f'./vg-cat/eda/tablas/cramers_v_correlaciones_{execution_type}.csv', 
                       index=False, encoding='utf-8-sig')
        
        # Create summary table by variable
        self.create_variable_summary_table(correlation_df, execution_type)
        
        return correlation_df, significant_df, strong_df, chi2_pvalues_df, cramer_df

    def create_variable_summary_table(self, correlation_df, execution_type):
        """
        Build summary tables grouped by each variable (both perspectives).

        - Counts how many significant associations each variable has.
        - Computes mean and max of Cramér's V per variable.
        - Counts moderate/strong associations per variable.
        - Saves a combined CSV with both 'Variable_1' and 'Variable_2' perspectives.
        """
        # Summary for Variable 1
        var1_summary = correlation_df.groupby('Variable_1').agg({
            'p_value': lambda x: (pd.notna(x) & (x < 0.05)).sum(),
            'cramers_v': ['mean', 'max'],
            'categoria_fuerza': lambda x: (x.isin(['Moderada', 'Fuerte'])).sum()
        }).round(4)
        
        var1_summary.columns = ['num_significativas', 'cramers_v_promedio', 'cramers_v_maximo', 'num_fuertes_moderadas']
        var1_summary = var1_summary.sort_values('num_significativas', ascending=False)
        
        # Summary for Variable 2
        var2_summary = correlation_df.groupby('Variable_2').agg({
            'p_value': lambda x: (pd.notna(x) & (x < 0.05)).sum(),
            'cramers_v': ['mean', 'max'],
            'categoria_fuerza': lambda x: (x.isin(['Moderada', 'Fuerte'])).sum()
        }).round(4)
        
        var2_summary.columns = ['num_significativas', 'cramers_v_promedio', 'cramers_v_maximo', 'num_fuertes_moderadas']
        var2_summary = var2_summary.sort_values('num_significativas', ascending=False)
        
        # Combine both perspectives
        combined_summary = pd.concat([var1_summary, var2_summary], axis=1, 
                                   keys=['Como_Variable_1', 'Como_Variable_2'])
        
        combined_summary.to_csv(f'./vg-cat/eda/tablas/resumen_por_variable_{execution_type}.csv', 
                              encoding='utf-8-sig')
        
        return combined_summary

    def summary_results(self, pvals_chi2, pvals_cramer, pvals_residues, correlation_df=None, 
                       chi2_pvalues_df=None, cramer_df=None, execution_type=""):
        """
        Print a human-readable summary of the correlation analysis.

        - Interprets Cramér's V bins.
        - Reports counts of analyzed associations and significance breakdown.
        - Shows top associations by lowest p-values and provides formatted outputs.
        - Lists saved files and prints small previews of result tables if present.
        """
        
        # Convert to float for easier handling
        chi2_float = pvals_chi2.astype(float)
        cramer_float = pvals_cramer.astype(float)
        
        # INTERPRETACIÓN DE CRAMÉR'S V
        print("\n📊 INTERPRETACIÓN DE CRAMÉR'S V:")
        print("-" * 50)
        print("0.00 - 0.10: Asociación muy débil o nula")
        print("0.10 - 0.30: Asociación débil")
        print("0.30 - 0.50: Asociación moderada")
        print("0.50 - 1.00: Asociación fuerte")
        
        # Find most significant correlations (lowest p-values)
        if not chi2_float.isnull().all().all():
            chi2_flat = chi2_float.stack()
            valid_chi2 = chi2_flat.dropna()
            
            if len(valid_chi2) > 0:
                # Separate into significant and non-significant
                significant = valid_chi2[valid_chi2 < 0.05]
                non_significant = valid_chi2[valid_chi2 >= 0.05]
                
                print(f"\n🔍 RESUMEN ESTADÍSTICO:")
                print("-" * 30)
                print(f"Total de asociaciones analizadas: {len(valid_chi2)}")
                print(f"Asociaciones significativas (p < 0.05): {len(significant)}")
                print(f"Asociaciones no significativas: {len(non_significant)}")
                print(f"Porcentaje de asociaciones significativas: {len(significant)/len(valid_chi2)*100:.1f}%")
                
                # Additional info if we have the correlation dataframe
                if correlation_df is not None:
                    strong_correlations = len(correlation_df[
                        (correlation_df['significativo'] == 'Sí') & 
                        (correlation_df['categoria_fuerza'].isin(['Moderada', 'Fuerte']))
                    ])
                    print(f"Asociaciones significativas y fuertes/moderadas: {strong_correlations}")
                
                print(f"\n⭐ TOP 10 ASOCIACIONES MÁS SIGNIFICATIVAS (p-valores más bajos):")
                print("-" * 60)
                for (var1, var2), p_value in valid_chi2.nsmallest(10).items():
                    cramer_v = cramer_float.loc[var1, var2] if pd.notna(cramer_float.loc[var1, var2]) else np.nan
                    interpretation = self.interpret_cramers_v(cramer_v) if pd.notna(cramer_v) else "No calculable"
                    
                    print(f"{var1} ↔ {var2}")
                    print(f"   p-value: {self.format_p_value(p_value)}")
                    print(f"   Cramér's V: {cramer_v:.4f} - {interpretation}")
                    print()
            else:
                print("\n❌ No hay resultados válidos de chi-cuadrado.")
        else:
            print("\n❌ No hay resultados válidos de chi-cuadrado.")
        
        # Information about saved files
        print("\n💾 ARCHIVOS GUARDADOS:")
        print("-" * 30)
        print(f"• ./vg-cat/eda/tablas/correlaciones_completas_{execution_type}.csv - Todas las correlaciones")
        print(f"• ./vg-cat/eda/tablas/correlaciones_significativas_{execution_type}.csv - Correlaciones significativas (p < 0.05)")
        print(f"• ./vg-cat/eda/tablas/correlaciones_fuertes_{execution_type}.csv - Correlaciones moderadas/fuertes")
        print(f"• ./vg-cat/eda/tablas/chi2_pvalues_significancia_{execution_type}.csv - Resultados de Chi-cuadrado y p-values")
        print(f"• ./vg-cat/eda/tablas/cramers_v_correlaciones_{execution_type}.csv - Resultados de Cramér's V")
        print(f"• ./vg-cat/eda/tablas/resumen_por_variable_{execution_type}.csv - Resumen por variable")

        # Show preview of the tables
        if chi2_pvalues_df is not None:
            print(f"\n📋 PREVIEW TABLA CHI2 Y P-VALUES (primeras 5 filas):")
            print("-" * 60)
            print(chi2_pvalues_df.head().to_string(index=False))
        
        if cramer_df is not None:
            print(f"\n📊 PREVIEW TABLA CRAMÉR'S V (primeras 5 filas):")
            print("-" * 50)
            print(cramer_df.head().to_string(index=False))

    def apply_standarized_residues(self, df, vars_subset1, vars_subset2, execution_type):
        """
        Calculate standardized residuals for important variable pairs and save heatmaps.

        - Limits analysis to a subset of variable pairs to avoid generating too many plots.
        - Identifies important pairs by testing significance with chi-squared and selecting p < 0.05.
        - Saves heatmaps of standardized residuals for pairs with significant association.
        - Returns a dict with standardized residual DataFrames (or None on failure).
        """
        os.makedirs("./vg-cat/eda/residues", exist_ok=True)
        
        results = {}
        
        # Only compute residuals for the most important associations to avoid too many charts
        important_pairs = []
        
        # First identify significant associations
        for var1 in vars_subset1[:10]:  # Limit to first 10 variables to avoid overload
            for var2 in vars_subset2[:10]:
                if var1 != var2:
                    temp_df = df[[var1, var2]].dropna()
                    if len(temp_df) > 0:
                        contingency_table = pd.crosstab(temp_df[var1], temp_df[var2])
                        if contingency_table.shape[0] >= 2 and contingency_table.shape[1] >= 2:
                            try:
                                chi2, p, dof, expected = chi2_contingency(contingency_table)
                                if p < 0.05:  # Only for significant associations
                                    important_pairs.append((var1, var2))
                            except:
                                pass
        
        # Compute residuals only for important pairs
        for var1, var2 in important_pairs[:20]:  # Max 20 charts
            temp_df = df[[var1, var2]].dropna()
            contingency_table = pd.crosstab(temp_df[var1], temp_df[var2])
            
            try:
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                standardized_residuals = (contingency_table - expected) / np.sqrt(expected)
                
                # Plot heatmap of standardized residuals
                plt.figure(figsize=(12, 8))
                sns.heatmap(standardized_residuals, annot=True, cmap='RdBu_r', 
                           center=0, fmt='.2f', annot_kws={'size': 10},
                           cbar_kws={'label': 'Residual Estandarizado'})
                plt.title(f'Residuales Estandarizados: {var1} vs {var2}\n(Valores > |2| indican asociación significativa)', 
                         fontsize=12, pad=20)
                plt.tight_layout()
                plt.savefig(f'./vg-cat/eda/residues/{var1[:30]}_{var2[:30]}_residuals_{execution_type}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                results[f'{var1}_{var2}'] = standardized_residuals
            except:
                results[f'{var1}_{var2}'] = None
        
        return results

    def find_correlations(self, case_records: pd.DataFrame, vars_subset1: list, vars_subset2: list, execution_type:str) -> pd.DataFrame:
        """
        Main entry point to perform the full correlation analysis pipeline:

        - Cleans the data.
        - Runs chi-squared tests and computes Cramér's V.
        - Builds and saves result tables.
        - Computes standardized residuals for important pairs.
        - Prints a human-readable summary.
        """
        # Clean the data (don't drop rows yet)
        cleaned_df = self.remove_nans(case_records)
        
        #all_columns = vars_subset1 + vars_subset2

        # Variant to be analyzed
        all_columns = list(cleaned_df.columns)

        print_title("PERFORMING CHI-SQUARE ANALYSES", level=2, style='fixed', width=80, emoji='🔍')
        pvals_chi2, chi2_stats, dof_chi2 = self.apply_chi2(cleaned_df, all_columns, all_columns, execution_type)
        
        print_title("CALCULATING CRAMÉR'S V", level=2, style='fixed', width=80, emoji='🔍')
        pvals_cramer = self.apply_cramers(cleaned_df, all_columns, all_columns, execution_type)
        
        print_title("CALCULATING STANDARDISED RESIDUALS", level=2, style='fixed', width=80, emoji='n📈')
        pvals_residues = self.apply_standarized_residues(cleaned_df, all_columns, all_columns, execution_type)

        print_title("CREATING RESULTS TABLES", level=2, style='fixed', width=80, emoji='💾')
        correlation_df, significant_df, strong_df, chi2_pvalues_df, cramer_df = self.create_correlation_tables(
            pvals_chi2, chi2_stats, dof_chi2, pvals_cramer, execution_type
        )
        
        # Generate enhanced summary
        print_title("COMPLETE SUMMARY OF THE CORRELATION ANALYSIS", level=2, style='fixed', width=80, emoji='💾')
        self.summary_results(pvals_chi2, pvals_cramer, pvals_residues, 
                           correlation_df, chi2_pvalues_df, cramer_df, execution_type)
        
        return correlation_df, significant_df, strong_df, chi2_pvalues_df, cramer_df, pvals_residues

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .utils import print_title


class DescriptiveAnalysis:
    
    def _setup_plotting(self):
        """Initial plotting configuration"""
        plt.style.use('seaborn-v0_8-whitegrid')

    def get_contingency_tables(self, df, out_var, pred_var, group_name=""):
        """
        Generate contingency tables for categorical variables
        """
        print(f"\n{'📊 CONTINGENCY TABLE: ' + out_var + ' vs ' + pred_var + ' ' + group_name:-^75}")
        
        contingency_table = pd.crosstab(df[out_var], df[pred_var], normalize='index') * 100
        rounded_table = contingency_table.round(1)
        
        # Add totals for better interpretation
        total_counts = pd.crosstab(df[out_var], df[pred_var])
        print("Frequencies (n):")
        print(total_counts)
        print(f"\nRow percentages (%):")
        print(rounded_table)
        
        return rounded_table

    def plot_hist(self, df, var, save_dir):
        """
        Generate horizontal bar plots for categorical variables
        """
        self._setup_plotting()
        os.makedirs(save_dir, exist_ok=True)

        plt.figure(figsize=(8, 6))
        df.sort_values().plot(kind="barh")
        plt.title(f"Top 20 {var}")
        plt.xlabel("Counts")
        plt.ylabel(f"{var}")

        plt.tight_layout()
        plt.savefig(f"{save_dir}/hist_{var}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_spider(self, df, variables, save_dir):
        """
        Generate spider/radar charts for categorical variable distributions
        """
        self._setup_plotting()
        os.makedirs(save_dir, exist_ok=True)

        for var in variables:
            categories = df[var].dropna().unique().tolist()
            counts = df[var].value_counts(normalize=True) * 100
            values = [counts.get(cat, 0) for cat in categories]
            
            values_completed = values + values[:1]
            num_cats = len(categories)
            angles = [n / float(num_cats) * 2 * np.pi for n in range(num_cats)]
            angles_completed = angles + angles[:1]

            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
            
            ax.plot(angles_completed, values_completed, linewidth=2, linestyle='solid')
            ax.fill(angles_completed, values_completed, alpha=0.25)
            
            ax.set_thetagrids(np.degrees(angles), categories, fontsize=20, fontweight='bold')
            ax.tick_params(axis='x', pad=15)

            for i, (angle, value) in enumerate(zip(angles, values)):
                x = angle - 0.08
                y = value + 15
                ax.text(x, y, f'{value:.1f}%', 
                    ha='center', va='center', fontsize=20, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3',
                            edgecolor='gray'))
            
            ax.set_ylim(0, 100)
            ax.set_yticks(range(0, 101, 20))
            ax.set_yticklabels([f"{i}%" for i in range(0, 101, 20)], fontsize=18)
            
            plt.title(f'{var}', size=22, y=1.08, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{save_dir}/spider_{var}.png", dpi=300, bbox_inches='tight')
            plt.close()

    def _count_violence_corrected(self, df, violence_columns):
        """
        Count violence types with corrected categorization
        """
        results = {}
        for col in violence_columns:
            if col in df.columns:
                yes_count = (df[col] == "Sí").sum()
                no_count = (df[col] == "No").sum()
                no_data_count = (df[col] == "No consta").sum()
                other_count = len(df) - yes_count - no_count - no_data_count
                
                results[col] = {
                    'Sí': yes_count,
                    'No': no_count,
                    'No consta': no_data_count,
                    'Otros/NaN': other_count
                }
            else:
                results[col] = {'Sí': 0, 'No': 0, 'No consta': 0, 'Otros/NaN': 0}
        return results

    def _calculate_rr_or(self, h_yes, h_total, m_yes, m_total):
        """
        Calculate Risk Ratio and Odds Ratio
        """
        # Risk Ratio
        if h_yes == 0:
            rr = float('inf')
        else:
            women_risk = m_yes / m_total if m_total > 0 else 0
            men_risk = h_yes / h_total if h_total > 0 else 0
            rr = women_risk / men_risk if men_risk > 0 else float('inf')
        
        # Odds Ratio
        if h_yes == 0 and m_yes == 0:
            or_ratio = 1.0
        elif h_yes == 0:
            or_ratio = float('inf')
        elif h_yes == h_total:
            or_ratio = float('inf')
        elif m_yes == 0:
            or_ratio = 0.0
        else:
            women_odds = m_yes / (m_total - m_yes) if (m_total - m_yes) > 0 else float('inf')
            men_odds = h_yes / (h_total - h_yes) if (h_total - h_yes) > 0 else float('inf')
            or_ratio = women_odds / men_odds if men_odds > 0 else float('inf')
        
        return rr, or_ratio

    def _analyze_violence_prevalence(self, df_men, df_women, total_men, total_women):
        """
        Analyze violence prevalence by gender
        """

        # Identify violence columns
        violence_cols = [col for col in pd.concat([df_men, df_women]).columns if "Violència" in col]
        if "Detecció violència masclista" not in violence_cols:
            violence_cols = ["Detecció violència masclista"] + violence_cols

        # Count violence occurrences
        men_violence = self._count_violence_corrected(df_men, violence_cols)
        women_violence = self._count_violence_corrected(df_women, violence_cols)

        # Build results table
        results_table = []
        for col in violence_cols:
            men_yes = men_violence[col]['Sí']
            women_yes = women_violence[col]['Sí']
            
            men_pct = (men_yes / total_men) * 100 if total_men > 0 else 0
            women_pct = (women_yes / total_women) * 100 if total_women > 0 else 0
            
            rr, or_ratio = self._calculate_rr_or(men_yes, total_men, women_yes, total_women)

            # Create descriptive name
            if col == "Detecció violència masclista":
                desc_name = "Detección violencia machista"
            else:
                desc_name = col.replace("Violència", "").replace("_", " ").strip()
                if desc_name.startswith("de "):
                    desc_name = desc_name[3:]
                desc_name = desc_name.capitalize()

            results_table.append({
                'Tipo de violencia': desc_name,
                'Hombres (n)': men_yes,
                'Hombres (%)': round(men_pct, 1),
                'Mujeres (n)': women_yes,
                'Mujeres (%)': round(women_pct, 1),
                'Risk Ratio (RR)': round(rr, 1) if rr != float('inf') else "Inf",
                'Odds Ratio (OR)': round(or_ratio, 1) if or_ratio != float('inf') else "Inf",
                'Diferencia (p.p.)': round(women_pct - men_pct, 1)
            })

        return pd.DataFrame(results_table), violence_cols

    def _analyze_family_type(self, df_men, df_women, total_men, total_women):
        """
        Analyze family type distribution
        """

        family_cols = [col for col in pd.concat([df_men, df_women]).columns 
                      if "família" in col.lower() or "tipus" in col.lower()]
        
        if not family_cols:
            print("❌ No family type columns found")
            return

        family_col = family_cols[0]
        men_with_detection = df_men[df_men["Detecció violència masclista"] == "Sí"]
        women_with_detection = df_women[df_women["Detecció violència masclista"] == "Sí"]
        
        total_men_detection = len(men_with_detection)
        total_women_detection = len(women_with_detection)

        print(f"📊 CASES WITH MACHISTA VIOLENCE DETECTION:")
        print(f"   • Men: {total_men_detection}/{total_men} ({total_men_detection/total_men*100:.1f}%)")
        print(f"   • Women: {total_women_detection}/{total_women} ({total_women_detection/total_women*100:.1f}%)")

        # Men analysis
        if total_men_detection > 0 and family_col in men_with_detection.columns:
            print(f"\n👨 MEN - Family type distribution")
            print("-" * 50)
            family_men = men_with_detection[family_col].value_counts()
            
            table_men = []
            for family_type, count in family_men.items():
                pct = (count / total_men_detection) * 100
                table_men.append({
                    'Tipo de familia': family_type,
                    'Casos (n)': count,
                    'Porcentaje (%)': round(pct, 1)
                })
            
            df_table_men = pd.DataFrame(table_men)
            print(df_table_men.to_string(index=False))

        # Women analysis
        if total_women_detection > 0 and family_col in women_with_detection.columns:
            print(f"\n👩 WOMEN - Family type distribution")
            print("-" * 50)
            family_women = women_with_detection[family_col].value_counts()
            
            table_women = []
            for family_type, count in family_women.items():
                pct = (count / total_women_detection) * 100
                table_women.append({
                    'Tipo de familia': family_type,
                    'Casos (n)': count,
                    'Porcentaje (%)': round(pct, 1)
                })
            
            df_table_women = pd.DataFrame(table_women)
            print(df_table_women.to_string(index=False))

    def _analyze_services(self, df_men, df_women, total_men, total_women):
        """
        Analyze services attending to cases
        """

        service_cols = [col for col in pd.concat([df_men, df_women]).columns 
                       if any(term in col.lower() for term in ['servei', 'servicio', 'aten'])]
        
        if not service_cols:
            print("❌ No service columns found")
            return

        service_col = service_cols[0]
        men_with_detection = df_men[df_men["Detecció violència masclista"] == "Sí"]
        women_with_detection = df_women[df_women["Detecció violència masclista"] == "Sí"]
        
        total_men_detection = len(men_with_detection)
        total_women_detection = len(women_with_detection)

        # Men analysis
        if total_men_detection > 0 and service_col in men_with_detection.columns:
            print(f"\n👨 MEN - Attending services")
            print("-" * 40)
            services_men = men_with_detection[service_col].value_counts()
            
            table_men = []
            for service, count in services_men.items():
                if pd.notna(service) and service != "":
                    pct = (count / total_men_detection) * 100
                    table_men.append({
                        'Servicio': service,
                        'Casos (n)': count,
                        'Porcentaje (%)': round(pct, 1)
                    })
            
            if table_men:
                df_table_men = pd.DataFrame(table_men).sort_values('Casos (n)', ascending=False)
                print(df_table_men.to_string(index=False))
                print(f"\n📋 Unique services: {len(services_men)}")

        # Women analysis
        if total_women_detection > 0 and service_col in women_with_detection.columns:
            print(f"\n👩 WOMEN - Attending services")
            print("-" * 40)
            services_women = women_with_detection[service_col].value_counts()
            
            table_women = []
            for service, count in services_women.items():
                if pd.notna(service) and service != "":
                    pct = (count / total_women_detection) * 100
                    table_women.append({
                        'Servicio': service,
                        'Casos (n)': count,
                        'Porcentaje (%)': round(pct, 1)
                    })
            
            if table_women:
                df_table_women = pd.DataFrame(table_women).sort_values('Casos (n)', ascending=False)
                print(df_table_women.to_string(index=False))
                print(f"\n📋 Unique services: {len(services_women)}")

    def _any_violence_analysis(self, df_men, df_women, total_men, total_women, violence_cols):
        """
        Analyze individuals experiencing any type of violence
        """

        def has_any_violence(row, violence_columns):
            for col in violence_columns:
                if col in row.index and row[col] == "Sí":
                    return True
            return False

        men_any_violence = df_men.apply(lambda row: has_any_violence(row, violence_cols), axis=1).sum()
        women_any_violence = df_women.apply(lambda row: has_any_violence(row, violence_cols), axis=1).sum()

        men_pct = (men_any_violence / total_men) * 100 if total_men > 0 else 0
        women_pct = (women_any_violence / total_women) * 100 if total_women > 0 else 0

        print(f"👨 Men: {men_any_violence}/{total_men} ({men_pct:.1f}%)")
        print(f"👩 Women: {women_any_violence}/{total_women} ({women_pct:.1f}%)")

        return men_any_violence, women_any_violence, men_pct, women_pct

    def _analyze_ages(self, df_men, df_women, case_records):
        """
        Analyze contingency tables by group
        """

        # Contingency tables by group
        # print(f"\n{'👥 GROUP ANALYSIS ':->75}")
        
        print_title("MEN", level=3, style='fixed', width=80, emoji='👨')
        self.get_contingency_tables(df_men, "Edat", "Any", "(Men)")
        
        print_title("WOMEN", level=3, style='fixed', width=80, emoji='👩')
        self.get_contingency_tables(df_women, "Edat", "Any", "(Women)")
        
        print_title("TOTAL POPULATION", level=3, style='fixed', width=80, emoji='👥')
        self.get_contingency_tables(case_records, "Edat", "Any", "(Total)")

    def _generate_plots(self, case_records):
        """
        Generate various plots and charts
        """

        # Radar charts
        print_title("Generating radar charts...", level=3, style='fixed', width=80, emoji='🕸️')
        self.plot_spider(case_records, list(case_records.columns), "./vg-cat/eda/spider")
        
        print("✅ Plots generated and saved in ./vg-cat/eda/spider")

        # Histograms
        print_title("Generating histograms...", level=3, style='fixed', width=80, emoji='📊')
        comarca_counts = case_records["Comarca"].value_counts()
        siad_counts = case_records["SIAD"].value_counts()
        
        self.plot_hist(comarca_counts.head(20), "Comarcas", "./vg-cat/eda/hist")
        self.plot_hist(siad_counts.head(20), "SIAD", "./vg-cat/eda/hist")
        self.plot_hist(comarca_counts, "Comarcas (ALL)", "./vg-cat/eda/hist")
        self.plot_hist(siad_counts, "SIAD (ALL)", "./vg-cat/eda/hist")
        
        print("✅ Plots generated and saved in ./vg-cat/eda/hist")

    def _executive_summary(self, prevalence_df, total_men, total_women, men_any, women_any, men_pct, women_pct):
        """
        Generate executive summary with key findings
        """

        # Get detection data
        if not prevalence_df.empty:
            detection_data = prevalence_df[
                prevalence_df['Tipo de violencia'].str.contains('Detección', na=False)
            ]
            if not detection_data.empty:
                detection_men = detection_data['Hombres (n)'].iloc[0]
                detection_women = detection_data['Mujeres (n)'].iloc[0]
            else:
                detection_men = 0
                detection_women = 0
        else:
            detection_men = 0
            detection_women = 0

        print(f"🔍 MACHISTA VIOLENCE DETECTION:")
        print(f"   • Men: {detection_men}/{total_men} ({detection_men/total_men*100:.1f}%)")
        print(f"   • Women: {detection_women}/{total_women} ({detection_women/total_women*100:.1f}%)")

        print(f"\n💥 ANY TYPE OF VIOLENCE:")
        print(f"   • Men: {men_any}/{total_men} ({men_pct:.1f}%)")
        print(f"   • Women: {women_any}/{total_women} ({women_pct:.1f}%)")

        # Key findings from distribution table
        if not prevalence_df.empty:
            print(f"\n🔎 KEY FINDINGS:")
            
            # Highest Risk Ratio
            max_rr = prevalence_df[prevalence_df['Risk Ratio (RR)'] != "Inf"]
            if not max_rr.empty:
                max_rr_row = max_rr.loc[max_rr['Risk Ratio (RR)'].idxmax()]
                print(f"   • Highest relative risk: {max_rr_row['Tipo de violencia']} (RR={max_rr_row['Risk Ratio (RR)']})")
            
            # Highest Odds Ratio
            max_or = prevalence_df[prevalence_df['Odds Ratio (OR)'] != "Inf"]
            if not max_or.empty:
                max_or_row = max_or.loc[max_or['Odds Ratio (OR)'].idxmax()]
                print(f"   • Highest odds ratio: {max_or_row['Tipo de violencia']} (OR={max_or_row['Odds Ratio (OR)']})")
            
            # Significant risk in partner violence
            parella_row = prevalence_df[prevalence_df['Tipo de violencia'] == 'Masclista en la parella']
            if not parella_row.empty:
                rr_val = parella_row['Risk Ratio (RR)'].iloc[0]
                or_val = parella_row['Odds Ratio (OR)'].iloc[0]
                print(f"   • Partner violence: RR={rr_val}, OR={or_val}")

        print("\n\n")  

    def get_descriptive_data(self, case_records: pd.DataFrame, vars_subset1: list, vars_subset2: list) -> pd.DataFrame:
        """
        Main method to generate descriptive analysis
        """
        
        # Data preparation
        df_men = case_records[case_records["Sexe"] == "Homes"]
        df_women = case_records[case_records["Sexe"] == "Dones"]
        total_men = len(df_men)
        total_women = len(df_women)

        # Total counts
        print_title("TOTAL COUNTS", level=2, style='fixed', width=80, emoji='👥')
        print(f"   • Men: {total_men}")
        print(f"   • Women: {total_women}")

        # 1. Violence distribution analysis
        print_title("VIOLENCE TYPE DISTRIBUTION BY GROUP", level=2, style='fixed', width=80, emoji='📈')
        prevalence_df, violence_cols = self._analyze_violence_prevalence(df_men, df_women, total_men, total_women)

        # Filter types with data
        filtered_table = prevalence_df[
            (prevalence_df['Hombres (n)'] + prevalence_df['Mujeres (n)']) > 0
        ]

        print_title("VIOLENCE TYPE DISTRIBUTION (types with data)", level=3, style='fixed', width=80)
        display_cols = ['Tipo de violencia', 'Hombres (n)', 'Hombres (%)', 'Mujeres (n)', 'Mujeres (%)', 
                       'Risk Ratio (RR)', 'Odds Ratio (OR)']
        print(filtered_table[display_cols].to_string(index=False))

        # 2. Any violence analysis
        print_title("ANY VIOLENCE EXPERIENCE ANALYSIS", level=3, style='fixed', width=80)
        men_any, women_any, men_pct, women_pct = self._any_violence_analysis(
            df_men, df_women, total_men, total_women, violence_cols
        )

        # 3. Family type analysis
        print_title("FAMILY TYPE DISTRIBUTION", level=2, style='fixed', width=80, emoji='👥')
        self._analyze_family_type(df_men, df_women, total_men, total_women)

        # 4. Service analysis
        print_title("SERVICE DISTRIBUTION ", level=2, style='fixed', width=80, emoji='🏥')
        self._analyze_services(df_men, df_women, total_men, total_women)

        # 5. Age
        print_title("AGE - ANYS DISTRIBUTION", level=2, style='fixed', width=80, emoji='📊')
        self._analyze_ages(df_men, df_women, case_records)

        # 6. Plots
        print_title("PLOT GENERATION", level=2, style='fixed', width=80, emoji='📈')
        self._generate_plots(case_records)

        # 7. Executive summary (at the end)
        print_title("EXECUTIVE SUMMARY", level=2, style='fixed', width=80, emoji='🎯')
        self._executive_summary(prevalence_df, total_men, total_women, men_any, women_any, men_pct, women_pct)

        return prevalence_df
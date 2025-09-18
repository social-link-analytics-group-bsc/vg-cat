import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class DescriptiveAnalysis:


    def get_contingency_tables(self, df, out_var, pred_var):
        """
        """
        print(pd.crosstab(df[out_var], df[pred_var], normalize='index') * 100)


    def plot_spider(self, df, variables, save_dir):
        """
        """
        #
        plt.style.use('seaborn-v0_8-whitegrid')

        #
        os.makedirs("./vg-cat/eda/spider", exist_ok=True)

        for var in variables:
            # Obtener categorías únicas de la variable
            categories = df[var].dropna().unique().tolist()
            
            # Calcular porcentajes para cada categoría
            counts = df[var].value_counts(normalize=True) * 100
            values = [counts.get(cat, 0) for cat in categories]
            
            # Completar el círculo (añadir el primer valor al final)
            values_completed = values + values[:1]
            
            # Calcular ángulos para cada categoría
            num_cats = len(categories)
            angles = [n / float(num_cats) * 2 * np.pi for n in range(num_cats)]
            angles_completed = angles + angles[:1]  # Completar el círculo
            
            # Crear figura
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
            
            # Dibujar el gráfico de radar
            ax.plot(angles_completed, values_completed, linewidth=2, linestyle='solid')
            ax.fill(angles_completed, values_completed, alpha=0.25)
            
            # Añadir etiquetas para cada categoría con mayor tamaño de fuente
            ax.set_thetagrids(np.degrees(angles), categories, fontsize=20, fontweight='bold')

            # Aumentar la separación de las etiquetas de categoría del centro del gráfico
            ax.tick_params(axis='x', pad=15)  # Aumentar este valor para más separación
        
            # Añadir porcentajes en las puntas con mayor separación y tamaño de fuente
            for i, (angle, value) in enumerate(zip(angles, values)):
                # Calcular posición del texto (más alejado del punto)
                x = angle - 0.08
                y = value + 15  # Mayor desplazamiento para separar del punto
                
                # Añadir el texto del porcentaje
                ax.text(x, y, f'{value:.1f}%', 
                    ha='center', va='center', fontsize=20, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3',
                            edgecolor='gray'))
            
            # Configurar eje Y con mayor tamaño de fuente
            ax.set_ylim(0, 100)
            ax.set_yticks(range(0, 101, 20))
            ax.set_yticklabels([f"{i}%" for i in range(0, 101, 20)], fontsize=18)
            
            # Añadir título con mayor tamaño y separación
            plt.title(f'{var}', size=22, y=1.08, fontweight='bold')
            
            # Guardar gráfico
            plt.tight_layout()
            plt.savefig(f"{save_dir}/spider_{var}.png", dpi=300, bbox_inches='tight')
            plt.close()



    def get_plots(self, case_records: pd.DataFrame, vars_subset1: pd.DataFrame, vars_subset2: pd.DataFrame) -> pd.DataFrame:
        """
        """

        # -----------------------
        # Tables
        # -----------------------

        # # Contingency table by index
        # for out in vars_subset1:
        #     for pred in vars_subset2:
        #         print(f"\n--- Crosstab: {out} vs {pred} ---")
        #         self.get_contingency_tables(case_records, out, pred)

        # # -----------------------
        # # Plots
        # # -----------------------

        # # Spiders
        # 
        # self.plot_spider(case_records, vars_subset1 + vars_subset2, "./vg-cat/eda/spider")

        # # Pile bars

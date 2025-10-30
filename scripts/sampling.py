import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


class SamplingRecords:

    def plot_distribution_comparison(self, column, original_data, subset_data, output_dir, figsize=(12, 5)):
        """
        Compara distribuciones de una columna antes y después del subset
        y guarda el gráfico
        """
        try:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            # Datos originales
            original_counts = original_data[column].value_counts(normalize=True) * 100
            original_counts.plot(kind='bar', ax=axes[0], color='skyblue', alpha=0.7)
            axes[0].set_title(f'Original\n(n={len(original_data):,})', fontweight='bold')
            axes[0].set_ylabel('Porcentaje (%)')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Añadir porcentajes en las barras
            for i, (category, percentage) in enumerate(original_counts.items()):
                axes[0].text(i, percentage + 1, f'{percentage:.1f}%', 
                            ha='center', va='bottom', fontweight='bold')
            
            # Datos del subset
            subset_counts = subset_data[column].value_counts(normalize=True) * 100
            subset_counts.plot(kind='bar', ax=axes[1], color='lightgreen', alpha=0.7)
            axes[1].set_title(f'Subset\n(n={len(subset_data):,})', fontweight='bold')
            axes[1].set_ylabel('Porcentaje (%)')
            axes[1].tick_params(axis='x', rotation=45)
            
            for i, (category, percentage) in enumerate(subset_counts.items()):
                axes[1].text(i, percentage + 1, f'{percentage:.1f}%', 
                            ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # Guardar gráfico
            filename = f"distribution_{column.replace(' ', '_').replace('/', '_')}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filepath
            
        except Exception as e:
            print(f"❌ Error creando gráfico para {column}: {e}")
            plt.close()
            return None

    def analyze_distributions(self, columns_to_analyze, original_data, subset_data, output_dir):
        """
        Analiza y compara distribuciones para múltiples columnas
        Guarda gráficos para cada columna
        """
        print("="*80)
        print("📊 ANALIZANDO DISTRIBUCIONES Y GUARDANDO GRÁFICOS")
        print("="*80)
        print(f"📁 Los gráficos se guardarán en: {output_dir}")
        
        saved_plots = []
        
        for column in columns_to_analyze:
            if column in original_data.columns:
                print(f"\n🔍 Analizando: {column}")
                print("-" * 40)
                
                # Calcular distribuciones
                original_dist = original_data[column].value_counts(normalize=True) * 100
                subset_dist = subset_data[column].value_counts(normalize=True) * 100
                
                # Mostrar tabla comparativa
                comparison_df = pd.DataFrame({
                    'Original (%)': original_dist.round(1),
                    'Subset (%)': subset_dist.round(1),
                    'Diferencia': (subset_dist - original_dist).round(2)
                }).fillna(0)
                
                print(comparison_df.to_string())
                
                # Crear y guardar gráfico comparativo
                plot_path = self.plot_distribution_comparison(column, original_data, subset_data, output_dir)
                
                if plot_path:
                    saved_plots.append(plot_path)
                    print(f"💾 Gráfico guardado: {os.path.basename(plot_path)}")
                else:
                    print(f"❌ No se pudo guardar el gráfico para {column}")
                
            else:
                print(f"⚠️  {column} no encontrada en el dataset")
        
        return saved_plots

    def create_stratification_column(self, data):
        """
        Crea una columna simple para estratificación basada en las variables más importantes
        """
        # Prioridad de columnas para estratificación
        priority_columns = [
            'Any',           # Año - muy importante para series temporales
            'Sexe',          # Género - variable fundamental
            'Detecció violència masclista'  # Variable de resultado principal
        ]
        
        # Encontrar columnas disponibles
        available_columns = [col for col in priority_columns if col in data.columns]
        
        if not available_columns:
            return None
        
        # Crear columna de estratificación simple
        if len(available_columns) == 1:
            strat_col = data[available_columns[0]].astype(str)
        else:
            # Usar solo las 2 primeras columnas disponibles para evitar demasiadas categorías
            strat_col = data[available_columns[0]].astype(str) + "_" + data[available_columns[1]].astype(str)
        
        # Manejar categorías con pocos ejemplos
        value_counts = strat_col.value_counts()
        min_count = value_counts.min()
        
        if min_count < 2:
            # Agrupar categorías raras
            rare_categories = value_counts[value_counts < 2].index
            strat_col = strat_col.replace(rare_categories, 'OTRAS')
            print(f"🔧 Agrupadas {len(rare_categories)} categorías raras")
        
        print(f"🎯 Estratificación: {available_columns[:2]}")
        print(f"📊 Categorías: {len(strat_col.unique())}, Mínimo: {strat_col.value_counts().min()} casos")
        
        return strat_col

    def generate_representative_subset(self, data, target_percentage=0.1, random_state=42):
        """
        Genera un subset representativo usando estratificación simple
        """
        print("="*80)
        print("🎯 GENERANDO SUBSET REPRESENTATIVO")
        print("="*80)
        
        target_size = int(len(data) * target_percentage)
        print(f"🎯 Target: {target_size:,} casos ({target_percentage*100:.1f}% de {len(data):,})")
        
        # Crear columna de estratificación
        stratify_col = self.create_stratification_column(data)
        
        if stratify_col is None:
            print("⚠️  No se pudo crear estratificación. Usando muestra aleatoria.")
            return data.sample(n=target_size, random_state=random_state)
        
        # Verificar que tenemos suficientes casos para estratificación
        min_class_size = stratify_col.value_counts().min()
        if min_class_size < 2:
            print("⚠️  Categorías muy pequeñas. Usando muestra aleatoria.")
            return data.sample(n=target_size, random_state=random_state)
        
        # Realizar split estratificado
        try:
            train_data, _ = train_test_split(
                data,
                train_size=target_size,
                random_state=random_state,
                stratify=stratify_col
            )
            print(f"✅ Subset creado: {len(train_data):,} casos")
            return train_data
            
        except Exception as e:
            print(f"⚠️  Error en estratificación: {e}")
            print("🔄 Usando muestra aleatoria como fallback")
            return data.sample(n=target_size, random_state=random_state)


    def apply_filter(self, df: pd.DataFrame, vars_subset1: pd.DataFrame, vars_subset2: pd.DataFrame) -> pd.DataFrame:
        """
        Función principal con tamaño de subset muy reducido para análisis estadísticos óptimos
        """
        # Crear directorio de salida
        output_dir = "./subset_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        print("="*80)
        print("🔍 CREANDO SUBSET REPRESENTATIVO - TAMAÑO ÓPTIMO")
        print("="*80)
        
        # Calcular tamaño óptimo basado en el dataset original
        original_size = len(df)
        
        # ESTRATEGIA MUY CONSERVADORA - ÓPTIMA PARA CHI-CUADRADO
        if original_size > 500000:
            target_percentage = 0.007  # 0.7% → ~5,000 casos
            reason = "Muy grande (>500K) - óptimo para chi-cuadrado"
        elif original_size > 100000:
            target_percentage = 0.01   # 1% → ~7,000 casos
            reason = "Grande (>100K) - excelente para tests"
        else:
            target_percentage = 0.02   # 2% para datasets más pequeños
            reason = "Mediano/pequeño - buen balance"
        
        target_size = int(original_size * target_percentage)
        
        print(f"📊 Estrategia de muestreo: {reason}")
        print(f"🎯 Target optimizado: {target_size:,} casos ({target_percentage*100:.1f}%)")
        print(f"💡 Justificación: Tamaño óptimo para tests chi-cuadrado - evita potencia excesiva")
        
        # Columnas para análisis
        analysis_columns = [
            'Any', 'Sexe', 'Edat', 'Tipus de família', 'Comarca',
            'Detecció violència masclista', 'Violència física', 
            'Violència psicològica', 'Servei que atén'
        ]
        
        # Filtrar columnas disponibles
        available_columns = [col for col in analysis_columns if col in df.columns]
        print(f"📋 Analizando {len(available_columns)} columnas")
        
        # Generar subset con tamaño optimizado
        representative_subset = self.generate_representative_subset(
            data=df,
            target_percentage=target_percentage,
            random_state=42
        )
        
        # Analizar distribuciones y guardar gráficos
        print("\n" + "="*80)
        print("📈 COMPARANDO DISTRIBUCIONES")
        print("="*80)
        
        saved_plots = self.analyze_distributions(
            columns_to_analyze=available_columns,
            original_data=df,
            subset_data=representative_subset,
            output_dir=output_dir
        )
        
        # Resumen final con evaluación estadística detallada
        print("\n" + "="*80)
        print("🎯 RESUMEN FINAL - EVALUACIÓN ESTADÍSTICA")
        print("="*80)
        
        final_size = len(representative_subset)
        print(f"📊 Dataset original: {original_size:,} casos")
        print(f"📊 Subset creado: {final_size:,} casos")
        print(f"📊 Porcentaje: {final_size/original_size*100:.2f}%")
        print(f"📈 Gráficos guardados: {len(saved_plots)}")
        
        # Evaluación estadística detallada
        print(f"\n🔬 EVALUACIÓN ESTADÍSTICA DETALLADA:")
        
        if final_size > 20000:
            assessment = "❌ EXCESIVO"
            power_note = "Potencia > 0.99 para efectos muy pequeños (φ < 0.05)"
            recommendation = "Reducir a 5,000-10,000 casos"
            chi2_note = "Chi-cuadrado detectará diferencias triviales como significativas"
            
        elif final_size > 10000:
            assessment = "⚠️  ALTO"
            power_note = "Potencia ~0.95-0.99 para efectos pequeños (φ = 0.1)"
            recommendation = "Adecuado para análisis complejos, pero aún potencia alta"
            chi2_note = "Bueno para detectar efectos pequeños-moderados"
            
        elif final_size > 5000:
            assessment = "✅ ÓPTIMO"
            power_note = "Potencia ~0.80-0.90 para efectos pequeños-moderados (φ = 0.1-0.2)"
            recommendation = "Excelente para la mayoría de aplicaciones"
            chi2_note = "Ideal para chi-cuadrado - detecta efectos relevantes"
            
        elif final_size > 2000:
            assessment = "✅ ADECUADO"
            power_note = "Potencia ~0.70-0.80 para efectos moderados (φ = 0.2)"
            recommendation = "Bueno para análisis exploratorios"
            chi2_note = "Adecuado para efectos moderados-grandes"
            
        else:
            assessment = "⚠️  PEQUEÑO"
            power_note = "Potencia < 0.70 para efectos pequeños"
            recommendation = "Solo para análisis preliminares"
            chi2_note = "Puede faltar potencia para efectos pequeños"
        
        print(f"   • Evaluación: {assessment}")
        print(f"   • Tamaño: {final_size:,} casos")
        print(f"   • Potencia: {power_note}")
        print(f"   • Chi-cuadrado: {chi2_note}")
        print(f"   • Recomendación: {recommendation}")
        
        # Mostrar archivos guardados
        if saved_plots:
            print("\n📁 Archivos de gráficos guardados:")
            for plot_path in saved_plots:
                print(f"   📄 {os.path.basename(plot_path)}")
        
        # Proporciones clave con mayor precisión
        print("\n🔍 PROPORCIONES CLAVE:")
        if 'Sexe' in df.columns:
            orig_women = (df['Sexe'] == 'Dones').mean() * 100
            subset_women = (representative_subset['Sexe'] == 'Dones').mean() * 100
            diff = subset_women - orig_women
            print(f"   👩 Mujeres: {orig_women:.2f}% → {subset_women:.2f}% (Δ{diff:+.3f}p.p.)")
        
        if 'Detecció violència masclista' in df.columns:
            orig_violence = (df['Detecció violència masclista'] == 'Sí').mean() * 100
            subset_violence = (representative_subset['Detecció violència masclista'] == 'Sí').mean() * 100
            diff = subset_violence - orig_violence
            print(f"   🔍 Violencia detectada: {orig_violence:.2f}% → {subset_violence:.2f}% (Δ{diff:+.3f}p.p.)")
        
        # Calcular diferencia máxima en proporciones clave
        max_diff = 0
        for col in ['Sexe', 'Detecció violència masclista']:
            if col in df.columns:
                orig = (df[col] == 'Dones').mean() * 100 if col == 'Sexe' else (df[col] == 'Sí').mean() * 100
                subset_val = (representative_subset[col] == 'Dones').mean() * 100 if col == 'Sexe' else (representative_subset[col] == 'Sí').mean() * 100
                diff = abs(subset_val - orig)
                if diff > max_diff:
                    max_diff = diff
        
        print(f"   📏 Máxima diferencia: {max_diff:.3f}p.p.")
        
        # Guardar subset como CSV
        subset_path = os.path.join(output_dir, "representative_subset.csv")
        try:
            representative_subset.to_csv(subset_path, index=False)
            print(f"\n💾 Subset guardado en: {subset_path}")
        except Exception as e:
            print(f"❌ Error guardando subset: {e}")
        
        # Guardar resumen estadístico detallado
        summary_path = os.path.join(output_dir, "statistical_assessment.txt")
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("EVALUACIÓN ESTADÍSTICA - SUBSET REPRESENTATIVO\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Dataset original: {original_size:,} casos\n")
                f.write(f"Subset creado: {final_size:,} casos\n")
                f.write(f"Porcentaje: {final_size/original_size*100:.3f}%\n\n")
                
                f.write("EVALUACIÓN ESTADÍSTICA:\n")
                f.write(f"- Evaluación: {assessment}\n")
                f.write(f"- Tamaño: {final_size:,} casos\n")
                f.write(f"- Potencia: {power_note}\n")
                f.write(f"- Chi-cuadrado: {chi2_note}\n")
                f.write(f"- Recomendación: {recommendation}\n\n")
                
                f.write("PROPORCIONES CLAVE:\n")
                if 'Sexe' in df.columns:
                    orig_women = (df['Sexe'] == 'Dones').mean() * 100
                    subset_women = (representative_subset['Sexe'] == 'Dones').mean() * 100
                    f.write(f"- Mujeres: {orig_women:.3f}% → {subset_women:.3f}%\n")
                
                if 'Detecció violència masclista' in df.columns:
                    orig_violence = (df['Detecció violència masclista'] == 'Sí').mean() * 100
                    subset_violence = (representative_subset['Detecció violència masclista'] == 'Sí').mean() * 100
                    f.write(f"- Violencia detectada: {orig_violence:.3f}% → {subset_violence:.3f}%\n")
                
                f.write(f"- Máxima diferencia: {max_diff:.3f} p.p.\n\n")
                
                f.write("RECOMENDACIONES PARA ANÁLISIS CHI-CUADRADO:\n")
                if final_size <= 5000:
                    f.write("- Tamaño adecuado para efectos moderados (φ > 0.2)\n")
                    f.write("- Buen balance entre potencia y especificidad\n")
                    f.write("- Resultados más interpretables\n")
                else:
                    f.write("- Considerar ajustar nivel de significancia (ej: α = 0.01)\n")
                    f.write("- Interpretar tamaños de efecto (V de Cramer) en lugar de solo p-valores\n")
                    f.write("- Las diferencias pequeñas pueden ser estadísticamente significativas\n")
            
            print(f"📊 Evaluación estadística guardada en: {summary_path}")
        except Exception as e:
            print(f"❌ Error guardando evaluación: {e}")
        
        return representative_subset
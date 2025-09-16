import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import load_config 
import missingno as msno
from scipy.stats import chi2_contingency, pointbiserialr
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
import xgboost as xgb
import prince
import shap
from sklearn.inspection import permutation_importance


class ExploratoryAnalysis:

    def __init__(self, config_violences_path, config_demographics_path):
        self.config_violences = load_config(config_violences_path)
        self.config_demographics = load_config(config_demographics_path)

    def get_variables(self, config_type: str, name_variable: str):
        """Obtiene variables de configuración"""
        config = self.config_violences if config_type == "outcomes" else self.config_demographics
        return [var[name_variable] for var in config[config_type]]

    def exploratory_data_analysis(self, df: pd.DataFrame, demographic_vars: list, violence_vars: list):
        """Análisis exploratorio de datos"""
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Crear directorios si no existen
        import os
        os.makedirs("./vg-cat/eda", exist_ok=True)
        
        # Función auxiliar para crear gráficos
        def plot_distributions(variables, title_prefix, save_path, normalize=False):
            fig, axes = plt.subplots(1, len(variables), figsize=(5*max(1, len(variables)), 5))
            if len(variables) == 1:
                axes = [axes]
            for i, col in enumerate(variables):
                if normalize:
                    (df[col].value_counts(normalize=True) * 100).plot(kind='bar', ax=axes[i])
                    axes[i].set_ylabel('Porcentaje')
                else:
                    df[col].value_counts().plot(kind='bar', ax=axes[i])
                    axes[i].set_ylabel('Counts')
                
                axes[i].set_title(f'{title_prefix} - {col}')
                axes[i].tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Gráficos de distribución
        plot_distributions(demographic_vars, 'Demográficas', "./vg-cat/eda/demographic_counts.png")
        plot_distributions(violence_vars, 'Violencia', "./vg-cat/eda/violence_counts.png")
        plot_distributions(demographic_vars, 'Demográficas', "./vg-cat/eda/demographic_percentages.png", normalize=True)
        plot_distributions(violence_vars, 'Violencia', "./vg-cat/eda/violence_percentages.png", normalize=True)
        
        # Heatmaps de tablas de contingencia
        for demo_var in demographic_vars:
            for violence_var in violence_vars:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Conteos
                cont_table = pd.crosstab(df[demo_var], df[violence_var])
                sns.heatmap(cont_table, annot=True, fmt='d', cmap='Blues', ax=ax1)
                ax1.set_title(f'{demo_var} vs {violence_var} (Conteos)')
                
                # Porcentajes
                cont_table_pct = pd.crosstab(df[demo_var], df[violence_var], normalize='index') * 100
                sns.heatmap(cont_table_pct, annot=True, fmt='.1f', cmap='Blues', ax=ax2)
                ax2.set_title(f'{demo_var} vs {violence_var} (%)')
                
                plt.tight_layout()
                plt.savefig(f"./vg-cat/eda/heatmaps_{demo_var}_{violence_var}.png", dpi=300, bbox_inches='tight')
                plt.close()

    def multiple_correspondence_analysis(self, df, demographic_vars, violence_vars):
        """Análisis de Correspondencias Múltiples (ACM)"""
        # Preparar datos
        mca_df = df[demographic_vars].copy()
        
        # Convertir a string
        for col in demographic_vars:
            mca_df[col] = mca_df[col].astype(str)
        
        # Aplicar MCA
        mca = prince.MCA(n_components=2, random_state=42)
        mca.fit(mca_df)
        
        # Transformar datos
        coordinates = mca.transform(mca_df)
        
        # Calcular varianza explicada
        eigenvalues = mca.eigenvalues_
        total_inertia = sum(eigenvalues)
        explained_inertia = [eig / total_inertia for eig in eigenvalues[:2]]
        
        # Visualización
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Configurar colores según la primera variable de violencia
        if violence_vars and len(violence_vars) > 0:
            color_var = df[violence_vars[0]].astype(str)
            palette = sns.color_palette('Set2', len(color_var.unique()))
            scatter = sns.scatterplot(
                x=coordinates.iloc[:, 0], 
                y=coordinates.iloc[:, 1], 
                hue=color_var, 
                ax=ax, 
                palette=palette,
                s=30, 
                alpha=0.6
            )
        else:
            scatter = sns.scatterplot(
                x=coordinates.iloc[:, 0], 
                y=coordinates.iloc[:, 1], 
                ax=ax, 
                s=30, 
                alpha=0.6
            )
        
        # Obtener coordenadas de categorías
        column_coords = mca.column_coordinates(mca_df)
        
        # Añadir etiquetas para las categorías más importantes
        distances = np.sqrt(column_coords[0]**2 + column_coords[1]**2)
        threshold = np.percentile(distances, 75)
        
        for idx, row in column_coords.iterrows():
            if distances[idx] > threshold:
                ax.plot([row[0], row[0]*1.05], [row[1], row[1]*1.05], 
                        color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
                ax.text(row[0]*1.07, row[1]*1.07, idx, 
                        color='darkred', fontsize=8, weight='bold',
                        bbox=dict(facecolor='white', alpha=0.7, 
                                    edgecolor='none', boxstyle='round,pad=0.2'))
        
        # Configurar ejes y título
        ax.set_xlabel(f'Componente 1 ({explained_inertia[0]*100:.1f}% varianza explicada)')
        ax.set_ylabel(f'Componente 2 ({explained_inertia[1]*100:.1f}% varianza explicada)')
        ax.set_title('Análisis de Correspondencias Múltiples (MCA)')
        
        # Leyenda
        if violence_vars and len(violence_vars) > 0:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=violence_vars[0])
        
        # Ajustar límites
        x_min, x_max = coordinates.iloc[:, 0].min(), coordinates.iloc[:, 0].max()
        y_min, y_max = coordinates.iloc[:, 1].min(), coordinates.iloc[:, 1].max()
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
        
        # Cuadrícula
        ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./vg-cat/mca/mca_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return mca, coordinates, explained_inertia

    def analyze_missing_patterns(self, df, columns):
        """Analiza patrones de valores faltantes"""
        # Crear directorios si no existen
        import os
        os.makedirs("./vg-cat/missingness", exist_ok=True)
        
        df = df.replace("No consta", np.nan)
        
        # Visualizaciones de missingness
        msno.bar(df[columns], figsize=(10, 6), color='steelblue')
        plt.title('Porcentaje de Valores No Faltantes por Columna')
        plt.savefig('./vg-cat/missingness/bar_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        msno.heatmap(df[columns], figsize=(12, 8), cmap='coolwarm')
        plt.title('Correlación de Valores Faltantes entre Variables')
        plt.savefig('./vg-cat/missingness/heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        msno.dendrogram(df[columns], figsize=(12, 8))
        plt.title('Dendrograma de Patrones de Valores Faltantes')
        plt.savefig('./vg-cat/missingness/dendrogram.png', dpi=300, bbox_inches='tight')
        plt.close()

        msno.matrix(df, sort='descending', figsize=(12, 8), fontsize=10)
        plt.title('Matriz de Valores Faltantes (Ordenada por Completitud)', fontsize=14)
        plt.savefig('./vg-cat/missingness/matrix.png', dpi=300, bbox_inches='tight')
        plt.close()


        # Análisis estadístico de missingness
        results = {}
        missing_matrix = df[columns].isnull().astype(int).corr()
        
        for col in columns:
            if df[col].isnull().sum() > 0:
                missing_flag = df[col].isnull().astype(int)
                p_values = {}
                
                for other_col in [c for c in columns if c != col]:
                    if df[other_col].dtype == 'object':
                        contingency = pd.crosstab(missing_flag, df[other_col].fillna('Missing'))
                        _, p, _, _ = chi2_contingency(contingency)
                        p_values[other_col] = p
                    else:
                        _, p = pointbiserialr(missing_flag, df[other_col].fillna(df[other_col].mean()))
                        p_values[other_col] = p
                
                results[col] = p_values
        
        # Visualizar correlación de missingness
        plt.figure(figsize=(10, 8))
        sns.heatmap(missing_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlación entre Patrones de Valores Faltantes')
        plt.tight_layout()
        plt.savefig('./vg-cat/missingness/missing_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Resultados del análisis de patrones de missing values:")
        for col, p_values in results.items():
            print(f"\nVariable: {col}")
            for other_col, p in p_values.items():
                print(f"  vs {other_col}: p={p:.4f} {'***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''}")
        
        return results

    def prepare_data_for_modeling(self, df, demographic_vars, violence_var):
        """Prepara datos para modelado"""
        data = df.copy()
        
        # Variable objetivo binaria
        data[violence_var] = data[violence_var].apply(lambda x: 1 if x == 'Sí' else 0)
        
        # Manejo de missing values en variables demográficas
        for col in demographic_vars:
            data[f'{col}_missing'] = data[col].isnull().astype(int)
            data[col] = data[col].fillna('Missing')
        
        # One-hot encoding
        data = pd.get_dummies(data, columns=demographic_vars, drop_first=False)
        
        return data.drop(columns=[violence_var]), data[violence_var]

    def train_and_evaluate_model(self, model, X, y, model_name, violence_var):
        """Entrena y evalúa un modelo"""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        model.fit(X, y)
        
        print(f"{model_name} para {violence_var}: AUC medio = {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")
        return model, cv_scores.mean(), cv_scores.std()

    def run_models(self, X, y, violence_var):
        """Ejecuta y compara todos los modelos"""
        # Crear directorios si no existen
        import os
        os.makedirs("./vg-cat/models", exist_ok=True)
        
        models = {
            'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100),
            'XGBoost': xgb.XGBClassifier(
                scale_pos_weight=(len(y) - sum(y)) / sum(y), 
                random_state=42, 
                n_estimators=100,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        }
        
        results = {}
        for name, model in models.items():
            fitted_model, mean_auc, std_auc = self.train_and_evaluate_model(model, X, y, name, violence_var)
            results[name] = {'model': fitted_model, 'mean_auc': mean_auc, 'std_auc': std_auc}
            
            # Guardar importancia de características para modelos tree-based
            if hasattr(fitted_model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': fitted_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Visualizar importancia
                plt.figure(figsize=(10, 8))
                top_features = feature_importance.head(15)
                plt.barh(range(len(top_features)), top_features['importance'])
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.title(f'Importancia de Variables - {name} - {violence_var}')
                plt.tight_layout()
                plt.savefig(f'./vg-cat/models/{name.lower().replace(" ", "_")}_importance_{violence_var}.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # Comparar modelos
        plt.figure(figsize=(10, 6))
        model_names = list(results.keys())
        auc_means = [results[name]['mean_auc'] for name in model_names]
        auc_stds = [results[name]['std_auc'] for name in model_names]
        
        plt.bar(range(len(model_names)), auc_means, yerr=auc_stds, capsize=5)
        plt.xticks(range(len(model_names)), model_names, rotation=45)
        plt.ylabel('AUC Score')
        plt.title(f'Comparación de Modelos - {violence_var}')
        plt.tight_layout()
        plt.savefig(f'./vg-cat/models/model_comparison_{violence_var}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return results

    def explain_with_shap(self, model, X, model_name, violence_var):
        """Explica el modelo usando SHAP"""
        try:
            if model_name == 'Logistic Regression':
                # Para modelos lineales, usar explainer lineal
                explainer = shap.LinearExplainer(model, X)
                shap_values = explainer.shap_values(X)
            else:
                # Para modelos tree-based
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
            
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
            plt.title(f'SHAP Values - {model_name} - {violence_var}')
            plt.tight_layout()
            plt.savefig(f'./vg-cat/models/{model_name.lower().replace(" ", "_")}_shap_{violence_var}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Force plot para una instancia específica
            plt.figure(figsize=(10, 4))
            shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:], show=False, matplotlib=True)
            plt.title(f'SHAP Force Plot - {model_name} - {violence_var} (Primera instancia)')
            plt.tight_layout()
            plt.savefig(f'./vg-cat/models/{model_name.lower().replace(" ", "_")}_force_plot_{violence_var}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"No se pudo calcular SHAP values para {model_name}: {e}")

    def bootstrap_validation(self, X, y, model, n_bootstraps=1000):
        """Realiza validación por bootstrapping para estimar intervalos de confianza."""
        bootstrapped_scores = []
        for _ in range(n_bootstraps):
            # Muestra bootstrap
            indices = np.random.choice(range(len(X)), size=len(X), replace=True)
            X_boot = X.iloc[indices]
            y_boot = y.iloc[indices]
            
            # Dividir en train y test
            X_train, X_test, y_train, y_test = train_test_split(
                X_boot, y_boot, test_size=0.3, random_state=42, stratify=y_boot)
            
            # Entrenar y evaluar
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            y_pred = model_clone.predict_proba(X_test)[:, 1]
            score = roc_auc_score(y_test, y_pred)
            bootstrapped_scores.append(score)
        
        # Calcular intervalos de confianza
        mean_score = np.mean(bootstrapped_scores)
        ci_lower = np.percentile(bootstrapped_scores, 2.5)
        ci_upper = np.percentile(bootstrapped_scores, 97.5)
        
        return mean_score, ci_lower, ci_upper, bootstrapped_scores

    def run(self, case_records: pd.DataFrame, siad_centers: pd.DataFrame) -> pd.DataFrame:
        """Flujo principal de análisis"""
        # Crear directorios si no existen
        import os
        os.makedirs("./vg-cat/eda", exist_ok=True)
        os.makedirs("./vg-cat/mca", exist_ok=True)
        os.makedirs("./vg-cat/missingness", exist_ok=True)
        os.makedirs("./vg-cat/models", exist_ok=True)
        
        # Obtener variables
        violences_variables = self.get_variables("outcomes", "name")
        demographics_variables = self.get_variables("predictors", "name")
        
        # EDA
        print("EDA...")
        self.exploratory_data_analysis(case_records, demographics_variables, violences_variables)

        # ACM
        print("Realizando Análisis de Correspondencias Múltiples...")
        self.multiple_correspondence_analysis(case_records, demographics_variables, violences_variables)

        # Análisis de missing values
        print("Analyzing missing patterns...")
        missing_results = self.analyze_missing_patterns(case_records, demographics_variables + violences_variables)
        





        # Decisión sobre imputación
        for col, p_values in missing_results.items():
            significant_deps = [p for p in p_values.values() if p < 0.05]
            if len(significant_deps) > 0:
                print(f"La variable {col} parece tener missingness MAR")
                use_imputation = True
            else:
                print(f"La variable {col} podría ser MCAR o MNAR")
                use_imputation = False
        
        # Imputación MICE si es necesario
        if use_imputation:
            print("Aplicando imputación MICE...")
            # Codificar variables categóricas para imputación
            label_encoders = {}
            analysis_df_imputed = case_records.copy()
            
            for col in demographics_variables:
                if analysis_df_imputed[col].dtype == 'object':
                    le = LabelEncoder()
                    # Ajustar el encoder solo con valores no nulos
                    non_missing = analysis_df_imputed[col].dropna()
                    if len(non_missing) > 0:
                        le.fit(non_missing)
                        analysis_df_imputed[col] = analysis_df_imputed[col].apply(
                            lambda x: le.transform([x])[0] if pd.notnull(x) else np.nan)
                        label_encoders[col] = le
            
            # Aplicar MICE
            imputer = IterativeImputer(random_state=42, max_iter=10)
            imputed_data = imputer.fit_transform(analysis_df_imputed[demographics_variables])
            
            # Convertir de vuelta a categorías
            for i, col in enumerate(demographics_variables):
                if col in label_encoders:
                    analysis_df_imputed[col] = imputed_data[:, i].round()
                    analysis_df_imputed[col] = analysis_df_imputed[col].apply(
                        lambda x: label_encoders[col].inverse_transform([int(x)])[0] 
                        if not np.isnan(x) and int(x) in range(len(label_encoders[col].classes_)) 
                        else 'Missing')
                else:
                    analysis_df_imputed[col] = imputed_data[:, i]
            
            case_records = analysis_df_imputed
        
        # # Modelado para cada variable de violencia
        # all_results = {}
        # for violence_var in violences_variables:
        #     print(f"\nAnalizando {violence_var}...")
            
        #     X, y = self.prepare_data_for_modeling(case_records, demographics_variables, violence_var)
        #     results = self.run_models(X, y, violence_var)
        #     all_results[violence_var] = results
            
        #     # Explicabilidad con SHAP para cada modelo
        #     for model_name, model_info in results.items():
        #         self.explain_with_shap(model_info['model'], X, model_name, violence_var)
            
        #     # Validación por bootstrapping para Random Forest
        #     rf_model = results['Random Forest']['model']
        #     mean_score, ci_lower, ci_upper, boot_scores = self.bootstrap_validation(X, y, rf_model)
        #     print(f"Random Forest Bootstrapping para {violence_var}: AUC = {mean_score:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})")
            
        #     # Guardar resultados
        #     results_df = pd.DataFrame({
        #         'violence_type': violence_var,
        #         'logistic_regression_auc': results['Logistic Regression']['mean_auc'],
        #         'random_forest_auc': results['Random Forest']['mean_auc'],
        #         'xgboost_auc': results['XGBoost']['mean_auc'],
        #         'rf_bootstrap_auc': mean_score,
        #         'rf_bootstrap_ci_lower': ci_lower,
        #         'rf_bootstrap_ci_upper': ci_upper
        #     }, index=[0])
            
        #     results_df.to_csv(f'./vg-cat/models/results_summary_{violence_var}.csv', index=False)
        
        # return all_results
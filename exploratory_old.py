import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import textwrap
from .utils import load_config 
import missingno as msno
from scipy.stats import spearmanr, kendalltau, chi2_contingency, pointbiserialr
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.inspection import permutation_importance
import xgboost as xgb
import prince  # Para Análisis de Correspondencias Múltiples
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import shap  # Para explicabilidad de modelos


class ExploratoryAnalysis:

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


    def exploratory_data_analysis(self, df: pd.DataFrame, demographic_vars: list, violence_vars:list ):
        """
        """

        # Set style for better-looking plots
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 1. Demographic Variables (Counts)
        fig, axes = plt.subplots(1, len(demographic_vars), figsize=(5*len(demographic_vars), 5))
        if len(demographic_vars) == 1:
            axes = [axes]
        for i, col in enumerate(demographic_vars):
            df[col].value_counts().plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'Distribución de {col} (Conteos)')
            axes[i].set_ylabel('Counts')
            axes[i].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(f"./vg-cat/eda/demographic_counts.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Violence Variables (Counts)
        fig, axes = plt.subplots(1, len(violence_vars), figsize=(5*len(violence_vars), 5))
        if len(violence_vars) == 1:
            axes = [axes]
        for i, col in enumerate(violence_vars):
            df[col].value_counts().plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'Distribución de {col} (Conteos)')
            axes[i].set_ylabel('Counts')
            axes[i].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(f"./vg-cat/eda/violence_counts.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Demographic Variables (Percentages)
        fig, axes = plt.subplots(1, len(demographic_vars), figsize=(5*len(demographic_vars), 5))
        if len(demographic_vars) == 1:
            axes = [axes]
        for i, col in enumerate(demographic_vars):
            (df[col].value_counts(normalize=True) * 100).plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'Distribución de {col} (%)')
            axes[i].set_ylabel('Porcentaje')
            axes[i].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(f"./vg-cat/eda/demographic_percentages.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Violence Variables (Percentages)
        fig, axes = plt.subplots(1, len(violence_vars), figsize=(5*len(violence_vars), 5))
        if len(violence_vars) == 1:
            axes = [axes]
        for i, col in enumerate(violence_vars):
            (df[col].value_counts(normalize=True) * 100).plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'Distribución de {col} (%)')
            axes[i].set_ylabel('Porcentaje')
            axes[i].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(f"./vg-cat/eda/violence_percentages.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Contingency Tables with Heatmaps
        for demo_var in demographic_vars:
            for violence_var in violence_vars:
                # Counts contingency table
                cont_table = pd.crosstab(df[demo_var], df[violence_var])
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Plot counts
                sns.heatmap(cont_table, annot=True, fmt='d', cmap='Blues', ax=ax1)
                ax1.set_title(f'{demo_var} vs {violence_var} (Conteos)')
                
                # Plot percentages
                cont_table_pct = pd.crosstab(df[demo_var], df[violence_var], normalize='index') * 100
                sns.heatmap(cont_table_pct, annot=True, fmt='.1f', cmap='Blues', ax=ax2)
                ax2.set_title(f'{demo_var} vs {violence_var} (%)')
                
                plt.tight_layout()
                plt.savefig(f"./vg-cat/eda/heatmaps_{demo_var}_{violence_var}.png", dpi=300, bbox_inches='tight')
                plt.close()
       


    def missigness_patterns_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        """
        # Reemplazar "No consta" con NaN
        df = df.replace("No consta", np.nan)
        
        # Gráfico de barras que muestra el conteo/porcentaje de valores no faltantes
        msno.bar(df, figsize=(10, 6), fontsize=10, color='steelblue')
        plt.title('Porcentaje de Valores No Faltantes por Columna', fontsize=14)
        plt.savefig('./vg-cat/missingness/bar_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
            
        # Mapa de calor que muestra correlaciones entre valores faltantes
        msno.heatmap(df, figsize=(12, 8), fontsize=10, cmap='coolwarm')
        plt.title('Correlación de Valores Faltantes entre Variables', fontsize=14)
        plt.savefig('./vg-cat/missingness/heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Dendrograma que agrupa variables con patrones similares de valores missing
        msno.dendrogram(df, figsize=(12, 8), fontsize=10, orientation='top')
        plt.title('Agrupamiento de Variables por Patrones de Valores Faltantes', fontsize=14)
        plt.savefig('./vg-cat/missingness/dendrogram.png', dpi=300, bbox_inches='tight')
        plt.close()
            
        # Matrix con ordenamiento para destacar patrones
        msno.matrix(df, sort='descending', figsize=(12, 8), fontsize=10)
        plt.title('Matriz de Valores Faltantes (Ordenada por Completitud)', fontsize=14)
        plt.savefig('./vg-cat/missingness/matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        
    def analyze_missing_patterns(self, df, columns):
        """Analiza si los missing values son MCAR, MAR o MNAR"""
        results = {}

        # Reemplazar "No consta" con NaN
        df = df.replace("No consta", np.nan)
        
        # Matriz de correlación de missingness
        missing_matrix = df[columns].isnull().astype(int).corr()
        
        # Test de independencia entre missingness y variables observadas
        for col in columns:
            if df[col].isnull().sum() > 0:
                # Crear flag de missingness
                missing_flag = df[col].isnull().astype(int)
                
                # Testar contra otras variables
                p_values = {}
                for other_col in [c for c in columns if c != col]:
                    if df[other_col].dtype == 'object':
                        # Para variables categóricas, usar chi-cuadrado
                        contingency = pd.crosstab(missing_flag, df[other_col].fillna('Missing'))
                        chi2, p, dof, expected = chi2_contingency(contingency)
                        p_values[other_col] = p
                    else:
                        # Para variables numéricas, usar point-biserial
                        not_missing = df.loc[missing_flag == 0, other_col]
                        missing = df.loc[missing_flag == 1, other_col]
                        if len(not_missing) > 0 and len(missing) > 0:
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

    def multiple_correspondence_analysis(self, df, demographic_vars, violence_vars):
        """
        Realiza Análisis de Correspondencias Múltiples (MCA) sin usar adjust_text
        """
        
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
        
        # Obtener varianza explicada (manera compatible con versiones recientes)
        # Para prince 0.16.0, usamos eigenvalues para calcular la inercia explicada
        eigenvalues = mca.eigenvalues_
        total_inertia = sum(eigenvalues)
        explained_inertia = [eig / total_inertia for eig in eigenvalues]
        
        # Visualización
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Configurar colores
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
        
        # Estrategia para mostrar etiquetas sin superposición
        distances = np.sqrt(column_coords[0]**2 + column_coords[1]**2)
        threshold = np.percentile(distances, 75)  # Mostrar solo el 25% superior
        
        # Añadir etiquetas con conexiones para las categorías más importantes
        important_labels = []
        for idx, row in column_coords.iterrows():
            if distances[idx] > threshold:
                # Dibujar línea desde el punto hasta la etiqueta
                ax.plot([row[0], row[0]*1.05], [row[1], row[1]*1.05], 
                        color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
                # Añadir etiqueta en posición desplazada
                label = ax.text(row[0]*1.07, row[1]*1.07, idx, 
                            color='darkred', fontsize=8, weight='bold',
                            bbox=dict(facecolor='white', alpha=0.7, 
                                        edgecolor='none', boxstyle='round,pad=0.2'))
                important_labels.append(label)
        
        # Mejorar etiquetas de ejes con varianza explicada calculada
        ax.set_xlabel(f'Componente 1 ({explained_inertia[0]*100:.1f}% varianza explicada)')
        ax.set_ylabel(f'Componente 2 ({explained_inertia[1]*100:.1f}% varianza explicada)')
        ax.set_title('Análisis de Correspondencias Múltiples (MCA)')
        
        # Mejorar leyenda
        if violence_vars and len(violence_vars) > 0:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=violence_vars[0])
        
        # Ajustar límites para dar espacio a las etiquetas
        x_min, x_max = coordinates.iloc[:, 0].min(), coordinates.iloc[:, 0].max()
        y_min, y_max = coordinates.iloc[:, 1].min(), coordinates.iloc[:, 1].max()
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
        
        # Añadir cuadrícula para mejor referencia
        ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./vg-cat/mca/mca_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Información adicional para diagnóstico
        print("Valores propios:", eigenvalues)
        print("Inercia total:", total_inertia)
        print("Inercia explicada por componente:", explained_inertia)
        
        return mca, coordinates, explained_inertia
    

    def prepare_data_for_modeling(self, df, demographic_vars, violence_var):
        """
        Prepara los datos para el modelado:
        - Convierte la variable de violencia en binaria (1: 'Sí', 0: otro)
        - Aplica one-hot encoding a las variables demográficas.
        - Maneja missing values en demográficas: crea categoría 'Missing' y flags de missing.
        """
        data = df.copy()
        
        # Variable objetivo
        data[violence_var] = data[violence_var].apply(lambda x: 1 if x == 'Sí' else 0)
        
        # Para variables demográficas: reemplazar missing por 'Missing' y crear flags
        for col in demographic_vars:
            # Crear flag de missing
            data[f'{col}_missing'] = data[col].isnull().astype(int)
            # Reemplazar missing por 'Missing'
            data[col] = data[col].fillna('Missing')
        
        # One-hot encoding para variables demográficas
        data = pd.get_dummies(data, columns=demographic_vars, drop_first=False)
        
        # Separar en X e y
        X = data.drop(columns=[violence_var])
        y = data[violence_var]
        
        return X, y

    def logistic_regression_analysis(self, X, y, violence_var):
        """
        Realiza análisis de regresión logística con validación cruzada.
        """
        # Entrenar modelo
        model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
        
        # Validación cruzada
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        print(f"Regresión Logística para {violence_var}: AUC medio = {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")
        
        # Entrenar modelo final
        model.fit(X, y)
        
        # Obtener coeficientes
        coefficients = pd.DataFrame({
            'feature': X.columns,
            'coefficient': model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
        
        # Visualizar coeficientes
        plt.figure(figsize=(10, 8))
        top_coeffs = coefficients.head(15)
        colors = ['red' if c < 0 else 'green' for c in top_coeffs['coefficient']]
        plt.barh(range(len(top_coeffs)), top_coeffs['coefficient'], color=colors)
        plt.yticks(range(len(top_coeffs)), top_coeffs['feature'])
        plt.title(f'Coeficientes de Regresión Logística - {violence_var}')
        plt.tight_layout()
        plt.savefig(f'./vg-cat/models/lr_coefficients_{violence_var}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return model, coefficients

    def random_forest_analysis(self, X, y, violence_var):
        """
        Realiza análisis con Random Forest con validación cruzada.
        """
        # Entrenar modelo
        model = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
        
        # Validación cruzada
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        print(f"Random Forest para {violence_var}: AUC medio = {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")
        
        # Entrenar modelo final
        model.fit(X, y)
        
        # Obtener importancia de características
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Visualizar importancia
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.title(f'Importancia de Variables - Random Forest - {violence_var}')
        plt.tight_layout()
        plt.savefig(f'./vg-cat/models/rf_importance_{violence_var}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return model, feature_importance

    def xgboost_analysis(self, X, y, violence_var):
        """
        Realiza análisis con XGBoost con validación cruzada.
        """
        # Calcular balance de clases para scale_pos_weight
        scale_pos_weight = (len(y) - sum(y)) / sum(y)
        
        # Entrenar modelo
        model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight, 
            random_state=42, 
            n_estimators=100,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        # Validación cruzada
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        print(f"XGBoost para {violence_var}: AUC medio = {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")
        
        # Entrenar modelo final
        model.fit(X, y)
        
        # Obtener importancia de características
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Visualizar importancia
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.title(f'Importancia de Variables - XGBoost - {violence_var}')
        plt.tight_layout()
        plt.savefig(f'./vg-cat/models/xgb_importance_{violence_var}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # SHAP analysis para explicabilidad
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
            plt.title(f'SHAP Values - XGBoost - {violence_var}')
            plt.tight_layout()
            plt.savefig(f'./vg-cat/models/xgb_shap_{violence_var}.png', dpi=300, bbox_inches='tight')
            plt.close()
        except:
            print(f"No se pudo calcular SHAP values para {violence_var}")
        
        return model, feature_importance

    def compare_models(self, X, y, violence_var):
        """
        Compara múltiples modelos y sus resultados.
        """
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
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
            results[name] = {
                'mean_auc': cv_scores.mean(),
                'std_auc': cv_scores.std(),
                'scores': cv_scores
            }
            print(f"{name} para {violence_var}: AUC medio = {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")
        
        # Visualizar comparación
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

    def bootstrap_validation(self, X, y, model, n_bootstraps=1000):
        """
        Realiza validación por bootstrapping para estimar intervalos de confianza.
        """
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
        """
        """
            
        # Get choosen variables to analyze
        print("Get Variables...\n")
        violences_variables = self.get_variables("outcomes", "name")
        demographics_variables = self.get_variables("predictors", "name")
        
        # # EDA
        # print("EDA...\n")
        # self.exploratory_data_analysis(case_records, demographics_variables, violences_variables)

        # # ACM
        # print("Realizando Análisis de Correspondencias Múltiples...")
        # self.multiple_correspondence_analysis(case_records, demographics_variables, violences_variables)  

        # # Missingness patterns detection
        # print("Missingness pattern detection...\n")
        # self.missigness_patterns_detection(case_records)

        # # Analyzing missing patterns
        # print("Analyzing missing patterns...\n")
        # missing_results = self.analyze_missing_patterns(case_records, demographics_variables + violences_variables)
        
        # # Decisión sobre imputación basada en el análisis
        # use_imputation = True
        # for col, p_values in missing_results.items():
        #     # Si hay correlaciones significativas, es probablemente MAR
        #     significant_deps = [p for p in p_values.values() if p < 0.05]
        #     if len(significant_deps) > 0:
        #         print(f"La variable {col} parece tener missingness MAR")
        #     else:
        #         print(f"La variable {col} podría ser MCAR o MNAR")
        #         use_imputation = False
        #         # Para MNAR, considerar no imputar o usar métodos específicos
            
        
        use_imputation = True


        # Estrategia 2: Imputación con MICE
        if use_imputation:
            print("Aplicando imputación MICE...")
            # Codificar variables categóricas para imputación
            label_encoders = {}
            analysis_df_imputed = analysis_df.copy()
            
            for col in demographic_columns:
                if analysis_df_imputed[col].dtype == 'object':
                    le = LabelEncoder()
                    non_missing = analysis_df_imputed[col].dropna()
                    le.fit(non_missing)
                    analysis_df_imputed[col] = analysis_df_imputed[col].apply(
                        lambda x: le.transform([x])[0] if pd.notnull(x) else np.nan)
                    label_encoders[col] = le
            
            # Aplicar MICE
            imputer = IterativeImputer(random_state=42, max_iter=10, sample_posterior=True)
            imputed_data = imputer.fit_transform(analysis_df_imputed[demographic_columns])
            
            for i, col in enumerate(demographic_columns):
                analysis_df_imputed[col] = imputed_data[:, i]
                # Convertir de vuelta a categorías si es necesario
                if col in label_encoders:
                    # Redondear y mapear a categorías originales
                    analysis_df_imputed[col] = analysis_df_imputed[col].round().astype(int)
                    analysis_df_imputed[col] = analysis_df_imputed[col].apply(
                        lambda x: label_encoders[col].inverse_transform([x])[0] if x in label_encoders[col].classes_ else 'Missing')
                else:
                    # Para variables numéricas, mantener como está
                    pass

        # 6. Preparación de datos para modelado
        def prepare_model_data(analysis_df, strategy_name):
            """Prepara datos para modelado según la estrategia"""
            model_data = analysis_df.copy()
            
            # One-hot encoding para variables categóricas
            categorical_cols = [col for col in demographic_columns if model_data[col].dtype == 'object']
            model_data = pd.get_dummies(model_data, columns=categorical_cols, drop_first=False)
            
            # Asegurar que todas las violence columns estén presentes
            for col in violence_columns:
                if col not in model_data.columns:
                    model_data[col] = analysis_df[col]
            
            return model_data

        # Preparar datos para ambas estrategias
        model_data_no_impute = prepare_model_data(analysis_df_no_impute, "no_imputation")
        if use_imputation:
            model_data_imputed = prepare_model_data(analysis_df_imputed, "imputation")


        all_results = {}
        for violence_var in violences_variables:
            print(f"\nAnalizando {violence_var}...")
            
            # Preparar datos
            X, y = self.prepare_data_for_modeling(case_records, demographics_variables, violence_var)
            
            # Comparar modelos
            model_results = self.compare_models(X, y, violence_var)
            all_results[violence_var] = model_results
            
            # Análisis detallado por modelo
            lr_model, lr_coeffs = self.logistic_regression_analysis(X, y, violence_var)
            rf_model, rf_importance = self.random_forest_analysis(X, y, violence_var)
            xgb_model, xgb_importance = self.xgboost_analysis(X, y, violence_var)
            
            # Validación por bootstrapping (ejemplo con Random Forest)
            mean_score, ci_lower, ci_upper, boot_scores = self.bootstrap_validation(
                X, y, RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
            )
            print(f"Random Forest Bootstrapping para {violence_var}: AUC = {mean_score:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})")
            
            # Guardar resultados
            results_df = pd.DataFrame({
                'violence_type': violence_var,
                'logistic_regression_auc': model_results['Logistic Regression']['mean_auc'],
                'random_forest_auc': model_results['Random Forest']['mean_auc'],
                'xgboost_auc': model_results['XGBoost']['mean_auc'],
                'rf_bootstrap_auc': mean_score,
                'rf_bootstrap_ci_lower': ci_lower,
                'rf_bootstrap_ci_upper': ci_upper
            }, index=[0])
            
            results_df.to_csv(f'./vg-cat/models/results_summary_{violence_var}.csv', index=False)






















    # def exploratory_statistics(self, df, demographic_var, violence_var): 
    #     """
    #     """
    #     contingency_table = pd.crosstab(df[demographic_var], df[violence_var])
        
    #     percentages = pd.crosstab(df[demographic_var], df[violence_var], normalize='index') * 100


    #     print(percentages)


        
        # return {}

# class ExploratoryAnalysis:

#     def __init__(self, config_violences_path, config_demographics_path):
#         self.config_violences = load_config(config_violences_path)
#         self.config_demographics = load_config(config_demographics_path)

#     def get_variables(self, config_type:str, name_variable:str ):
#         """
#         """
#         if config_type == "outcomes":
#             return [var[name_variable] for var in self.config_violences[config_type]]

#         if config_type == "predictors":
#             return [var[name_variable] for var in self.config_demographics[config_type]]

    
#     def exploratory_statistics(self, df, demographic_var, violence_var): 
#         """
#         """
#         contingency_table = pd.crosstab(df[demographic_var], df[violence_var])
        
#         percentages = pd.crosstab(df[demographic_var], df[violence_var], normalize='index') * 100


#         print(percentages)

#         return {}
   
#     def run(self, case_records: pd.DataFrame, siad_centers: pd.DataFrame) -> pd.DataFrame:
#         """
#         """
    
#         statistics = {}
        
#         violences_variables = self.get_variables("outcomes", "name")
#         demographics_variables = self.get_variables("predictors", "name")

#         for demographic_var, violence_var in itertools.product(demographics_variables, violences_variables):
#             if demographic_var in case_records.columns and violence_var in case_records.columns:
#                 result_exploratory = self.exploratory_statistics(case_records, demographic_var, violence_var)
#         #         statistics[f"{demographic_var}_{violence_var}"] = result_exploratory
        
#         # return statistics




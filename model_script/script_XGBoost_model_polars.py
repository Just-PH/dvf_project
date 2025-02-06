import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy.stats import norm
from scipy.spatial import cKDTree
from sklearn.preprocessing import OneHotEncoder,RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


# Étape 1 : Nettoyage des données (Colonnes nécessaires et NaN)

class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, nombre_lots_max = 5, cutoff_valeur_fonciere_min = 0, cutoff_valeur_fonciere_max = 1e9, min_surface = 0, max_surface = 1e9):
        self.nombre_lots_max = nombre_lots_max
        self.cutoff_valeur_fonciere_min = cutoff_valeur_fonciere_min
        self.cutoff_valeur_fonciere_max = cutoff_valeur_fonciere_max
        self.min_surface = min_surface
        self.max_surface = max_surface


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Utilisation de Polars pour charger et filtrer les données
        print(f"Shape before filtering: {X.shape}")
        surface_totale = (pl.col("surface_reelle_bati").fill_null(0) +
                          pl.col("surface_terrain").fill_null(0) +
                          pl.col("lot1_surface_carrez").fill_null(0) +
                          pl.col("lot2_surface_carrez").fill_null(0) +
                          pl.col("lot3_surface_carrez").fill_null(0) +
                          pl.col("lot4_surface_carrez").fill_null(0) +
                          pl.col("lot5_surface_carrez").fill_null(0))
        # Appliquer les filtres sur les colonnes
        X = X.filter(
            (pl.col("valeur_fonciere").is_not_null()) &
            (pl.col("longitude").is_not_null()) &
            (pl.col("latitude").is_not_null()) &
            (pl.col("nombre_lots") <= self.nombre_lots_max) &
            (pl.col("valeur_fonciere") >= self.cutoff_valeur_fonciere_min) &
            (pl.col("valeur_fonciere") <= self.cutoff_valeur_fonciere_max) &
            (surface_totale >= self.min_surface) &
            (surface_totale <= self.max_surface)
        )
        print(f"Shape after filtering: {X.shape}")
        return X

# Étape 2 : Création de colonnes calculées
class FeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self,cutoff_prix_m2_min=0,cutoff_prix_m2_max=1e6):
        self.cutoff_prix_m2_min = cutoff_prix_m2_min
        self.cutoff_prix_m2_max = cutoff_prix_m2_max

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X = X.clone()
        # Ajout de transformations basées sur X
        X = X.with_columns([
            (pl.col("date_mutation").dt.month() / 12).sin().alias("sin_month"),
            (pl.col("date_mutation").dt.month() / 12).cos().alias("cos_month"),
            pl.col("date_mutation").dt.year().alias("year"),
            pl.col("lot1_surface_carrez").fill_null(0).alias("lot1_surface_carrez"),
            pl.col("lot2_surface_carrez").fill_null(0).alias("lot2_surface_carrez"),
            pl.col("lot3_surface_carrez").fill_null(0).alias("lot3_surface_carrez"),
            pl.col("lot4_surface_carrez").fill_null(0).alias("lot4_surface_carrez"),
            pl.col("lot5_surface_carrez").fill_null(0).alias("lot5_surface_carrez"),
            pl.col("surface_reelle_bati").fill_null(0).alias("surface_reelle_bati"),
            pl.col("surface_terrain").fill_null(0).alias("surface_terrain")

        ])


        # Groupby et agrégation
        X = X.group_by(['id_mutation', 'valeur_fonciere']).agg([
            pl.col("surface_reelle_bati").sum(),
            pl.col("year").first(),
            pl.col("sin_month").first(),
            pl.col("cos_month").first(),
            pl.col("type_local").map_elements(lambda x: ', '.join(sorted(x.unique())), return_dtype=pl.Utf8).alias("type_local"),
            pl.col("nombre_lots").max(),
            pl.col("lot1_surface_carrez").mean(),
            pl.col("lot2_surface_carrez").mean(),
            pl.col("lot3_surface_carrez").mean(),
            pl.col("lot4_surface_carrez").mean(),
            pl.col("lot5_surface_carrez").mean(),
            pl.col("nombre_pieces_principales").sum(),
            pl.col("surface_terrain").sum(),
            pl.col("longitude").first(),
            pl.col("latitude").first(),
        ])
        X = X.with_columns(
            pl.col("type_local").fill_null("Inconnu")
        )
        X = X.with_columns([
            (pl.col("lot1_surface_carrez").cast(pl.Float64) +
            pl.col("lot2_surface_carrez").cast(pl.Float64) +
            pl.col("lot3_surface_carrez").cast(pl.Float64) +
            pl.col("lot4_surface_carrez").cast(pl.Float64) +
            pl.col("lot5_surface_carrez").cast(pl.Float64)).alias("total_surface_carrez")
        ])

        # X = X.with_columns([
        #     pl.when(pl.col("total_surface_carrez") == 0)
        #     .then(pl.col("surface_reelle_bati"))
        #     .otherwise(pl.col("total_surface_carrez"))
        #     .alias("total_surface_carrez")
        # ])
        X = X.with_columns(
                pl.when(pl.col("total_surface_carrez") > 0)
                .then(pl.col("valeur_fonciere") / pl.col("total_surface_carrez"))
                .when(pl.col("surface_terrain") > 0)
                .then(pl.col("valeur_fonciere") / pl.col("surface_terrain"))
                .when(pl.col("surface_reelle_bati") > 0)
                .then(pl.col("valeur_fonciere") / pl.col("surface_reelle_bati"))
                .alias("prix_m2")
        )
        X = X.filter((pl.col("prix_m2") > self.cutoff_prix_m2_min)
                     & (pl.col("prix_m2") <= self.cutoff_prix_m2_max))
        X = X.drop("id_mutation")

        return X

## Étape 3 : Filtrage des anomalies
class AnomalyFilter(BaseEstimator, TransformerMixin):
    def __init__(self, contamination=0.1, target_elimination = False, prix_m2 = False):
        self.contamination = contamination
        self.model = IsolationForest(contamination=self.contamination, random_state=42)
        self.target_elimination = target_elimination
        self.prix_m2 = prix_m2
        if self.target_elimination == False:
            self.anomaly_columns =['surface_reelle_bati', 'nombre_lots', 'surface_terrain', 'nombre_pieces_principales',"total_surface_carrez","densite_weighted"]
        else:
            self.anomaly_columns =['surface_reelle_bati', 'nombre_lots', 'surface_terrain', 'nombre_pieces_principales',"total_surface_carrez","densite_weighted",'prix_m2']

    def fit(self, X, y=None):
        self.model.fit(X[self.anomaly_columns])
        return self

    def transform(self, X):
        X = X.with_columns([
        pl.Series("anomalie", self.model.predict(X[self.anomaly_columns])).alias("anomalie")
        ])
        X = X.filter(pl.col("anomalie") == 1)
        if self.prix_m2:
            return X.drop(["anomalie","valeur_fonciere"])
        else:
            return X.drop(["anomalie","prix_m2"])

# class AnomalyFilter(BaseEstimator, TransformerMixin):
#     def __init__(self, contamination=0.1, target_elimination=False):
#         self.contamination = contamination
#         self.model = IsolationForest(contamination=self.contamination, random_state=42)
#         self.target_elimination = target_elimination
#         self.anomaly_columns = [
#             "surface_reelle_bati", "nombre_lots", "surface_terrain",
#             "nombre_pieces_principales", "total_surface_carrez", "densite_weighted"
#         ]

#         if self.target_elimination:
#             self.anomaly_columns.append("valeur_fonciere")

#     def fit(self, X, y=None):
#         # Si target_elimination est activé, calculer prix_m2 en évitant la division par zéro
#         if self.target_elimination:
#            self.anomaly_columns.append("prix_m2")


#         # Entraîner IsolationForest
#         self.model.fit(X[self.anomaly_columns])
#         return self

#     def transform(self, X):
#         # Recalculer prix_m2 en évitant la division par zéro
#         X = X.drop_nulls(self.anomaly_columns)

#         # Prédire les anomalies
#         X = X.with_columns(
#             pl.Series("anomalie", self.model.predict(X[self.anomaly_columns])).alias("anomalie")
#         )

#         # Filtrer les anomalies et supprimer la colonne anomalie
#         X = X.filter(pl.col("anomalie") == 1).drop("anomalie")
#         X = X.drop("prix_m2")

#         return X

# Étape 4 : ajout des données de densité et de pois
class WeightedPOICountsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=4, df_grid=None):
        self.poi_columns = ['densite', 'transport_pois', 'education_pois', 'health_pois', 'food_pois',
                            'shopping_pois', 'park_pois', 'entertainment_pois', 'cultural_pois']
        self.n_neighbors = n_neighbors
        self.df_grid = df_grid

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.df_grid is None:
            raise ValueError("df_grid doit être fourni dans fit avant d'appeler transform.")

        # Utilisation de Polars pour calculer la distance et les poids
        tree = cKDTree(self.df_grid[['lon', 'lat']].to_numpy())

        distances, indices = tree.query(X[['longitude', 'latitude']].to_numpy(), k=self.n_neighbors)

        # Calcul des poids et des POIs pondérés
        weights = 1 / np.where(distances == 0, 1e-10, distances)
        normalized_weights = weights / weights.sum(axis=1, keepdims=True)

        for col in self.poi_columns:
            poi_values = self.df_grid[col].to_numpy()
            neighbors_poi = poi_values[indices]
            weighted_poi = (neighbors_poi * normalized_weights).sum(axis=1)
            X = X.with_columns([pl.Series(f"{col}_weighted", weighted_poi)])


        return X

def data_loader(path, departements=[], annees=[], fraction=None, chunksize=None):
    df = pl.DataFrame()

    annees_list = os.listdir(path) if not annees else [str(annee) for annee in annees]

    for annee in annees_list:
        cur_year = os.path.join(path, annee)
        departements_list = os.listdir(cur_year) if not departements else [f"{departement:02}.csv.gz" for departement in departements]

        for departement in departements_list:
            try:

                file = os.path.join(cur_year, departement)

                if chunksize:
                    chunk_list = []
                    for chunk in pl.read_csv(file, batch_size=chunksize):
                        if fraction:
                            chunk = chunk.sample(frac=fraction, random_state=42)
                        chunk_list.append(chunk)

                    temp_df = pl.concat(chunk_list)
                else:
                    temp_df = pl.read_csv(
                                        file,
                                        columns=['id_mutation','date_mutation', 'type_local',
                                                'surface_reelle_bati','nombre_lots', 'lot1_surface_carrez',
                                                'lot2_surface_carrez','lot3_surface_carrez', 'lot4_surface_carrez',
                                                'lot5_surface_carrez','nombre_pieces_principales',
                                                'surface_terrain', 'longitude', 'latitude',
                                                'valeur_fonciere'
                                            ],
                                        schema_overrides={
                                                'id_mutation':pl.Utf8,
                                                'date_mutation':pl.Date,
                                                'type_local':pl.Utf8,
                                                'surface_reelle_bati':pl.Float32,
                                                'nombre_lots':pl.Float32,
                                                'lot1_surface_carrez':pl.Float32,
                                                'lot2_surface_carrez':pl.Float32,
                                                'lot3_surface_carrez':pl.Float32,
                                                'lot4_surface_carrez':pl.Float32,
                                                'lot5_surface_carrez':pl.Float32,
                                                'nombre_pieces_principales':pl.Float32,
                                                'surface_terrain':pl.Float32,
                                                'longitude':pl.Float32,
                                                'latitude':pl.Float32,
                                                'valeur_fonciere':pl.Float64
                                            },
                                        ignore_errors=True)

                df = pl.concat([df, temp_df], how="vertical")
            except Exception as e:
                print(f"❌ Erreur avec {file} : {e}")
                continue

    return df





def build_xgboost_model(params,numerical_columns, categorical_columns_onehot,unique_categories):



    onehot_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', categories=unique_categories))
    ])

    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', RobustScaler())
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ('onehot', onehot_pipeline, categorical_columns_onehot),
            ('numeric', numeric_pipeline, numerical_columns)
        ]
    )

    # pipeline_model = Pipeline(steps=[
    #     ('encoding', column_transformer),
    #     ('model', XGBRegressor(params))
    # ])
    return column_transformer,  XGBRegressor(params)


def plot_train_test_predictions(y_train, y_test, X_train, X_test, model, save_path):
    """
    Génère une figure avec deux graphiques côte à côte : prédictions pour l'ensemble d'entraînement et de test.

    Parameters:
    - y_train : array-like, les vraies valeurs de l'ensemble d'entraînement.
    - y_test : array-like, les vraies valeurs de l'ensemble de test.
    - X_train : array-like, les features de l'ensemble d'entraînement.
    - X_test : array-like, les features de l'ensemble de test.
    - model : object, le modèle entraîné qui possède une méthode .predict().
    - save_path : str, chemin pour sauvegarder la figure générée.

    Returns:
    - None. La figure est sauvegardée à l'emplacement spécifié par save_path.
    """
    # Prédictions pour les ensembles d'entraînement et de test
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calcul des métriques pour l'ensemble d'entraînement
    mse_train = mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)

    # Calcul des métriques pour l'ensemble de test
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    # Création de la figure
    plt.figure(figsize=(14, 6))

    # Graphique 1 : Ensemble d'entraînement
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_train, y=y_pred_train, alpha=0.6, edgecolor=None)
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color="red", linestyle="--", label="Perfect Prediction")
    plt.title("Train Set", fontsize=14)
    plt.xlabel("True Values", fontsize=12)
    plt.ylabel("Predicted Values", fontsize=12)
    plt.yscale('log')
    plt.xscale('log')
    plt.text(
        0.05, 0.95,  # Position dans le graphique (proportions)
        f"MSE: {mse_train:.2f}\nR²: {r2_train:.2f}",
        fontsize=10,
        ha="left",
        va="top",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
    )
    plt.grid(True)
    plt.legend()

    # Graphique 2 : Ensemble de test
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.6, edgecolor=None)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", label="Perfect Prediction")
    plt.title("Test Set", fontsize=14)
    plt.xlabel("True Values", fontsize=12)
    plt.ylabel("Predicted Values", fontsize=12)
    plt.yscale('log')
    plt.xscale('log')
    plt.text(
        0.05, 0.95,  # Position dans le graphique (proportions)
        f"MSE: {mse_test:.2f}\nR²: {r2_test:.2f}",
        fontsize=10,
        ha="left",
        va="top",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
    )
    plt.grid(True)
    plt.legend()

    # Sauvegarde de la figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()  # Fermer la figure après sauvegarde

def save_model(pipeline, filename):
    """Sauvegarde le modèle dans un fichier"""
    joblib.dump(pipeline, filename)
    print(f"Modèle sauvegardé sous {filename}")

def load_model(filename):
    """Charge un modèle depuis un fichier"""
    return joblib.load(filename)

if __name__ == "__main__":
    print("Loading data")
    path = 'data_dvf'
    df = data_loader(path, annees=["2021"]) #
    print("Data loaded")
    df_grid = pl.read_csv('data_pop_density/dataframe_densite&amenities_radius=500.csv')

    # Pipeline complète
    pipeline_preprocess = Pipeline(steps=[
        ("cleaner", DataCleaner(nombre_lots=1,cutoff_valeur_fonciere_min=1e5, cutoff_valeur_fonciere_max=10e6, min_surface= 15)),
        ("feature_creator", FeatureCreator()),
        ("anomaly_filter", AnomalyFilter(contamination=0.2, target_elimination = True)),
        ('weighted_poi', WeightedPOICountsTransformer(n_neighbors=4)),
    ])

    pipeline_preprocess.set_params(weighted_poi__df_grid = df_grid)

    df = pipeline_preprocess.fit_transform(df)
    print("Data preprocessed")
    X = df.drop('valeur_fonciere')
    y= df['valeur_fonciere']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    categorical_columns_onehot = ['type_local'] # Columns that need OneHotEncoding
    numerical_columns = ['surface_reelle_bati', 'year', 'sin_month',
    'cos_month', 'nombre_lots', 'total_surface_carrez','lot1_surface_carrez',
    'lot2_surface_carrez', 'lot3_surface_carrez', 'lot4_surface_carrez',
    'lot5_surface_carrez', 'nombre_pieces_principales', 'surface_terrain',
    'longitude', 'latitude', 'densite_weighted', 'transport_pois_weighted',
    'education_pois_weighted', 'health_pois_weighted', 'food_pois_weighted',
    'shopping_pois_weighted', 'park_pois_weighted',
    'entertainment_pois_weighted', 'cultural_pois_weighted']
    unique_categories = [X[col].drop_nulls().unique() for col in categorical_columns_onehot]

    params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_jobs": 2,
        }
    xgb_model = build_xgboost_model(params, numerical_columns, categorical_columns_onehot,unique_categories)
    xgb_model.fit(X_train, y_train)
    save_model(xgb_model, 'result_xgb_model/xgboost_model.pkl')  # Sauvegarde du modèle
    plot_train_test_predictions(
    y_train=y_train,
    y_test=y_test,
    X_train=X_train,
    X_test=X_test,
    model=xgb_model,
    save_path="result_xgb_model/train_test_scatter_plots_with_metrics.png"
)

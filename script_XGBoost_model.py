import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy.stats import norm
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from model_script.script_XGBoost_model_polars import DataCleaner, FeatureCreator, AnomalyFilter, WeightedPOICountsTransformer, data_loader, build_xgboost_model,save_model,plot_train_test_predictions


if __name__ == "__main__":
    regions = {
    "Auvergne-Rhône-Alpes": ["01", "03", "07", "15", "26", "38", "42", "43", "63", "69", "73", "74"],
    "Bourgogne-Franche-Comté": ["21", "25", "39", "58", "70", "71", "89", "90"],
    "Bretagne": ["22", "29", "35", "56"],
    "Centre-Val de Loire": ["18", "28", "36", "37", "41", "45"],
    "Corse": ["2A", "2B"],
    "Grand Est": ["08", "10", "51", "52", "54", "55", "57", "67", "68", "88"],
    "Hauts-de-France": ["02", "59", "60", "62", "80"],
    "Île-de-France": ["75", "77", "78", "91", "92", "93", "94", "95"],
    "Normandie": ["14", "27", "50", "61", "76"],
    "Nouvelle-Aquitaine": ["16", "17", "19", "23", "24", "33", "40", "47", "64", "79", "86", "87"],
    "Occitanie": ["09", "11", "12", "30", "31", "32", "34", "46", "48", "65", "66", "81", "82"],
    "Pays de la Loire": ["44", "49", "53", "72", "85"],
    "Provence-Alpes-Côte d'Azur": ["04", "05", "06", "13", "83", "84"]
        }
    for region in regions.keys():

        YEARS = ["2019","2020", "2021", "2022", "2023","2024"]
        PREPROCESSED_FILE = "data_processed/data_dvf_preprocessed.csv"
        OVERWRITE = True
        BATCH_SIZE = 100_000
        dep = regions[region]

        print("Starting preprocessing by year...")

        # Charger la grille pour le pipeline
        df_grid = pl.read_csv('data_pop_density/dataframe_densite&amenities_radius=500.csv')

        # Définir le pipeline de preprocessing
        pipeline_preprocess = Pipeline(steps=[
            ("cleaner", DataCleaner(nombre_lots_max=5, cutoff_valeur_fonciere_min=0.75e5, min_surface=15)),
            ("feature_creator", FeatureCreator(cutoff_prix_m2_min=3e3,cutoff_prix_m2_max=18e3)),
            ('weighted_poi', WeightedPOICountsTransformer(n_neighbors=4)),
            ("anomaly_filter", AnomalyFilter(contamination=0.1, target_elimination=True)),
            # ('weighted_poi', WeightedPOICountsTransformer(n_neighbors=4)),
        ])

        pipeline_preprocess.set_params(weighted_poi__df_grid=df_grid)

        # Effacer le fichier CSV s'il existe déjà (évite d'ajouter plusieurs fois les mêmes données)
        if OVERWRITE:
            with open(PREPROCESSED_FILE, "w") as f:
                pass

            # === 1. Charger et prétraiter chaque année séparément ===
            for year in YEARS:
                print(f"Processing year {year}...")

                # Charger les données de l'année en cours
                df_year = data_loader('data_dvf', annees=[year],departements=dep)
                print(f"Data loaded for {year}: {df_year.shape[0]} rows")

                # Appliquer le préprocessing
                df_processed = pipeline_preprocess.fit_transform(df_year)
                df_processed.drop_nulls()
                print(f"Data processed for {year}: {df_processed.shape[0]} rows")

                # Sauvegarder les données prétraitées en ajoutant au CSV
                with open(PREPROCESSED_FILE, mode="a") as f:
                    df_processed.write_csv(f,include_header=True)

            print("All years processed and saved!")

        # === 2. Charger toutes les données prétraitées une fois que tout est stocké ===
        df_lazy = pl.scan_csv(
            PREPROCESSED_FILE,
            has_header = True,
            schema_overrides={
                'surface_reelle_bati': pl.Float32,
                'type_local': pl.Utf8,  # Optimisation mémoire
                'year': pl.Int32,  # Réduction mémoire
                'sin_month': pl.Float32,
                'cos_month': pl.Float32,
                'nombre_lots': pl.Float32,
                'total_surface_carrez': pl.Float32,
                'lot1_surface_carrez': pl.Float32,
                'lot2_surface_carrez': pl.Float32,
                'lot3_surface_carrez': pl.Float32,
                'lot4_surface_carrez': pl.Float32,
                'lot5_surface_carrez': pl.Float32,
                'nombre_pieces_principales': pl.Float32,
                'surface_terrain': pl.Float32,
                'longitude': pl.Float32,
                'latitude': pl.Float32,
                'valeur_fonciere': pl.Float64,
                'densite_weighted': pl.Float32,
                'transport_pois_weighted': pl.Float32,
                'education_pois_weighted': pl.Float32,
                'health_pois_weighted': pl.Float32,
                'food_pois_weighted': pl.Float32,
                'shopping_pois_weighted': pl.Float32,
                'park_pois_weighted': pl.Float32,
                'entertainment_pois_weighted': pl.Float32,
                'cultural_pois_weighted': pl.Float32
            },
            ignore_errors = True
        ).drop_nulls()


        # === 3. Séparation Train / Test (Lazy) ===
        df = df_lazy.collect().sample(fraction=1.0,shuffle=True) # Convertit en DataFrame
        df = df.with_columns(pl.lit(np.random.rand(len(df))).alias("split"))  # Ajoute une colonne random
        df_lazy = df.lazy()  # Reconvertit en LazyFrame# Ajoute une colonne random
        df_train = df_lazy.filter(pl.col("split") < 0.8).drop("split")  # 80% pour l'entraînement
        df_test = df_lazy.filter(pl.col("split") >= 0.8).drop("split")  # 20% pour le test

        df_train_features = df_train.drop("valeur_fonciere")
        df_train_target = df_train.select("valeur_fonciere")
        print(df_train_features.collect().shape)

        df_test_features = df_test.drop("valeur_fonciere")
        df_test_target = df_test.select("valeur_fonciere")
        print(df_test_features.collect().shape)


        # === 4. Pipeline d'encodage ===
        categorical_columns_onehot = ['type_local']
        numerical_columns = [
            'surface_reelle_bati', 'year', 'sin_month', 'cos_month', 'nombre_lots',
            'total_surface_carrez', 'lot1_surface_carrez', 'lot2_surface_carrez',
            'lot3_surface_carrez', 'lot4_surface_carrez', 'lot5_surface_carrez',
            'nombre_pieces_principales', 'surface_terrain', 'longitude', 'latitude',
            'densite_weighted', 'transport_pois_weighted', 'education_pois_weighted',
            'health_pois_weighted', 'food_pois_weighted', 'shopping_pois_weighted',
            'park_pois_weighted', 'entertainment_pois_weighted', 'cultural_pois_weighted'
        ]
        unique_categories = [df_lazy.select(col).drop_nulls().unique().collect().to_series().to_list()
        for col in categorical_columns_onehot]
        # Créer les catégories pour OneHotEncoder
        # === 5. Entraînement du modèle ===
        print('Fitting du modèle :')
        params = {
        "objective": "reg:squarederror",  # Régression avec erreur quadratique
        "learning_rate": 0.5,            # Plus bas = plus précis mais plus lent
        "max_depth": 10,                  # Profondeur des arbres (trop élevé = overfitting)
        "n_jobs": 2,                      # Nombre de threads utilisés
        "subsample": 0.8,                 # Échantillonnage aléatoire pour éviter l'overfitting
        "gamma": 0.1,                     # Prune les arbres faibles (réduction de l'overfitting)
        "min_child_weight": 10,           # Empêche d'ajouter des feuilles inutiles
        "lambda": 1,                     # Régularisation L2 (Ridge)
        "alpha": 0.5                      # Régularisation L1 (Lasso)
    }

        column_transformer, xgb_model = build_xgboost_model(params, numerical_columns, categorical_columns_onehot, unique_categories )

        def batch_generator(features_lazy, target_lazy, batch_size, params, column_transformer,xgb_model=None):
            """
            Génère des lots (batches) et ajuste le modèle XGBoost sur chaque batch, avec transformation des données.

            Parameters:
            - xgb_model: Le modèle XGBoost qui sera ajusté sur chaque batch
            - features_lazy: LazyFrame des features
            - target_lazy: LazyFrame de la cible (target)
            - batch_size: Taille du batch
            - params: Paramètres de XGBoost
            - column_transformer: Transformer qui s'occupe de l'encodage et de la mise à l'échelle des données

            Yields:
            - xgb_model: Le modèle ajusté après le batch
            """
            # Collecte des données sous forme de DataFrame en mémoire
            df_features = features_lazy.collect()
            df_target = target_lazy.collect()
            print("df_features.shape = ",df_features.shape)
            # Calcul du nombre de batches
            num_batches = len(df_features) // batch_size
            if len(df_features) % batch_size != 0:
                num_batches += 1

            print('num batches =',num_batches)

            # Générer les batches et ajuster le modèle
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(df_features))

                # Sélectionner un batch de données
                X_train_batch = df_features[start_idx:end_idx]
                y_train_batch = df_target[start_idx:end_idx].to_numpy().ravel()  # Flatten en vecteur 1D

                # Appliquer la transformation du pipeline (OneHot et Scaling)
                X_train_batch_transformed = column_transformer.fit_transform(X_train_batch)
                # Convertir en DMatrix et entraîner le modèle XGBoost
                dtrain = xgb.DMatrix(X_train_batch_transformed, label=y_train_batch)
                # Entraîner le modèle XGBoost
                if xgb_model is None:
                    xgb_model = xgb.train(params, dtrain, num_boost_round=10)
                else:
                    xgb_model = xgb.train(params, dtrain, num_boost_round=10, xgb_model=xgb_model)

                yield xgb_model

        # Exemple d'utilisation dans votre boucle d'entraînement
        for xgb_model in batch_generator(df_train_features, df_train_target, BATCH_SIZE, params, column_transformer):
            # Vous pouvez faire d'autres opérations avec xgb_model après chaque batch, si nécessaire
            pass

        print("✅ Entraînement terminé !")
        # === 6. Sauvegarde du modèle ===
        save_model(xgb_model, f'result_xgb_model/xgboost_model_{region}.pkl')

        # === 7. Visualisation des prédictions ===
        df_test_features_transformed = column_transformer.transform(df_test_features.collect())

        # 2. Convertir les données de test en DMatrix
        dtest = xgb.DMatrix(df_test_features_transformed)

        # 3. Effectuer les prédictions sur les données de test
        y_pred_test = xgb_model.predict(dtest)

        # 4. Extraire les valeurs réelles (target) de df_test_target
        y_test = df_test_target.collect().to_numpy().ravel()
        save_path = f'result_xgb_model/plot_reelles_vs_predites_{region}.png'
        # 5. Plot : Comparaison des valeurs réelles et des prédictions
        dx = 5000  # Taille des intervalles (ajuste selon l'échelle de tes données)

        # Tri des données
        bins = np.arange(min(y_test), max(y_test) + dx, dx)
        bin_centers = bins[:-1] + dx / 2  # Centres des intervalles
        mean_predictions = [np.mean(y_pred_test[(y_test >= bins[i]) & (y_test < bins[i+1])])
                            for i in range(len(bins)-1)]

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred_test, alpha=0.6)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Ligne idéale
        plt.plot(bin_centers, mean_predictions, color='green', linestyle='-', marker='o',
                label="Moyenne par intervalle")
        plt.title("Prédictions vs Valeurs Réelles")
        plt.xlabel("Valeurs Réelles")
        plt.ylabel("Prédictions")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(0.75*min(y_test),1.25*max(y_test))
        plt.ylim(0.75*min(y_test),1.25*max(y_test))
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()  # Fermer la figure après sauvegarde

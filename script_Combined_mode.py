import os
from sklearn.pipeline import Pipeline
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
# Import the necessary components from your combined model module
from model_script.XGBoost_model import (
    DataCleaner,
    FeatureCreator,
    AnomalyFilter,
    WeightedPOICountsTransformer,
    data_loader,
    save_model
)
from model_script.Combined_Model import(train_combined_model,compute_shap_impact,combined_prediction)
# Define regions as in your original script
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

if __name__ == "__main__":
    # Define parameters for preprocessing and model training
    YEARS = ["2019", "2020", "2021", "2022", "2023", "2024"]
    PREPROCESSED_FILE = "data_processed/data_dvf_preprocessed_combined.csv"
    OVERWRITE = True
    BATCH_SIZE = 100_000  # if using batch training later

    # For each region
    for region, dep in regions.items():
        print(f"Processing region: {region}")
        print("Starting preprocessing by year...")

        # Load the grid data for POI weighting
        df_grid = pl.read_csv('data_pop_density/dataframe_densite&amenities_radius=500.csv')

        # Define the preprocessing pipeline
        pipeline_preprocess = Pipeline(steps=[
            ("cleaner", DataCleaner(nombre_lots_max=5, cutoff_valeur_fonciere_min=0.75e5, min_surface=15)),
            ("feature_creator", FeatureCreator(cutoff_prix_m2_min=3e3, cutoff_prix_m2_max=18e3)),
            ("weighted_poi", WeightedPOICountsTransformer(n_neighbors=4)),
            ("anomaly_filter", AnomalyFilter(contamination=0.1, target_elimination=True)),
        ])
        pipeline_preprocess.set_params(weighted_poi__df_grid=df_grid)

        # Overwrite the preprocessed file if needed
        if OVERWRITE:
            with open(PREPROCESSED_FILE, "w") as f:
                pass  # clear the file

            # Process each year separately and append to CSV
            for year in YEARS:
                print(f"Processing year {year}...")
                df_year = data_loader('data_dvf', annees=[year], departements=dep)
                print(f"Data loaded for {year}: {df_year.shape[0]} rows")
                df_processed = pipeline_preprocess.fit_transform(df_year)
                df_processed = df_processed.drop_nulls()
                print(f"Data processed for {year}: {df_processed.shape[0]} rows")
                with open(PREPROCESSED_FILE, mode="a") as f:
                    df_processed.write_csv(f, include_header=True)
            print("All years processed and saved!")

        # Load all preprocessed data as a LazyFrame
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
                'prix_m2': pl.Float64,
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

        df_train_features = df_train.drop("prix_m2")
        df_train_target = df_train.select("prix_m2")

        df_test_features = df_test.drop("prix_m2")
        df_test_target = df_test.select("prix_m2")

        # Define feature groups for the combined model
        physical_features = ['surface_reelle_bati', 'type_local', 'nombre_pieces_principales','nombre_lots','total_surface_carrez',
                     'lot1_surface_carrez', 'lot2_surface_carrez','lot3_surface_carrez', 'lot4_surface_carrez',
                     'lot5_surface_carrez','surface_terrain']

        # Caractéristiques pour le modèle localisation
        contextual_features = ['year', 'sin_month', 'cos_month','longitude', 'latitude',
            'densite_weighted', 'transport_pois_weighted', 'education_pois_weighted',
            'health_pois_weighted', 'food_pois_weighted', 'shopping_pois_weighted',
            'park_pois_weighted', 'entertainment_pois_weighted', 'cultural_pois_weighted']

        # === 2. Pipelines d'encodage ===
        categorical_columns_physical = ['type_local']
        numerical_columns_physical = ['surface_reelle_bati', 'nombre_pieces_principales','nombre_lots','total_surface_carrez',
                            'lot1_surface_carrez', 'lot2_surface_carrez','lot3_surface_carrez', 'lot4_surface_carrez',
                            'lot5_surface_carrez','surface_terrain']
        numerical_columns_contextual = ['year', 'sin_month', 'cos_month','longitude', 'latitude',
            'densite_weighted', 'transport_pois_weighted', 'education_pois_weighted',
            'health_pois_weighted', 'food_pois_weighted', 'shopping_pois_weighted',
            'park_pois_weighted', 'entertainment_pois_weighted', 'cultural_pois_weighted']

        # Catégories uniques pour l'encodage OneHot
        unique_categories = [df_lazy.select(col).drop_nulls().unique().collect().to_series().to_list()
                            for col in categorical_columns_physical]

        # Pipeline pour les caractéristiques physiques
        physical_pipeline = ColumnTransformer([
            ('onehot', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', categories=unique_categories))
            ]), categorical_columns_physical),
            ('numeric', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('scaler', RobustScaler())
            ]), numerical_columns_physical)
        ])

        # Pipeline pour les caractéristiques de localisation
        contextual_pipeline = ColumnTransformer([
            ('numeric', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('scaler', RobustScaler())
            ]), numerical_columns_contextual)
        ])

        # Define model parameters for each sub-model
        # (For non-GAM mode, physical uses ElasticNet and contextual uses XGBRegressor)
        params_physical = {
            "alpha": 0.5,
            "l1_ratio": 0.5,
            "max_iter": 1000,
            "tol": 1e-4,
            "fit_intercept": True
        }
        params_contextual = {
            "objective": "reg:squarederror",
            "learning_rate": 0.1,
            "max_depth": 8,
            "n_estimators" : 200,
            "n_jobs": 2,
            "subsample": 0.9,
            "gamma": 0.2,
            "min_child_weight": 5,
            "lambda": 1,
            "alpha": 0.3
        }

        # Initialize and train the combined model
        combined_model = train_combined_model(
            physical_params= params_physical,
            contextual_params= params_physical,
            physical_weight=0.5,
            stacking=True
        )
        print("Model training completed for region:", region)

        # Save the trained model
        save_model(combined_model, f'result_combined_model/combined_model_{region}.pkl')

        # --- Evaluate the model ---
        # Get predictions on the test set
        y_pred_test = combined_prediction(combined_model, df_test_features)

        # Plot true vs predicted values (log-scaled axes)
        y_test = df_test_target.collect().to_numpy().ravel()
        dx = 100  # Taille des intervalles (ajuste selon l'échelle de tes données)

        # Tri des données
        bins = np.arange(min(y_test), max(y_test) + dx, dx)
        bin_centers = bins[:-1] + dx / 2  # Centres des intervalles
        mean_predictions = [np.mean(y_pred_test[(y_test >= bins[i]) & (y_test < bins[i+1])])
                            for i in range(len(bins)-1)]
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred_test, alpha=0.6)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Ideal line
        plt.plot(bin_centers, mean_predictions, color='green', linestyle='-', marker='o', label="Average per interval")
        plt.title("Predictions vs True Values")
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.xlim(0.75*min(y_test), 1.1*max(y_test))
        plt.ylim(0.75*min(y_pred_test), 1.25*max(y_pred_test))

        # Compute and annotate SHAP impact results
        impact_dict = compute_shap_impact(combined_model, df_test_features, sample_size=250)
        impact_text = (
            f"Physical Model Impact: {impact_dict['physical_model_impact']:.1f}%\n"
            f"Contextual Model Impact: {impact_dict['contextual_model_impact']:.1f}%"
        )
        plt.text(0.05, 0.95, impact_text,
                transform=plt.gca().transAxes,
                fontsize=12,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

        plt.tight_layout()
        plt.show()
        plot_path = f'result_combined_model/predictions_{region}.png'
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Evaluation plot saved at {plot_path}")

        print(f"✅ Processing for region {region} completed.\n")

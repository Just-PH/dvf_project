# Preprocessing and Training Pipeline for Real Estate Data

This project implements a complete pipeline for processing real estate data (DVF) and training a combined regression model using XGBoost (and optionally GAM-based models). The pipeline is designed to handle data from multiple years and regions by applying several steps of cleaning, transformation, encoding, and aggregation, in order to produce a high-performance model capable of predicting the price per square meter of a property.

## Table of Contents

- [Overview](#overview)
- [Inputs and Outputs](#inputs-and-outputs)
- [Architecture and Workflow](#architecture-and-workflow)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Aggregation and Saving](#2-aggregation-and-saving)
  - [3. Train/Test Split](#3-traintest-split)
  - [4. Encoding Pipelines](#4-encoding-pipelines)
  - [5. Combined Model Training](#5-combined-model-training)
  - [6. Evaluation, Visualization, and SHAP Analysis](#6-evaluation-visualization-and-shap-analysis)
- [Installation and Dependencies](#installation-and-dependencies)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributions](#contributions)
- [License](#license)

## Overview

The goal of this project is to build a complete data processing chain to:
- **Preprocess** raw real estate data from the DVF.
- **Transform** the data by creating new features (e.g., total surfaces, time transformations, price per m²).
- **Filter** anomalies using an IsolationForest algorithm.
- **Calculate** additional indicators, such as the weighted count of points of interest (POI), using a KD-tree approach.
- **Train** pricing models:
  - **`script_XGBoost_model.py`**: Trains a standard pricing model using **XGBoost** on all available features, without physical/contextual separation, for a more direct and fast approach.
  - **`script_Combined_model.py`**: Trains a combined regression model integrating two sub-models: one based on physical features and the other on contextual features. The model can be configured to use standard algorithms (ElasticNet and XGBRegressor) or GAMs via a wrapper, with prediction stacking.

- **Evaluate** model performance and analyze the relative impact of sub-models using SHAP.

## Inputs and Outputs

### Inputs

- **Raw DVF Data**
  CSV files are organized by year and department (e.g., `data_dvf/2021/XX.csv.gz`). These files contain information such as:
  - `id_mutation`, `date_mutation`, `type_local`
  - `surface_reelle_bati`, `nombre_lots`, `lot1_surface_carrez`, …, `lot5_surface_carrez`
  - `nombre_pieces_principales`, `surface_terrain`
  - `longitude`, `latitude`, `valeur_fonciere`

  The directory structure looks like this:
```
  data_dvf
  ├── 2019/
  ├── 2020/
  ├── 2021/
  ├── 2022/
  ├── 2023/
  └── 2024/
      ├── 01.csv.gz
      ├── 02.csv.gz
      ├── ...
      └── 974.csv.gz
```

  The data was downloaded from the [DVF data website](https://files.data.gouv.fr/geo-dvf/latest/csv/) using `script_scrapping_dv.py`.

- **POI Data Grid**
  A CSV file (e.g., `data_pop_density/dataframe_densite&amenities_radius=500.csv`) containing coordinates (`lon`, `lat`) and various measures of density and proximity to points of interest (transport, education, health, etc.).
  This file was generated via `data_po_density/script_amenities_postgre_gist.py`, using a PostgreSQL database built from the PBF file of French geographical data from [Open Street Map](https://www.data.gouv.fr/fr/datasets/r/01fdab09-ed86-4259-b863-69913a3e04d1).

- **Configuration Parameters**
  Various parameters defined in the main script, such as:
  - Years to process (from 2019 to 2024)
  - List of departments for each region
  - Filtering thresholds for property values and surfaces
  - Parameters for encoding and scaling features
  - Model parameters (ElasticNet, XGBRegressor, or GAM options)

### Outputs

- **Preprocessed File**
  `data_processed/data_dvf_preprocessed_combined.csv` containing the cleaned, enriched, and aggregated data after all preprocessing steps.

- **Trained Model**
  One or more model files (e.g., `result_combined_model/combined_model_<region>.pkl` or `result_xgb_model/combined_model_<region>.pkl`) containing the trained combined model for each region.

- **Evaluation Charts**
  PNG files generated in the `result_combined_model/` or `result_xgb_model/` directory, displaying comparisons between actual values and predictions (with log scales and SHAP annotations, if applicable).

## Architecture and Workflow

### 1. Data Preprocessing

- **Loading**
  The `data_loader` function reads CSV files from the `data_dvf` folder based on the specified years and departments.

- **Transformation Pipeline**
  The preprocessing pipeline integrates several steps:
  - **DataCleaner**: Filters data according to defined criteria (number of lots, property value, surfaces).
  - **FeatureCreator**: Generates new variables from raw data, such as:
    - Date transformation (calculating `sin_month`, `cos_month`, extracting the year).
    - Surface aggregation and price per m² calculation.
  - **WeightedPOICountsTransformer**: Computes the weighted count of POIs using a KD-tree on the data grid.
  - **AnomalyFilter**: Removes records considered anomalies using IsolationForest.

### 2. Aggregation and Saving

- Each year is processed individually and then concatenated into a single CSV file (`data_processed/data_dvf_preprocessed_combined.csv`). This approach handles large data volumes iteratively and avoids duplicates.

### 3. Train/Test Split

- The preprocessed data is loaded using Polars in Lazy mode to optimize memory usage.
- The data is then shuffled (via a random column) and split into a training set (80%) and a test set (20%).

### 4. Encoding Pipelines

- **Physical Pipeline**
  Uses a `ColumnTransformer` that combines:
  - A pipeline for categorical variables (`OneHotEncoder` after imputation).
  - A pipeline for numerical variables (`RobustScaler` after imputation).

- **Contextual Pipeline**
  A similar pipeline for location and contextual features, focusing on scaling numerical variables.

### 5. Combined Model Training

- The `train_combined_model` function receives the training data and builds a combined model by merging two sub-models:
  - One model for physical features.
  - One model for contextual features.
- These models can either be standard (ElasticNet and XGBRegressor) or GAM-based (via GAMRegressor) depending on the `gam` parameter.
- The `stacking` option allows combining the predictions of these sub-models via a `StackingRegressor`.
- The model is then trained on the entire training dataset.

### 6. Evaluation, Visualization, and SHAP Analysis

- The `combined_prediction` function generates predictions on the test set.
- The `plot_train_test_predictions` function creates a chart comparing actual values to predictions, with log-scale axes, an ideal line, and an average curve per interval.
- The `compute_shap_impact` function calculates and returns the relative impact (as a percentage) of the physical and contextual sub-models on the final model using SHAP. These values can be annotated on the evaluation chart.

## Installation and Dependencies

The necessary dependencies are listed in the [requirements.txt](requirements.txt) file. To install the dependencies, run:

```bash
pip install -r requirements.txt
```




<!-- # Pipeline de Prétraitement et Entraînement pour Données Immobilières

Ce projet implémente un pipeline complet de traitement des données immobilières (DVF) et l'entraînement d'un modèle de régression combiné utilisant XGBoost (et éventuellement des modèles basés sur GAM). Le pipeline est conçu pour traiter des données issues de plusieurs années et régions, en appliquant plusieurs étapes de nettoyage, de transformation, d'encodage et d'agrégation, afin de produire un modèle performant capable de prédire le prix au m² d'un bien immobilier.

## Table des Matières

- [Aperçu](#aperçu)
- [Entrées et Sorties](#entrées-et-sorties)
- [Architecture et Fonctionnement](#architecture-et-fonctionnement)
  - [1. Prétraitement des Données](#1-prétraitement-des-données)
  - [2. Agrégation et Sauvegarde](#2-agrégation-et-sauvegarde)
  - [3. Séparation Train/Test](#3-séparation-traintest)
  - [4. Pipelines d'Encodage](#4-pipelines-dencodage)
  - [5. Entraînement du Modèle Combiné](#5-entrainement-du-modèle-combiné)
  - [6. Évaluation, Visualisation et Analyse SHAP](#6-évaluation-visualisation-et-analyse-shap)
- [Installation et Dépendances](#installation-et-dépendances)
- [Utilisation](#utilisation)
- [Structure du Projet](#structure-du-projet)
- [Contributions](#contributions)
- [Licence](#licence)

## Aperçu

Le but de ce projet est de construire une chaîne de traitement complète pour :
- **Prétraiter** des données immobilières brutes issues du DVF.
- **Transformer** ces données en créant de nouvelles features (ex. : surfaces totales, transformations temporelles, prix par m²).
- **Filtrer** les anomalies à l'aide d'un algorithme d'IsolationForest.
- **Calculer** des indicateurs complémentaires, comme le nombre pondéré de points d’intérêt (POI), grâce à une approche basée sur un KD-tree.
- **Entraîner** des modèles de pricing :
  - **`script_XGBoost_model.py`** : Entraîne un modèle de pricing classique utilisant **XGBoost** sur l'ensemble des caractéristiques disponibles, sans séparation physique/contextuelle, pour une approche plus directe et rapide.
  - **`script_Combined_model.py`** : Entraîne un modèle de régression combiné intégrant deux sous-modèles : l'un basé sur les caractéristiques physiques, l'autre sur les caractéristiques contextuelles. Le modèle peut être configuré pour utiliser des algorithmes standards (ElasticNet et XGBRegressor) ou des GAMs via un wrapper, avec une combinaison des prédictions par stacking.


- **Évaluer** les performances du modèle et analyser l’impact relatif des sous-modèles avec SHAP.

## Entrées et Sorties

### Entrées

- **Données brutes DVF**
  Les fichiers CSV sont organisés par année et par département (par exemple, `data_dvf/2021/XX.csv.gz`). Ces fichiers contiennent des informations telles que :
  - `id_mutation`, `date_mutation`, `type_local`
  - `surface_reelle_bati`, `nombre_lots`, `lot1_surface_carrez`, …, `lot5_surface_carrez`
  - `nombre_pieces_principales`, `surface_terrain`
  - `longitude`, `latitude`, `valeur_fonciere`

  L'architecture est la suivante :
  ```data_dvf
  ├── 2019/
  ├── 2020/
  ├── 2021/
  ├── 2022/
  ├── 2023/
  └── 2024/
      ├── 01.csv.gz
      ├── 02.csv.gz
      ├── ...
      └── 974.csv.gz

Les données ont été téléchargés sur le website des [données de valeurs foncières](https://files.data.gouv.fr/geo-dvf/latest/csv/), avec `script_scrapping_dv.py`

- **Grille de données pour POI**
  Le fichier CSV (par exemple, `data_pop_density/dataframe_densite&amenities_radius=500.csv`) contenant les coordonnées (`lon`, `lat`) et diverses mesures de densité et de proximité des points d'intérêt (transport, éducation, santé, etc.).
  Il a été obtenu via  `data_po_density/script_amenities_postgre_gist.py` qui se sert d'une BDD PostgreSQL montée à partir du fichier PBF concernant les données géographiques françaises d'[Open Street Map](https://www.data.gouv.fr/fr/datasets/r/01fdab09-ed86-4259-b863-69913a3e04d1)

- **Paramètres de configuration**
  Divers paramètres définis dans le script principal, tels que :
  - Les années à traiter (de 2019 à 2024)
  - La liste des départements pour chaque région
  - Les seuils de filtrage pour les valeurs foncières et surfaces
  - Les paramètres pour l'encodage et le scaling des features
  - Les paramètres des modèles (ElasticNet, XGBRegressor, ou options GAM)

### Sorties

- **Fichier prétraité**
  `data_processed/data_dvf_preprocessed_combined.csv` qui contient les données nettoyées, enrichies et agrégées après toutes les étapes de prétraitement.

- **Modèle entraîné**
  Un ou plusieurs fichiers modèles (par exemple, `result_combined_model/combined_model_<region>.pkl` ou `result_xgb_model/combined_model_<region>.pkl`) qui contiennent le modèle combiné entraîné pour chaque région.

- **Graphiques d'évaluation**
  Des fichiers PNG générés dans le répertoire `result_combined_model/` ou `result_xgb_model/` affichant les comparaisons entre les valeurs réelles et les prédictions (avec échelles log et annotations de SHAP, si applicable).

## Architecture et Fonctionnement

### 1. Prétraitement des Données

- **Chargement**
  La fonction `data_loader` lit les fichiers CSV du dossier `data_dvf` selon les années et départements spécifiés.

- **Pipeline de Transformation**
  Le pipeline de prétraitement intègre plusieurs étapes :
  - **DataCleaner** : Filtre les données selon les critères définis (nombre de lots, valeur foncière, surfaces).
  - **FeatureCreator** : Génère de nouvelles variables à partir des données brutes, telles que :
    - Transformation de la date (calcul de `sin_month`, `cos_month`, extraction de l'année).
    - Agrégation des surfaces et calcul du prix par m².
  - **WeightedPOICountsTransformer** : Calcule le nombre pondéré de POI en utilisant un KD-tree sur la grille de données.
  - **AnomalyFilter** : Élimine les enregistrements considérés comme anomalies grâce à IsolationForest.

### 2. Agrégation et Sauvegarde

- Chaque année est traitée individuellement puis concaténée dans un unique fichier CSV (`data_processed/data_dvf_preprocessed_combined.csv`). Cela permet de gérer les grandes volumétries de données de manière itérative et d’éviter les doublons.

### 3. Séparation Train/Test

- Les données prétraitées sont chargées en utilisant Polars en mode Lazy pour optimiser l’utilisation mémoire.
- Les données sont ensuite mélangées (via une colonne aléatoire) et divisées en un jeu d'entraînement (80%) et un jeu de test (20%).

### 4. Pipelines d'Encodage

- **Physical Pipeline**
  Utilise un `ColumnTransformer` qui combine :
  - Un pipeline pour les variables catégorielles (`OneHotEncoder` après imputation).
  - Un pipeline pour les variables numériques (`RobustScaler` après imputation).

- **Contextual Pipeline**
  Un pipeline similaire pour les caractéristiques de localisation et contextuelles, qui se concentre sur le scaling des variables numériques.

### 5. Entraînement du Modèle Combiné

- La fonction `train_combined_model` reçoit les données d'entraînement et construit un modèle combiné en fusionnant deux sous-modèles :
  - Un modèle pour les caractéristiques physiques.
  - Un modèle pour les caractéristiques contextuelles.
- Ces modèles peuvent être soit standards (ElasticNet et XGBRegressor), soit basés sur GAM (via GAMRegressor) selon le paramètre `gam`.
- L'option `stacking` permet de combiner les prédictions de ces sous-modèles via un `StackingRegressor`.
- Le modèle est ensuite entraîné sur l'ensemble des données d'entraînement.

### 6. Évaluation, Visualisation et Analyse SHAP

- La fonction `combined_prediction` est utilisée pour générer des prédictions sur le jeu de test.
- La fonction `plot_train_test_predictions` crée un graphique comparant les valeurs réelles aux prédictions, avec des axes en échelle logarithmique, une ligne idéale, et une courbe de moyenne par intervalle.
- La fonction `compute_shap_impact` calcule et retourne l'impact relatif (en pourcentage) des sous-modèles physique et contextuel sur le modèle final en utilisant SHAP. Ces valeurs peuvent être annotées sur le graphique d'évaluation.

## Installation et Dépendances

Les dépendances nécessaires sont listées dans le fichier [requirements.txt](requirements.txt). Pour installer les dépendances, exécutez :

```bash
pip install -r requirements.txt -->

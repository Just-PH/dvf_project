# Pipeline de Prétraitement et Entraînement pour Données Immobilières

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
- **Entraîner** un modèle de régression combiné qui intègre deux sous-modèles : l'un utilisant les caractéristiques physiques et l'autre les caractéristiques contextuelles. Ce modèle peut être configuré pour utiliser des modèles standards (ElasticNet et XGBRegressor) ou des GAMs via un wrapper, et peut également combiner les prédictions par stacking.
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

Les données ont été téléchargés sur le website des [données de valeurs foncières](https://files.data.gouv.fr/geo-dvf/latest/csv/)

- **Grille de données pour POI**
  Le fichier CSV (par exemple, `data_pop_density/dataframe_densite&amenities_radius=500.csv`) contenant les coordonnées (`lon`, `lat`) et diverses mesures de densité et de proximité des points d'intérêt (transport, éducation, santé, etc.).

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
  Un ou plusieurs fichiers modèles (par exemple, `result_combined_model/combined_model_<region>.pkl`) qui contiennent le modèle combiné entraîné pour chaque région.

- **Graphiques d'évaluation**
  Des fichiers PNG générés dans le répertoire `result_combined_model/` affichant les comparaisons entre les valeurs réelles et les prédictions (avec échelles log et annotations de SHAP, si applicable).

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
pip install -r requirements.txt

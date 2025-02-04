# Pipeline de Prétraitement et Entraînement XGBoost pour Données Immobilières

## Aperçu

Ce projet implémente un pipeline complet de traitement de données et d'entraînement d'un modèle XGBoost appliqué aux données immobilières (DVF). Il permet de :
- Prétraiter et transformer des données brutes issues de différentes années et régions.
- Appliquer des techniques de nettoyage, de création de features et de filtrage des anomalies.
- Construire et entraîner un modèle de régression XGBoost via un entraînement par batch.
- Sauvegarder le modèle entraîné et visualiser les prédictions comparées aux valeurs réelles sur des données de test jamais vu par les modèles.

## Fonctionnalités

- **Prétraitement des données**
  Utilisation d'un pipeline de transformation composé des étapes suivantes :
  - **DataCleaner** : Nettoyage des données (filtrage par nombre de lots, valeur foncière minimale, surface minimale).
  - **FeatureCreator** : Création de nouvelles features à partir des données brutes.
  - **WeightedPOICountsTransformer** : Calcul du nombre pondéré de points d’intérêt.
  - **AnomalyFilter** : Filtrage des anomalies avec une contamination fixée.

- **Agrégation des données prétraitées**
  Traitement itératif des données pour chaque année (2019 à 2024) et chaque région, avec sauvegarde dans un fichier CSV.

- **Séparation Train/Test**
  Utilisation de Polars en mode Lazy pour optimiser la mémoire, mélange des données et séparation en jeux d'entraînement (80%) et de test (20%).

- **Pipeline d'encodage et transformation**
  Encodage OneHot pour les variables catégorielles et transformation/scaling des variables numériques.

- **Entraînement du modèle XGBoost**
  Entraînement itératif par batch grâce à une fonction génératrice, permettant d'ajuster le modèle sur des sous-ensembles de données.

- **Sauvegarde du modèle et visualisation**
  Sauvegarde du modèle pour chaque région et création d'un graphique comparant les valeurs réelles et les prédictions (échelles log, moyenne par intervalle).

## Installation et Dépendances

Les dépendances nécessaires sont listées dans le fichier [`requirements.txt`](requirements.txt). Pour installer les dépendances, exécutez :

```bash
pip install -r requirements.txt

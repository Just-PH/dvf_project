import polars as pl
import pandas as pd
import numpy as np
import psycopg2
from tqdm import tqdm

def get_poi_counts(latitude, longitude, radius, db_params):
    """
    Récupère les comptes des POI autour d'une localisation donnée.

    Args:
        latitude (float): Latitude de la localisation.
        longitude (float): Longitude de la localisation.
        db_params (dict): Dictionnaire contenant les informations de connexion à la base de données.

    Returns:
        dict: Dictionnaire contenant les comptes des POI.
    """
    query = """
    SELECT
        COUNT(CASE WHEN public_transport IN ('stop_position', 'station', 'subway', 'railway') THEN 1 END) AS transport_pois,
        COUNT(CASE WHEN amenity IN ('school', 'college', 'prep_school') THEN 1 END) AS education_pois,
        COUNT(CASE WHEN amenity IN ('hospital', 'clinic', 'pharmacy', 'doctors') THEN 1 END) AS health_pois,
        COUNT(CASE WHEN amenity IN ('restaurant', 'cafe', 'bar', 'pub', 'fast_food') THEN 1 END) AS food_pois,
        COUNT(CASE WHEN amenity IN ('marketplace') OR shop IS NOT NULL THEN 1 END) AS shopping_pois,
        COUNT(CASE WHEN leisure IN ('park') THEN 1 END) AS park_pois,
        COUNT(CASE WHEN amenity IN ('cinema', 'theatre', 'bowling', 'nightclub') THEN 1 END) AS entertainment_pois,
        COUNT(CASE WHEN amenity IN ('library', 'community_centre', 'arts_centre', 'museum') THEN 1 END) AS cultural_pois
    FROM public.planet_osm_point
    WHERE ST_DWithin(
        ST_Transform(way, 3857),
        ST_Transform(ST_SetSRID(ST_MakePoint(%s, %s), 4326), 3857),
        %s
    )
    AND (
        (public_transport IN ('stop_position', 'station', 'subway', 'railway') AND name IS NOT NULL)
        OR
        (amenity IN ('school', 'college', 'prep_school') AND name IS NOT NULL)
        OR
        (amenity IN ('hospital', 'clinic', 'pharmacy', 'doctors') AND name IS NOT NULL)
        OR
        (amenity IN ('restaurant', 'cafe', 'bar', 'pub', 'fast_food') AND name IS NOT NULL)
        OR
        ((amenity IN ('marketplace') OR shop IS NOT NULL) AND name IS NOT NULL)
        OR
        (leisure IN ('park') AND name IS NOT NULL)
        OR
        (amenity IN ('cinema', 'theatre', 'bowling', 'nightclub') AND name IS NOT NULL)
        OR
        (amenity IN ('library', 'community_centre', 'arts_centre', 'museum') AND name IS NOT NULL)
    );
    """
    # Connexion à la base de données
    conn = None
    try:
        conn = psycopg2.connect(
            host=db_params['host'],
            port=db_params['port'],
            dbname=db_params['dbname'],
            user=db_params['user'],
            password=db_params['password']
        )
        with conn.cursor() as cur:
            # Exécuter la requête
            cur.execute(query, (longitude, latitude, radius))
            # Récupérer les résultats
            columns = [desc[0] for desc in cur.description]
            data = cur.fetchone()  # Seulement une ligne avec les résultats
            if data:
                result = dict(zip(columns, data))
                return result
    except Exception as e:
        print(f"Erreur lors de l'exécution de la requête : {e}")
    finally:
        if conn:
            conn.close()
    return {}

def add_poi_counts_to_df(df, db_params, radius=500):
    """
    Ajoute les colonnes POI à un DataFrame existant avec des latitudes et longitudes.

    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes 'lat' et 'lon'.
        db_params (dict): Paramètres de connexion à la base de données.
        radius (int): Rayon de recherche pour les POIs autour des coordonnées.

    Returns:
        pd.DataFrame: DataFrame avec les nouvelles colonnes POI.
    """
    df = df.to_pandas()
    tqdm.pandas()
    df['poi_counts'] = df.progress_apply(lambda row: get_poi_counts(row['lat'], row['lon'], radius, db_params), axis=1)

    # Convertir les résultats des POIs (qui sont des dictionnaires) en DataFrame
    poi_df = pd.json_normalize(df['poi_counts'])

    # Ajouter les nouvelles colonnes au DataFrame d'origine
    df = pd.concat([df.drop(columns=['poi_counts']), poi_df], axis=1)

    return pl.from_pandas(df)


if __name__ == "__main__":
    print("Loading data:")
    path_popd ='data_pop_density/fra_pd_2020_1km_ASCII_XYZ.csv'
    df_popd = pl.read_csv(path_popd).rename({'X': 'lon','Y':'lat','Z':'densite'})
    db_params = {
    'dbname': 'osm_db',
    'user': 'phdel',
    'port' : 5432,
    'password': 'your_password',
    'host': 'localhost'
}
    print("Data loaded")
    # Appliquer la fonction sur votre DataFrame
    df_pop_better = add_poi_counts_to_df(df_popd, db_params)

    # Afficher le DataFrame final
    df_pop_better.write_csv("data_pop_density/dataframe_densite&amenities.csv")
    print('Data saved')

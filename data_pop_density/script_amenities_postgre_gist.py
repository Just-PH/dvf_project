import polars as pl
import pandas as pd
import numpy as np
import psycopg2
from tqdm import tqdm

def get_poi_counts(radius, db_params):
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
        c.lat,
        c.lon,
        c.densite,
        COUNT(CASE WHEN p.public_transport IN ('stop_position', 'station', 'subway', 'railway') THEN 1 END) AS transport_pois,
        COUNT(CASE WHEN p.amenity IN ('school', 'college', 'prep_school') THEN 1 END) AS education_pois,
        COUNT(CASE WHEN p.amenity IN ('hospital', 'clinic', 'pharmacy', 'doctors') THEN 1 END) AS health_pois,
        COUNT(CASE WHEN p.amenity IN ('restaurant', 'cafe', 'bar', 'pub', 'fast_food') THEN 1 END) AS food_pois,
        COUNT(CASE WHEN p.amenity IN ('marketplace') OR p.shop IS NOT NULL THEN 1 END) AS shopping_pois,
        COUNT(CASE WHEN p.leisure IN ('park') THEN 1 END) AS park_pois,
        COUNT(CASE WHEN p.amenity IN ('cinema', 'theatre', 'bowling', 'nightclub') THEN 1 END) AS entertainment_pois,
        COUNT(CASE WHEN p.amenity IN ('library', 'community_centre', 'arts_centre', 'museum') THEN 1 END) AS cultural_pois
    FROM coord_density c
    LEFT JOIN (
        SELECT way, public_transport, amenity, shop, leisure, name
        FROM public.planet_osm_point
        WHERE
            public_transport IN ('stop_position', 'station', 'subway', 'railway')
            OR amenity IN ('school', 'college', 'prep_school', 'hospital', 'clinic', 'pharmacy', 'doctors',
                        'restaurant', 'cafe', 'bar', 'pub', 'fast_food', 'marketplace', 'cinema',
                        'theatre', 'bowling', 'nightclub', 'library', 'community_centre',
                        'arts_centre', 'museum')
            OR shop IS NOT NULL
            OR leisure IN ('park')
    ) p
    ON ST_DWithin(
        ST_Transform(p.way, 3857),
        ST_Transform(ST_SetSRID(ST_MakePoint(c.lon, c.lat), 4326), 3857),
        %s
    )/*
    WHERE p.name IS NOT NULL
    */
    GROUP BY c.lat, c.lon, c.densite;
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
            cur.execute(query, (radius,))
            # Récupérer les résultats
            columns = [desc[0] for desc in cur.description]
            data = cur.fetchall()
            if data:
                result = pd.DataFrame(data, columns=columns)
                return result
    except Exception as e:
        print(f"Erreur lors de l'exécution de la requête : {e}")
    finally:
        if conn:
            conn.close()
    return pd.DataFrame()

if __name__ == "__main__":
    print("Loading data:")
    db_params = {
    'dbname': 'osm_db',
    'user': 'phdel',
    'port' : 5432,
    'password': 'your_password',
    'host': 'localhost'
    }
    radius =500
    df_pop_better = get_poi_counts(radius=radius, db_params=db_params)
    # Afficher le DataFrame final
    df_pop_better.to_csv(f"data_pop_density/dataframe_densite&amenities_radius={radius}.csv")
    print('Data saved')

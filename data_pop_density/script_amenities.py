import polars as pl
import time
import numpy as np
import requests
import os

def fetch_all_pois(lat, lon, radius=500):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      node["amenity"](around:{radius}, {lat}, {lon});
      way["amenity"](around:{radius}, {lat}, {lon});
      relation["amenity"](around:{radius}, {lat}, {lon});
    );
    out body;
    """
    retries = 3
    for i in range(retries):
        try:
            response = requests.get(overpass_url, params={'data': overpass_query})
            response.raise_for_status()  # Will raise an HTTPError for bad status codes
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            if i < retries - 1:
                print(f"Retrying... ({i+1}/{retries})")
                time.sleep(5)  # Wait for 5 seconds before retrying
            else:
                raise Exception(f"Failed to fetch data after {retries} attempts")


def dict_amenities(lat,lon,dens,data):
    results = {"lat": lat, "lon": lon, "densite": dens, "amenities": 0, "transports": 0, "shops": 0, "schools": 0,
               "parks": 0, "restaurants": 0, "healthcare": 0,
               "entertainment": 0, "cultural_places": 0}

    for element in data.get("elements", []):
        tags = element.get("tags", {})


        if tags.get("amenity") in {"school", "college", "prep_school"}:
            results["schools"] += 1
        elif tags.get("amenity") in {"hospital", "clinic", "pharmacy", "doctors"}:
            results["healthcare"] += 1
        elif tags.get("amenity") in {"restaurant", "cafe", "bar", "pub", "fast_food"}:
            results["restaurants"] += 1
        elif tags.get("amenity") in {"marketplace", "shop"}:
            results["shops"] += 1
        elif tags.get("leisure") == "park":
            results["parks"] += 1
        elif tags.get("amenity") in {"cinema", "theatre", "bowling", "nightclub"}:
            results["entertainment"] += 1
        elif tags.get("amenity") in {"library", "community_centre", "arts_centre"} or tags.get("tourism") == "museum":
            results["cultural_places"] += 1
        else:
            results["amenities"] += 1

    return results

def save_partial_results(filename, df):
    df.write_csv(filename)

def enrich_with_combined_overpass(df, radius=500, backup_file="backup.csv", min_density = 200, save_interval=1000,backup_status = False):
    print("Enrichissement des données avec les résultats des fonctions Overpass API...")

    # Initialisation des listes pour les colonnes à ajouter

    result = pl.DataFrame()
    # Parcourir les lignes du DataFrame
    start_idx = 0
    if backup_status and os.path.exists(backup_file):
        print(f"Reprise à partir du fichier de sauvegarde : {backup_file}")
        result = pl.read_csv(backup_file)
        df1 = df.with_columns((pl.col("lon").cast(str) + "," + pl.col("lat").cast(str) + "," + pl.col("densite").cast(str)).alias("key"))
        df2 = result.with_columns((pl.col("lon").cast(str) + "," + pl.col("lat").cast(str) + "," + pl.col("densite").cast(str)).alias("key"))
        df = df1.filter(~pl.col("key").is_in(df2["key"]))[['lon','lat','densite']]
        print(f"Reprise des calculs, il reste {df.shape[0]} lignes")
        # Filtrage des lignes déjà traitées
        rows_to_process = df.rows()

    for idx, row in enumerate(rows_to_process):
        lon, lat , dens = row[0], row[1], row[2]
        if idx < start_idx:
            continue

        if dens >= min_density:
            print(f"Traitement pour ({lat}, {lon})...")
            response = fetch_all_pois(lat, lon, radius)
            dict_amenity = dict_amenities(lat, lon, dens, response)
        else:
            dict_amenity = dict_amenities(lat,lon,dens, {})
        if result.is_empty():
            result = pl.DataFrame(dict_amenity)
        else:
            partial_df = pl.DataFrame(dict_amenity)
            result = pl.concat([result,partial_df])
        # Sauvegarder les résultats partiels à intervalles réguliers
        if (idx + 1) % save_interval == 0:
            print(f"Sauvegarde intermédiaire après {idx + 1} lignes...")
            save_partial_results(backup_file, result)

    # Ajouter les nouvelles colonnes au DataFrame
    return result

if __name__ == "__main__":
    print("Loading data:")
    path_popd ='data_pop_density/fra_pd_2020_1km_ASCII_XYZ.csv'
    df_popd = pl.read_csv(path_popd).rename({'X': 'lon','Y':'lat','Z':'densite'})
    print("Data loaded")
    df_popd = enrich_with_combined_overpass(df_popd, radius=500, backup_file="data_pop_density/pois_script_result copy.csv",min_density=2000, save_interval=1000,backup_status=True)
    df_popd.write_csv("data_pop_density/dataframe_densite_amenities.csv")
    print("Script done")

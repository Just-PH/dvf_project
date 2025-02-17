import os
import requests
from bs4 import BeautifulSoup

# Base URL
base_url = "https://files.data.gouv.fr/geo-dvf/latest/csv/"

# Répertoire de sauvegarde
output_dir = "data_dvf"
os.makedirs(output_dir, exist_ok=True)

def fetch_years(base_url):
    """
    Récupère les années disponibles à partir de la base URL.
    """
    response = requests.get(base_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    years = [link.text.strip('/') for link in soup.find_all('a') if link.text.strip('/').isdigit()]
    return years

def fetch_gz_files(year_url):
    """
    Récupère tous les fichiers .gz dans le sous-dossier 'departements' d'une année donnée.
    """
    department_url = f"{year_url}departements/"
    response = requests.get(department_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    gz_files = [link['href'] for link in soup.find_all('a', href=True) if link['href'].endswith('.gz')]
    return department_url, gz_files

def download_file(url, output_path):
    """
    Télécharge un fichier depuis une URL donnée.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def main():
    years = fetch_years(base_url)
    for year in years:
        year_url = f"{base_url}{year}/"
        print(f"Fetching data for year: {year}")

        # Créer un répertoire pour chaque année
        year_dir = os.path.join(output_dir, year)
        os.makedirs(year_dir, exist_ok=True)

        # Récupérer les fichiers .gz dans 'departements'
        department_url, gz_files = fetch_gz_files(year_url)

        for gz_file in gz_files:
            file_url = f"{department_url}{gz_file}"
            output_file = os.path.join(year_dir, gz_file)

            # Télécharger le fichier si non déjà présent
            if not os.path.exists(output_file):
                print(f"Downloading: {gz_file}")
                download_file(file_url, output_file)
            else:
                print(f"File already exists: {gz_file}")

if __name__ == "__main__":
    main()

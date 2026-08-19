"""
download_noaa.py
Télécharge et dézippe automatiquement les fichiers AIS NOAA
pour une plage de dates donnée.
Streaming vers fichier temporaire + retry pour la robustesse.
"""

import requests
import zipfile
import zlib
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path

OUTPUT_FOLDER = Path("../../data/raw/noaa_gulf/")
BASE_URL = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{year}_{month:02d}_{day:02d}.zip"

MAX_RETRIES = 3
CHUNK_SIZE = 1024 * 1024  # 1 Mo


def daterange(start: date, end: date):
    for n in range((end - start).days + 1):
        yield start + timedelta(n)


def download_and_extract(day: date):
    url = BASE_URL.format(year=day.year, month=day.month, day=day.day)
    csv_name = f"AIS_{day.year}_{day.month:02d}_{day.day:02d}.csv"
    output_path = OUTPUT_FOLDER / csv_name

    if output_path.exists():
        print(f"{csv_name} : déjà présent, on saute")
        return

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Téléchargement {url} (tentative {attempt}/{MAX_RETRIES}) ...")
            response = requests.get(url, stream=True, timeout=(30, 300))

            if response.status_code != 200:
                print(f"  -> échec ({response.status_code}), fichier probablement indisponible")
                return  # pas la peine de réessayer, c'est un 404 ou similaire

            # Écrire le zip dans un fichier temporaire par chunks
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                tmp_path = Path(tmp.name)
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        tmp.write(chunk)

            # Extraire depuis le fichier temporaire
            with zipfile.ZipFile(tmp_path) as z:
                z.extractall(OUTPUT_FOLDER)

            tmp_path.unlink(missing_ok=True)
            print(f"  -> extrait dans {OUTPUT_FOLDER}")
            return  # succès

        except (requests.exceptions.RequestException, zipfile.BadZipFile, zlib.error, OSError) as e:
            print(f"  -> erreur : {e}")
            # Nettoyer le fichier temporaire si il existe
            try:
                tmp_path.unlink(missing_ok=True)
            except (NameError, OSError):
                pass
            if attempt < MAX_RETRIES:
                wait = 10 * attempt
                print(f"  -> nouvelle tentative dans {wait}s ...")
                time.sleep(wait)
            else:
                print(f"  -> ABANDON après {MAX_RETRIES} tentatives pour {csv_name}")


if __name__ == "__main__":
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    start_date = date(2024, 10, 18)
    end_date = date(2024, 12, 23)

    for day in daterange(start_date, end_date):
        download_and_extract(day)

    print("\nTerminé.")
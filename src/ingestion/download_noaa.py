"""
download_noaa.py
Télécharge et dézippe automatiquement les fichiers AIS NOAA
pour une plage de dates donnée.
"""

import requests
import zipfile
import io
from datetime import date, timedelta
from pathlib import Path

OUTPUT_FOLDER = Path("../../data/raw/noaa_gulf/")
BASE_URL = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{year}_{month:02d}_{day:02d}.zip"


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

    print(f"Téléchargement {url} ...")
    response = requests.get(url, stream=True, timeout=120)

    if response.status_code != 200:
        print(f"  -> échec ({response.status_code}), fichier probablement indisponible pour cette date")
        return

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(OUTPUT_FOLDER)

    print(f"  -> extrait dans {OUTPUT_FOLDER}")


if __name__ == "__main__":
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    start_date = date(2024, 12, 1)
    end_date = date(2024, 12, 23)  # tu as déjà 24-31

    for day in daterange(start_date, end_date):
        download_and_extract(day)

    print("\nTerminé.")
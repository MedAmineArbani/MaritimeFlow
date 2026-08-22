"""
run_batches.py
Exécute le pipeline d'ingestion par lots (batches) de 2 mois pour économiser l'espace disque.
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
DATA_INTERIM = ROOT / "data" / "interim"

LOTS = [
    {"name": "janv_fev_2023",   "start": "2023-01-01", "end": "2023-02-28", "skip_download": False},
    {"name": "mars_avril_2023", "start": "2023-03-01", "end": "2023-04-30", "skip_download": False},
    {"name": "mai_juin_2023",   "start": "2023-05-01", "end": "2023-06-30", "skip_download": False},
]

def run_step(folder: str, script: str, args: list = None):
    path = ROOT / folder
    print(f"\n{'='*60}")
    print(f"> {folder}/{script} {' '.join(args) if args else ''}")
    print(f"{'='*60}")

    cmd = [sys.executable, "-u", script]
    if args:
        cmd.extend(args)

    result = subprocess.run(cmd, cwd=path)

    if result.returncode != 0:
        print(f"ERREUR dans {script} (code {result.returncode})")
        sys.exit(1)


def rename_file(src_name: str, dst_name: str):
    src = DATA_INTERIM / src_name
    dst = DATA_INTERIM / dst_name
    if src.exists():
        # Si la destination existe déjà d'un run précédent, on l'écrase
        if dst.exists():
            dst.unlink()
        src.rename(dst)
        print(f"Renommé : {src_name} -> {dst_name}")
    else:
        print(f"Attention : {src_name} introuvable pour le renommage.")


if __name__ == "__main__":
    for lot in LOTS:
        print(f"\n\n{'#'*80}")
        print(f"### DÉMARRAGE DU TRAITEMENT : {lot['name'].upper()} ({lot['start']} au {lot['end']}) ###")
        print(f"{'#'*80}")

        # Vérifier si ce lot est déjà entièrement traité
        if (DATA_INTERIM / f"stay_points_{lot['name']}.csv").exists():
            print(f"\n-> Lot {lot['name']} déjà terminé (stay_points_{lot['name']}.csv existe), on passe.")
            continue

        # Déterminer à quelle étape reprendre
        has_resampled = (DATA_INTERIM / "trajectories_resampled.csv").exists()
        has_trajectories = (DATA_INTERIM / "trajectories.csv").exists()
        has_cleaned = len(list(DATA_INTERIM.glob("*_clean.csv"))) > 0

        if has_resampled:
            print("\n-> trajectories_resampled.csv existe déjà, on saute directement aux points d'arrêt.")
        elif has_trajectories:
            print("\n-> trajectories.csv existe déjà, on reprend au ré-échantillonnage.")
            run_step("src/ingestion", "resample_trajectories.py")
        elif has_cleaned:
            print("\n-> Fichiers *_clean.csv trouvés, on reprend à la construction des trajectoires.")
            # 3. Construction et segmentation des trajectoires
            run_step("src/ingestion", "build_trajectories.py")
            # 4. Ré-échantillonnage
            run_step("src/ingestion", "resample_trajectories.py")
        else:
            # Pipeline complet depuis le début
            # 1. Téléchargement
            if lot.get("skip_download"):
                print("\n-> Téléchargement ignoré (données déjà présentes)")
            else:
                run_step("src/ingestion", "download_noaa.py", ["--start", lot["start"], "--end", lot["end"]])

            # 2. Nettoyage
            run_step("src/ingestion", "clean_ais.py")

            # 3. Construction et segmentation des trajectoires
            run_step("src/ingestion", "build_trajectories.py")

            # 4. Ré-échantillonnage
            run_step("src/ingestion", "resample_trajectories.py")

        # 5. Points d'arrêt
        run_step("src/ports", "detect_stay_points.py")

        # 6. Renommage des fichiers de sortie avec le suffixe du lot
        print("\n--- Renommage des fichiers pour le", lot["name"], "---")
        rename_file("stay_points.csv", f"stay_points_{lot['name']}.csv")
        rename_file("trajectories.csv", f"trajectories_{lot['name']}.csv")
        rename_file("trajectories_resampled.csv", f"trajectories_resampled_{lot['name']}.csv")
        rename_file("vessel_info.csv", f"vessel_info_{lot['name']}.csv")

        # Note : clean_ais.py supprime déjà les raw CSVs
        # Note : build_trajectories.py supprime déjà les *_clean.csv

    print(f"\n{'='*60}")
    print("Traitement par lots terminé avec succès.")
    print(f"{'='*60}")


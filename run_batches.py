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
    {"name": "juil_aout",  "start": "2024-07-01", "end": "2024-08-31", "skip_download": True},
    {"name": "sept_oct",   "start": "2024-09-01", "end": "2024-10-31", "skip_download": False},
    {"name": "nov_dec",    "start": "2024-11-01", "end": "2024-12-31", "skip_download": False},
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

        # 1. Téléchargement pour le lot spécifique (sauf si déjà téléchargé)
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
    print("Fichiers générés : stay_points_lot1.csv, trajectories_lot1.csv, etc.")
    print(f"{'='*60}")

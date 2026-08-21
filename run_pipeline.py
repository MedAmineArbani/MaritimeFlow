"""
run_pipeline.py
Exécute l'ensemble du pipeline de traitement des données AIS, dans l'ordre.
Chaque étape est lancée depuis son propre dossier (chemins relatifs internes).
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

PIPELINE_STEPS = [
    ("src/ingestion", "download_noaa.py"),
    ("src/ingestion", "clean_ais.py"),
    ("src/ingestion", "build_trajectories.py"),
    ("src/ingestion", "resample_trajectories.py"),
    ("src/ports", "detect_stay_points.py"),
    #("src/ports", "cluster_ports_dbscan.py"),
    #("src/ports", "validate_ports.py"),
    #("src/network", "od_matrix.py"),
]


def run_step(folder: str, script: str):
    path = ROOT / folder
    print(f"\n{'='*60}")
    print(f"> {folder}/{script}")
    print(f"{'='*60}")

    result = subprocess.run(
        [sys.executable, "-u", script],
        cwd=path,
    )

    if result.returncode != 0:
        print(f"ERREUR dans {script} (code {result.returncode})")
        sys.exit(1)


if __name__ == "__main__":
    for folder, script in PIPELINE_STEPS:
        run_step(folder, script)

    print(f"\n{'='*60}")
    print("Pipeline complet exécuté avec succès")
    print(f"{'='*60}")
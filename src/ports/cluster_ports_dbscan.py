"""Clustering DBSCAN (métrique haversine) pour détecter les ports à partir des stay points."""
"""
cluster_ports_dbscan.py
Regroupe les points d'arrêt en clusters géographiques via DBSCAN
(métrique haversine) — chaque cluster dense = port ou mouillage candidat.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

EPS_KM = 5           # rayon de voisinage
MIN_SAMPLES = 15      # nombre minimal de points pour former un cluster dense
EARTH_RADIUS_KM = 6371


def cluster_ports(stay_points: pd.DataFrame) -> pd.DataFrame:
    """Applique DBSCAN sur les points d'arrêt, ajoute la colonne port_cluster."""
    coords_rad = np.radians(stay_points[["lat", "lon"]].values)
    eps_rad = EPS_KM / EARTH_RADIUS_KM

    db = DBSCAN(eps=eps_rad, min_samples=MIN_SAMPLES, metric="haversine")
    stay_points = stay_points.copy()
    stay_points["port_cluster"] = db.fit_predict(coords_rad)
    # port_cluster == -1 signifie "bruit", pas un port
    return stay_points


def summarize_clusters(clustered: pd.DataFrame) -> pd.DataFrame:
    """Résumé par cluster : centroïde, nombre de points, durée moyenne."""
    ports = clustered[clustered["port_cluster"] != -1]
    summary = ports.groupby("port_cluster").agg(
        centroid_lat=("lat", "mean"),
        centroid_lon=("lon", "mean"),
        n_stay_points=("lat", "count"),
        avg_duration_hours=("duration_hours", "mean"),
    ).reset_index()
    return summary.sort_values("n_stay_points", ascending=False)


if __name__ == "__main__":
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    interim_dir = PROJECT_ROOT / "data" / "interim"
    processed_dir = PROJECT_ROOT / "data" / "processed"

    stay_points = pd.read_csv(interim_dir / "stay_points.csv")

    clustered = cluster_ports(stay_points)
    summary = summarize_clusters(clustered)

    print("Nombre de clusters détectés (ports/mouillages candidats) :", len(summary))
    print("Nombre de points classés comme bruit (-1) :", (clustered["port_cluster"] == -1).sum())
    print("\nTop 10 clusters par nombre de points :")
    print(summary.head(10))

    clustered.to_csv(interim_dir / "stay_points_clustered.csv", index=False)
    summary.to_csv(processed_dir / "ports_candidates.csv", index=False)
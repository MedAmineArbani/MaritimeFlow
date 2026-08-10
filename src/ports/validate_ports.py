"""
validate_ports.py
Recoupe les clusters DBSCAN (ports candidats) avec le World Port Index
pour confirmer/nommer les vrais ports, et distinguer port réel vs mouillage.
"""

import numpy as np
import pandas as pd
import geopandas as gpd

MAX_DISTANCE_KM = 10  # distance max pour considérer un match valide


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_wpi(shp_path: str) -> pd.DataFrame:
    """Charge le World Port Index."""
    wpi = gpd.read_file(shp_path)
    wpi["wpi_lat"] = wpi["LATITUDE"]
    wpi["wpi_lon"] = wpi["LONGITUDE"]
    return wpi


def validate_clusters(clusters: pd.DataFrame, wpi: pd.DataFrame) -> pd.DataFrame:
    """Pour chaque cluster, trouve le port WPI le plus proche et calcule la distance."""
    results = []
    for _, cluster in clusters.iterrows():
        distances = haversine_km(
            cluster["centroid_lat"], cluster["centroid_lon"],
            wpi["wpi_lat"], wpi["wpi_lon"]
        )
        idx_min = distances.idxmin()
        min_dist = distances.min()

        results.append({
            **cluster.to_dict(),
            "matched_port_name": wpi.loc[idx_min, "PORT_NAME"] if min_dist <= MAX_DISTANCE_KM else None,
            "distance_to_port_km": min_dist,
            "is_validated_port": min_dist <= MAX_DISTANCE_KM,
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    clusters = pd.read_csv("../../data/processed/ports_candidates.csv")
    wpi = load_wpi("../../data/raw/world_port_index/WPI.shp")

    validated = validate_clusters(clusters, wpi)
    print(validated[["port_cluster", "matched_port_name", "distance_to_port_km", "is_validated_port"]])

    validated.to_csv("../../data/processed/ports_validated.csv", index=False)
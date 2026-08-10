"""Détection des points d'arrêt (stay points) par navire."""
"""
detect_stay_points.py
Détecte, pour chaque trajet, les segments où le navire reste
immobile (point d'arrêt = candidat port ou zone de mouillage).
"""

import pandas as pd
import numpy as np

SPEED_THRESHOLD_KNOTS = 1.0      # en dessous, on considère le bateau "arrêté"
MIN_STAY_HOURS = 2.0              # durée minimale pour être un point d'arrêt


def haversine_km(lat1, lon1, lat2, lon2):
    """Distance haversine entre deux points GPS, en km."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def detect_stay_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque trajet (trajectory_id), identifie les points où le
    bateau reste quasi immobile (SOG faible) pendant plus de MIN_STAY_HOURS.
    Retourne un point représentatif (moyenne lat/lon) par segment d'arrêt.
    """
    df = df.sort_values(["trajectory_id", "BaseDateTime"]).copy()

    # marque les points "lents" (candidats arrêt)
    df["is_slow"] = df["SOG"].fillna(0) <= SPEED_THRESHOLD_KNOTS

    # groupe les points lents consécutifs au sein d'un même trajet
    df["slow_block"] = (
        df.groupby("trajectory_id")["is_slow"]
        .apply(lambda s: (s != s.shift()).cumsum())
        .reset_index(drop=True)
    )
    df["block_id"] = df["trajectory_id"].astype(str) + "_" + df["slow_block"].astype(str)

    stay_points = []
    for block_id, group in df[df["is_slow"]].groupby("block_id"):
        duration = (group["BaseDateTime"].max() - group["BaseDateTime"].min()).total_seconds() / 3600
        if duration >= MIN_STAY_HOURS:
            stay_points.append({
                "block_id": block_id,
                "MMSI": group["MMSI"].iloc[0],
                "trajectory_id": group["trajectory_id"].iloc[0],
                "lat": group["LAT"].mean(),
                "lon": group["LON"].mean(),
                "duration_hours": duration,
                "start_time": group["BaseDateTime"].min(),
                "end_time": group["BaseDateTime"].max(),
            })

    return pd.DataFrame(stay_points)


if __name__ == "__main__":
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    interim_dir = PROJECT_ROOT / "data" / "interim"

    df = pd.read_csv(interim_dir / "trajectories_resampled.csv")
    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"])

    stay_points = detect_stay_points(df)

    print("Nombre de points d'arrêt détectés :", len(stay_points))
    stay_points.to_csv(interim_dir / "stay_points.csv", index=False)
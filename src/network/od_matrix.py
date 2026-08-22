"""
od_matrix.py
Reconstruit le réseau origine-destination : pour chaque trajet,
associe le point de départ/arrivée au port validé le plus proche,
calcule le temps de trajet réel et la distance parcourue.
Filtre les segments à vitesse implicite aberrante (sauts GPS / trous de signal).
"""

import numpy as np
import pandas as pd

MAX_DISTANCE_TO_PORT_KM = 15
MAX_SEGMENT_SPEED_KMH = 40  # ~22 noeuds, marge au-dessus du max vraquier (18 noeuds)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def nearest_port(lat, lon, ports: pd.DataFrame):
    valid_ports = ports[ports["is_validated_port"]]
    distances = haversine_km(lat, lon, valid_ports["centroid_lat"], valid_ports["centroid_lon"])
    idx_min = distances.idxmin()
    return valid_ports.loc[idx_min, "matched_port_name"], distances.min()


def build_od_matrix(trajectories: pd.DataFrame, ports: pd.DataFrame) -> pd.DataFrame:
    records = []

    for traj_id, group in trajectories.groupby("trajectory_id"):
        group = group.sort_values("BaseDateTime").reset_index(drop=True)
        start = group.iloc[0]
        end = group.iloc[-1]

        origin_port, origin_dist = nearest_port(start["LAT"], start["LON"], ports)
        dest_port, dest_dist = nearest_port(end["LAT"], end["LON"], ports)

        if origin_dist > MAX_DISTANCE_TO_PORT_KM or dest_dist > MAX_DISTANCE_TO_PORT_KM:
            continue

        duration_hours = (end["BaseDateTime"] - start["BaseDateTime"]).total_seconds() / 3600
        if duration_hours <= 0:
            continue

        # distance segment par segment, avec filtrage des sauts aberrants
        lat_shift = group["LAT"].shift()
        lon_shift = group["LON"].shift()
        time_shift = group["BaseDateTime"].shift()

        seg_dist = haversine_km(lat_shift, lon_shift, group["LAT"], group["LON"])
        seg_time_h = (group["BaseDateTime"] - time_shift).dt.total_seconds() / 3600
        seg_speed = seg_dist / seg_time_h.replace(0, np.nan)

        # on ne garde que les segments dont la vitesse implicite est réaliste
        valid_segments = seg_speed <= MAX_SEGMENT_SPEED_KMH
        total_distance_km = seg_dist[valid_segments].sum()
        excluded_segments = (~valid_segments).sum() - 1  # -1 car le premier point n'a pas de segment

        records.append({
            "trajectory_id": traj_id,
            "MMSI": start["MMSI"],
            "origin_port": origin_port,
            "destination_port": dest_port,
            "start_time": start["BaseDateTime"],
            "end_time": end["BaseDateTime"],
            "duration_hours": duration_hours,
            "distance_km": total_distance_km,
            "avg_speed_kmh": total_distance_km / duration_hours,
            "excluded_aberrant_segments": max(excluded_segments, 0),
        })

    return pd.DataFrame(records)

if __name__ == "__main__":
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    interim_dir = PROJECT_ROOT / "data" / "interim"
    processed_dir = PROJECT_ROOT / "data" / "processed"

    traj_files = [
        "trajectories_resampled_janv_fev_2023.csv",
        "trajectories_resampled_mars_avril_2023.csv",
        "trajectories_resampled_mai_juin_2023.csv",
        "trajectories_resampled_juil_aout.csv",
        "trajectories_resampled_sept_oct.csv",
        "trajectories_resampled_nov_dec.csv",
    ]

    traj_dfs = []
    for f in traj_files:
        d = pd.read_csv(interim_dir / f, low_memory=False)
        lot_name = f.replace("trajectories_resampled_", "").replace(".csv", "")
        d["trajectory_id"] = d["trajectory_id"] + "_" + lot_name  # rend l'ID unique par lot
        traj_dfs.append(d)

    trajectories = pd.concat(traj_dfs, ignore_index=True)
    trajectories["BaseDateTime"] = pd.to_datetime(trajectories["BaseDateTime"])

    # fusion des 6 lots de vessel_info
    vessel_files = [
        "vessel_info_janv_fev_2023.csv",
        "vessel_info_mars_avril_2023.csv",
        "vessel_info_mai_juin_2023.csv",
        "vessel_info_juil_aout.csv",
        "vessel_info_sept_oct.csv",
        "vessel_info_nov_dec.csv",
    ]
    vessel_info = pd.concat(
        [pd.read_csv(interim_dir / f) for f in vessel_files],
        ignore_index=True
    ).drop_duplicates(subset="MMSI")

    ports = pd.read_csv(processed_dir / "ports_validated.csv")

    od_matrix = build_od_matrix(trajectories, ports)
    od_matrix = od_matrix.merge(vessel_info, on="MMSI", how="left")

    print("Nombre de trajets O-D valides :", len(od_matrix))
    od_matrix.to_csv(processed_dir / "od_matrix.csv", index=False)
    print("Trajets inter-ports :", len(od_matrix[od_matrix["origin_port"] != od_matrix["destination_port"]]))
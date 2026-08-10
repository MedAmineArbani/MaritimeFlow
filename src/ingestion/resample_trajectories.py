"""
resample_trajectories.py
Ré-échantillonne chaque trajectoire à un intervalle fixe (10 min)
par interpolation linéaire sur la position.
"""

import pandas as pd
import os

RESAMPLE_INTERVAL = "10min"
INTERIM_FOLDER = "../../data/interim/"


def resample_trajectory(group: pd.DataFrame) -> pd.DataFrame:
    group = group.set_index("BaseDateTime").sort_index()

    numeric_cols = ["LAT", "LON", "SOG", "COG"]
    resampled = group[numeric_cols].resample(RESAMPLE_INTERVAL).mean().interpolate(method="linear")

    resampled["MMSI"] = group["MMSI"].iloc[0]
    resampled["trajectory_id"] = group["trajectory_id"].iloc[0]

    return resampled.reset_index()


def resample_all(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for traj_id, group in df.groupby("trajectory_id"):
        if len(group) < 2:
            continue
        results.append(resample_trajectory(group))
    return pd.concat(results, ignore_index=True)


if __name__ == "__main__":
    input_path = os.path.join(INTERIM_FOLDER, "trajectories.csv")

    df = pd.read_csv(input_path, low_memory=False)
    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"])

    print("Points avant ré-échantillonnage :", len(df))

    resampled = resample_all(df)

    print("Points après ré-échantillonnage :", len(resampled))

    resampled.to_csv(os.path.join(INTERIM_FOLDER, "trajectories_resampled.csv"), index=False)

    # trajectories.csv (non ré-échantillonné) n'est plus utilisé après cette étape
    del df
    os.remove(input_path)
    print("trajectories.csv supprimé (remplacé par trajectories_resampled.csv)")
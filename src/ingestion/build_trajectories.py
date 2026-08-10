"""
build_trajectories.py
Regroupe les points AIS par navire (MMSI), trie chronologiquement,
et segmente en trajets distincts dès qu'un trou > 6h est détecté.
"""

import pandas as pd
import glob
import os

GAP_THRESHOLD_HOURS = 6


def load_all_cleaned(folder: str) -> pd.DataFrame:
    """Charge et empile tous les CSV nettoyés du dossier."""
    files = glob.glob(os.path.join(folder, "*_clean.csv"))
    dfs = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"])
    return df


def segment_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    """Trie par MMSI/temps, puis assigne un trajectory_id par trou > 6h."""
    df = df.sort_values(["MMSI", "BaseDateTime"]).copy()

    df["time_gap"] = df.groupby("MMSI")["BaseDateTime"].diff()

    df["new_trajectory"] = (
        df["time_gap"].isna() | (df["time_gap"] > pd.Timedelta(hours=GAP_THRESHOLD_HOURS))
    )

    df["trajectory_id"] = df.groupby("MMSI")["new_trajectory"].cumsum()
    df["trajectory_id"] = df["MMSI"].astype(str) + "_" + df["trajectory_id"].astype(str)

    return df.drop(columns=["time_gap", "new_trajectory"])


if __name__ == "__main__":
    folder = "../../data/interim/"

    df = load_all_cleaned(folder)
    df = segment_trajectories(df)

    print("Nombre de points :", len(df))
    print("Nombre de trajets distincts :", df["trajectory_id"].nunique())

    df.to_csv(os.path.join(folder, "trajectories.csv"), index=False)

    # les fichiers _clean.csv individuels ne sont plus utiles
    # une fois regroupés dans trajectories.csv -> on les supprime pour gagner de la place
    for f in glob.glob(os.path.join(folder, "*_clean.csv")):
        os.remove(f)

    print("Fichiers *_clean.csv individuels supprimés (déjà regroupés dans trajectories.csv)")
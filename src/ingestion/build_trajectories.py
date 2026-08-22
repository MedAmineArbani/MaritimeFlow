"""
build_trajectories.py
Regroupe les points AIS par navire (MMSI), trie chronologiquement,
et segmente en trajets distincts dès qu'un trou > 6h est détecté.
"""

import pandas as pd
import glob
import os

GAP_THRESHOLD_HOURS = 6


import gc

import gc

def load_all_cleaned(folder: str) -> pd.DataFrame:
    """Charge et empile tous les CSV nettoyés du dossier."""
    files = glob.glob(os.path.join(folder, "*_clean.csv"))
    
    # Types optimisés pour réduire drastiquement la consommation RAM
    dtypes = {
        "LAT": "float32",
        "LON": "float32",
        "SOG": "float32",
        "COG": "float32",
        "Heading": "float32",
        "Length": "float32",
        "Width": "float32",
        "Draft": "float32"
    }
    
    dfs = []
    for f in files:
        df_chunk = pd.read_csv(f, dtype=dtypes, low_memory=False)
        dfs.append(df_chunk)
        
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    
    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"], errors="coerce")
    df.dropna(subset=["BaseDateTime", "MMSI"], inplace=True)
    df["MMSI"] = df["MMSI"].astype("int32")
    return df


def segment_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    """Trie par MMSI/temps, puis assigne un trajectory_id par trou > 6h."""
    df.sort_values(["MMSI", "BaseDateTime"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    gc.collect()

    df["time_gap"] = df.groupby("MMSI")["BaseDateTime"].diff()

    df["new_trajectory"] = (
        df["time_gap"].isna() | (df["time_gap"] > pd.Timedelta(hours=GAP_THRESHOLD_HOURS))
    )

    df["trajectory_id"] = df.groupby("MMSI")["new_trajectory"].cumsum()
    df["trajectory_id"] = df["MMSI"].astype(str) + "_" + df["trajectory_id"].astype(str)

    df.drop(columns=["time_gap", "new_trajectory"], inplace=True)
    return df

def extract_vessel_info(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait, par MMSI, les caractéristiques fixes du navire (proxy DWT)."""
    return (
        df.groupby("MMSI")[["Length", "Width", "Draft"]]
        .median()  # médiane : robuste aux valeurs aberrantes ponctuelles
        .reset_index()
    )

if __name__ == "__main__":
    folder = "../../data/interim/"

    # Check if there are any files to process
    if not glob.glob(os.path.join(folder, "*_clean.csv")):
        if os.path.exists(os.path.join(folder, "trajectories.csv")):
            print("trajectories.csv est déjà généré, on passe cette étape.")
            import sys
            sys.exit(0)
        else:
            print("Erreur: Aucun fichier *_clean.csv trouvé et trajectories.csv n'existe pas.")
            import sys
            sys.exit(1)

    df = load_all_cleaned(folder)
    df = segment_trajectories(df)

    print("Nombre de points :", len(df))
    print("Nombre de trajets distincts :", df["trajectory_id"].nunique())

    df.to_csv(os.path.join(folder, "trajectories.csv"), index=False)

    # les fichiers _clean.csv individuels ne sont plus utiles
    # une fois regroupés dans trajectories.csv -> on les supprime pour gagner de la place
    vessel_info = extract_vessel_info(df)
    vessel_info.to_csv(os.path.join(folder, "vessel_info.csv"), index=False)
    print("Table vessel_info.csv sauvegardée :", len(vessel_info), "navires")
    for f in glob.glob(os.path.join(folder, "*_clean.csv")):
        os.remove(f)

    print("Fichiers *_clean.csv individuels supprimés (déjà regroupés dans trajectories.csv)")
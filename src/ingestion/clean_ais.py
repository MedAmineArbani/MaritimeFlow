"""
clean_ais.py
Nettoyage des données AIS brutes. Traite tous les CSV présents dans
data/raw/noaa_gulf/, puis SUPPRIME le fichier brut une fois nettoyé.
Tolère les lignes malformées et les fichiers corrompus sans arrêter le pipeline.
"""

import pandas as pd
import glob
import os
import time
import gc

RAW_FOLDER = "../../data/raw/noaa_gulf/"
OUTPUT_FOLDER = "../../data/interim/"


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Force les colonnes numériques AIS en types numériques.
    Certains CSV NOAA mélangent str/num à cause de lignes malformées."""
    df = df.copy()
    numeric_cols = ["VesselType", "SOG", "COG", "Heading", "LAT", "LON",
                    "MMSI", "Length", "Width", "Draft"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def filter_cargo_vessels(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["VesselType"].between(70, 79)]


def remove_sentinel_values(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df["COG"] == 360.0, "COG"] = pd.NA
    df.loc[df["SOG"] == 102.3, "SOG"] = pd.NA
    df.loc[df["Heading"] == 511, "Heading"] = pd.NA
    return df


def remove_implausible_speeds(df: pd.DataFrame, max_knots: float = 18.0) -> pd.DataFrame:
    return df[(df["SOG"].isna()) | (df["SOG"] <= max_knots)]


def clean_ais(df: pd.DataFrame) -> pd.DataFrame:
    df = coerce_numeric_columns(df)
    df = filter_cargo_vessels(df)
    df = remove_sentinel_values(df)
    df = remove_implausible_speeds(df)
    return df


def safe_remove(path):
    for attempt in range(5):
        try:
            os.remove(path)
            return
        except PermissionError:
            time.sleep(1)


if __name__ == "__main__":
    csv_files = glob.glob(os.path.join(RAW_FOLDER, "*.csv"))
    print(f"Fichiers trouvés : {len(csv_files)}")

    total_before = 0
    total_after = 0
    failed_files = []

    for path in csv_files:
        filename = os.path.basename(path).replace(".csv", "_clean.csv")
        output_path = os.path.join(OUTPUT_FOLDER, filename)

        if os.path.exists(output_path):
            print(f"{os.path.basename(path)} : déjà nettoyé, on saute -> suppression du brut")
            safe_remove(path)
            continue

        try:
            print(f"{os.path.basename(path)} : lecture ...", end=" ", flush=True)
            raw = pd.read_csv(path, on_bad_lines="warn", engine="c", low_memory=False)
        except Exception as e:
            print(f"ECHEC ({e}) -> fichier ignoré")
            failed_files.append(os.path.basename(path))
            continue  # ne supprime PAS le fichier, on pourra réessaye

        cleaned = clean_ais(raw)

        total_before += len(raw)
        total_after += len(cleaned)

        cleaned.to_csv(output_path, index=False)
        print(f"{len(raw)} -> {len(cleaned)}")

        del raw, cleaned
        gc.collect()
        safe_remove(path)

    print(f"\nTotal avant nettoyage : {total_before}")
    print(f"Total après nettoyage : {total_after}")
    if failed_files:
        print(f"Fichiers ignorés (corrompus) : {failed_files}")
    print("Fichiers bruts supprimés.")
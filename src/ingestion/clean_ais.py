"""
clean_ais.py
Nettoyage des données AIS brutes. Traite tous les CSV présents dans
data/raw/noaa_gulf/, puis SUPPRIME le fichier brut une fois nettoyé
(pour ne jamais accumuler de gros volumes de données brutes).
"""

import pandas as pd
import glob
import os

RAW_FOLDER = "../../data/raw/noaa_gulf/"
OUTPUT_FOLDER = "../../data/interim/"


def filter_cargo_vessels(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["VesselType"].between(70, 79)].copy()


def remove_sentinel_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[df["COG"] == 360.0, "COG"] = pd.NA
    df.loc[df["SOG"] == 102.3, "SOG"] = pd.NA
    df.loc[df["Heading"] == 511, "Heading"] = pd.NA
    return df


def remove_implausible_speeds(df: pd.DataFrame, max_knots: float = 18.0) -> pd.DataFrame:
    df = df.copy()
    return df[(df["SOG"].isna()) | (df["SOG"] <= max_knots)]


def clean_ais(df: pd.DataFrame) -> pd.DataFrame:
    df = filter_cargo_vessels(df)
    df = remove_sentinel_values(df)
    df = remove_implausible_speeds(df)
    return df


if __name__ == "__main__":
    csv_files = glob.glob(os.path.join(RAW_FOLDER, "*.csv"))
    print(f"Fichiers trouvés : {len(csv_files)}")

    total_before = 0
    total_after = 0

    for path in csv_files:
        raw = pd.read_csv(path, low_memory=False)
        cleaned = clean_ais(raw)

        total_before += len(raw)
        total_after += len(cleaned)

        filename = os.path.basename(path).replace(".csv", "_clean.csv")
        output_path = os.path.join(OUTPUT_FOLDER, filename)

        if os.path.exists(output_path):
            print(f"{os.path.basename(path)} : déjà nettoyé, on saute -> suppression du brut")
        else:
            cleaned.to_csv(output_path, index=False)
            print(f"{os.path.basename(path)} : {len(raw)} -> {len(cleaned)}")

        # libère explicitement la mémoire liée au fichier avant suppression
        del raw, cleaned

        import gc
        gc.collect()

        import time
        for attempt in range(5):
            try:
                os.remove(path)
                break
            except PermissionError:
                time.sleep(1)
        else:
            print(f"  ATTENTION : impossible de supprimer {path}, à faire manuellement")

    print(f"\nTotal avant nettoyage : {total_before}")
    print(f"Total après nettoyage : {total_after}")
    print("Fichiers bruts supprimés.")
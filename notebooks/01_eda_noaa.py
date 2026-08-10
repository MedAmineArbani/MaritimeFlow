import pandas as pd
import glob
import os

# Trouve automatiquement le fichier CSV dans le dossier
folder = "../data/raw/noaa_gulf/"
csv_files = glob.glob(os.path.join(folder, "*.csv"))
print("Fichier trouvé :", csv_files)

path = csv_files[0]

# Charge le fichier
df = pd.read_csv(path)

# 1. Dimensions
print("Nombre de lignes :", len(df))
print("Nombre de colonnes :", df.shape[1])

# 2. Colonnes disponibles
print("\nColonnes :")
print(df.columns.tolist())

# 3. Aperçu des 5 premières lignes
print("\nAperçu :")
print(df.head())

# 4. Nombre de bateaux uniques (MMSI)
print("\nNombre de MMSI uniques :", df['MMSI'].nunique())

# 5. Types de navires présents
print("\nRépartition VesselType :")
print(df['VesselType'].value_counts().head(10))

# 6. Repérer les valeurs sentinelles (données manquantes déguisées)
print("\nValeurs COG = 360.0 :", (df['COG'] == 360.0).sum())
print("Valeurs SOG = 102.3 :", (df['SOG'] == 102.3).sum())
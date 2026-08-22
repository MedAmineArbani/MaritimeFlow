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
    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


if __name__ == "__main__":
    input_path = os.path.join(INTERIM_FOLDER, "trajectories.csv")
    output_path = os.path.join(INTERIM_FOLDER, "trajectories_resampled.csv")

    print("Début du ré-échantillonnage par lots (chunks)...")
    
    chunk_size = 2_000_000
    reader = pd.read_csv(input_path, chunksize=chunk_size, low_memory=False)
    
    first_chunk = True
    carry_over = pd.DataFrame()
    total_in = 0
    total_out = 0

    for i, chunk in enumerate(reader):
        chunk["BaseDateTime"] = pd.to_datetime(chunk["BaseDateTime"])
        total_in += len(chunk)
        print(f"  Traitement chunk {i+1}...")
        
        # Concaténer avec les données reportées du chunk précédent
        if not carry_over.empty:
            chunk = pd.concat([carry_over, chunk], ignore_index=True)
            
        # Reporter le dernier trajectory_id au prochain chunk (pour éviter de le couper)
        last_traj = chunk["trajectory_id"].iloc[-1]
        
        to_process = chunk[chunk["trajectory_id"] != last_traj]
        carry_over = chunk[chunk["trajectory_id"] == last_traj]
        
        if not to_process.empty:
            resampled = resample_all(to_process)
            if not resampled.empty:
                total_out += len(resampled)
                mode = 'w' if first_chunk else 'a'
                header = first_chunk
                resampled.to_csv(output_path, mode=mode, header=header, index=False)
                first_chunk = False

    # Traiter le tout dernier carry_over
    if not carry_over.empty:
        resampled = resample_all(carry_over)
        if not resampled.empty:
            total_out += len(resampled)
            mode = 'w' if first_chunk else 'a'
            header = first_chunk
            resampled.to_csv(output_path, mode=mode, header=header, index=False)
            
    print(f"Points avant ré-échantillonnage : {total_in}")
    print(f"Points après ré-échantillonnage : {total_out}")

    os.remove(input_path)
    print("trajectories.csv supprimé (remplacé par trajectories_resampled.csv)")
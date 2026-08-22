import pandas as pd

FILES_TO_MERGE = [
    "../../data/interim/stay_points_janv_fev_2023.csv",
    "../../data/interim/stay_points_mars_avril_2023.csv",
    "../../data/interim/stay_points_mai_juin_2023.csv",
    "../../data/interim/stay_points_juil_aout.csv",
    "../../data/interim/stay_points_sept_oct.csv",
    "../../data/interim/stay_points_nov_dec.csv",
]

if __name__ == "__main__":
    dfs = [pd.read_csv(f) for f in FILES_TO_MERGE]
    merged = pd.concat(dfs, ignore_index=True)
    print("Points d'arrêt combinés :", len(merged))
    merged.to_csv("../../data/interim/stay_points.csv", index=False)
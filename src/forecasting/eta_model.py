"""
eta_model.py
Modèle de prévision du temps de trajet (ETA) via LightGBM.
Features : distance, classe de navire (Length/Width/Draft), route encodée,
           saisonnalité (mois, jour de semaine).
Filtrage : vitesse implicite [1-40 km/h], durée <= 500h, distance > 5 km.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
# pyrefly: ignore [missing-import]
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"])

    # NE GARDER QUE LES VRAIS TRAJETS INTER-PORTS
    df = df[df["origin_port"] != df["destination_port"]]

    # --- FILTRAGE DES OUTLIERS (Priorité 1 du diagnostic) ---
    # Vitesse implicite : exclure attente massive (< 1 km/h) et erreurs GPS (> 40 km/h)
    df["implied_speed_kmh"] = df["distance_km"] / df["duration_hours"]
    df = df[(df["implied_speed_kmh"] >= 1) & (df["implied_speed_kmh"] <= 40)]

    # Durée max réaliste (P95 ~ 500h)
    df = df[df["duration_hours"] <= 500]

    # Distance min (< 5 km = erreur de segmentation)
    df = df[df["distance_km"] > 5]

    # Supprimer les navires sans dimensions connues
    df = df.dropna(subset=["distance_km", "Length", "Draft", "duration_hours"])

    # --- FEATURES ENRICHIES (Priorité 2 du diagnostic) ---
    df["month"] = df["start_time"].dt.month
    df["day_of_week"] = df["start_time"].dt.dayofweek  # 0=lundi, 6=dimanche

    # Route encodée (paire O-D) -> capture les spécificités de chaque corridor
    df["route"] = df["origin_port"] + " -> " + df["destination_port"]
    le = LabelEncoder()
    df["route_encoded"] = le.fit_transform(df["route"])

    print(f"  Filtrage applique :")
    print(f"    Vitesse implicite [1-40 km/h]")
    print(f"    Duree <= 500h")
    print(f"    Distance > 5 km")
    print(f"    NaN supprimes (Length, Draft)")

    return df, le


def train_eta_model(df: pd.DataFrame):
    features = ["distance_km", "Length", "Width", "Draft", "month", "day_of_week", "route_encoded"]
    target = "duration_hours"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = lgb.LGBMRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        min_child_samples=10,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )

    y_train_log = np.log1p(y_train)  # compresse la cible avant entraînement
    model.fit(X_train, y_train_log)

    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)  # décompresse la prédiction
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- Resultats ---")
    print(f"Exemples entrainement : {len(X_train)}")
    print(f"Exemples test         : {len(X_test)}")
    print(f"MAE (erreur moyenne)  : {mae:.2f} heures")
    print(f"R2                    : {r2:.3f}")

    # Erreur relative médiane
    median_dur = y_test.median()
    print(f"MAE / mediane(duree)  : {mae/median_dur*100:.1f}%")

    # Importance des features
    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print(f"\nImportance des features :")
    print(importance.to_string(index=False))

    # Cross-validation 5-fold pour robustesse
    print(f"\n--- Cross-validation 5-fold ---")
    from sklearn.compose import TransformedTargetRegressor
    model_cv = TransformedTargetRegressor(
        regressor=lgb.LGBMRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.05,
            min_child_samples=10, num_leaves=31, random_state=42, verbose=-1
        ),
        func=np.log1p, inverse_func=np.expm1
    )
    cv_mae = cross_val_score(model_cv, X, y, cv=5, scoring="neg_mean_absolute_error")
    cv_r2 = cross_val_score(model_cv, X, y, cv=5, scoring="r2")
    print(f"MAE 5-fold CV : {-cv_mae.mean():.2f} +/- {cv_mae.std():.2f} heures")
    print(f"R2  5-fold CV : {cv_r2.mean():.3f} +/- {cv_r2.std():.3f}")

    return model


if __name__ == "__main__":
    od_matrix = pd.read_csv(PROCESSED_DIR / "od_matrix.csv")

    print("=" * 60)
    print("  ENTRAINEMENT DU MODELE ETA (v2 - post-diagnostic)")
    print("=" * 60)

    df, label_encoder = prepare_features(od_matrix)
    print(f"\nTrajets exploitables apres filtrage : {len(df)}")
    print(f"  Duree  : median={df['duration_hours'].median():.1f}h  mean={df['duration_hours'].mean():.1f}h  std={df['duration_hours'].std():.1f}h")
    print(f"  Distance: median={df['distance_km'].median():.1f}km  mean={df['distance_km'].mean():.1f}km")

    model = train_eta_model(df)

    import joblib
    joblib.dump(model, PROCESSED_DIR / "eta_model.pkl")
    joblib.dump(label_encoder, PROCESSED_DIR / "route_encoder.pkl")
    print(f"\nModele sauvegarde : eta_model.pkl")
    print(f"Encodeur routes   : route_encoder.pkl")
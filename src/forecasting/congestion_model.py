"""
congestion_model.py
Modèle de congestion portuaire — deux approches complémentaires :
  1. Approche empirique : mesure directe du temps d'attente au mouillage
     à partir des données AIS (stay_points + od_matrix).
  2. Approche théorique : modèle de file d'attente M/M/c pour estimer
     le temps d'attente moyen en fonction du trafic.

Les résultats alimentent les contraintes de fenêtres de temps du MILP.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from math import factorial

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# =====================================================================
# 1. APPROCHE EMPIRIQUE : temps d'attente observé par port
# =====================================================================

def compute_empirical_congestion(od_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque port de destination, mesure :
    - Le nombre de navires qui arrivent par mois (proxy du trafic).
    - La vitesse implicite de chaque trajet (distance / durée).
    - Les trajets "lents" (vitesse < 5 km/h) indiquent probablement
      de l'attente au mouillage (le navire a passé du temps immobile).

    On estime le temps d'attente comme :
      wait_time = duration_hours - (distance_km / vitesse_cruise_mediane)
    où vitesse_cruise_mediane est la vitesse typique des trajets "normaux"
    sur la même route.
    """
    df = od_matrix.copy()
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])
    df["month"] = df["start_time"].dt.month

    # Vitesse implicite de chaque trajet
    df["implied_speed"] = df["distance_km"] / df["duration_hours"]

    # Pour chaque port de destination, calculer les stats de congestion
    results = []
    for port, group in df.groupby("destination_port"):
        n_arrivals = len(group)
        if n_arrivals < 5:
            continue

        # Vitesse de croisière "normale" = médiane des trajets rapides (> 5 km/h)
        fast_trips = group[group["implied_speed"] > 5]
        if len(fast_trips) < 3:
            cruise_speed = group["implied_speed"].median()
        else:
            cruise_speed = fast_trips["implied_speed"].median()

        if cruise_speed <= 0:
            continue

        # Temps de navigation pur estimé
        group = group.copy()
        group["nav_time_est"] = group["distance_km"] / cruise_speed

        # Temps d'attente estimé = durée totale - temps de navigation pur
        # (si négatif, on met 0 : pas d'attente détectée)
        group["wait_time_est"] = (group["duration_hours"] - group["nav_time_est"]).clip(lower=0)

        # Arrivées par mois (taux d'arrivée)
        arrivals_per_month = group.groupby("month").size()

        results.append({
            "port": port,
            "total_arrivals": n_arrivals,
            "avg_arrivals_per_month": arrivals_per_month.mean(),
            "cruise_speed_kmh": round(cruise_speed, 2),
            "wait_time_mean_hours": round(group["wait_time_est"].mean(), 2),
            "wait_time_median_hours": round(group["wait_time_est"].median(), 2),
            "wait_time_p75_hours": round(group["wait_time_est"].quantile(0.75), 2),
            "wait_time_p90_hours": round(group["wait_time_est"].quantile(0.90), 2),
            "pct_trips_with_wait": round(100 * (group["wait_time_est"] > 6).mean(), 1),
        })

    return pd.DataFrame(results).sort_values("wait_time_mean_hours", ascending=False)


# =====================================================================
# 2. APPROCHE THEORIQUE : file d'attente M/M/c
# =====================================================================

def mmc_wait_time(lam: float, mu: float, c: int) -> float:
    """
    Calcule le temps d'attente moyen Wq dans un système M/M/c.

    Paramètres (en mots simples) :
    - lam (lambda) : combien de navires arrivent par jour en moyenne
    - mu : combien de navires un poste à quai peut servir par jour
           (= 1 / durée moyenne de chargement en jours)
    - c : combien de postes à quai le port possède

    Retourne :
    - Wq : temps d'attente moyen en jours avant d'accéder à un poste
    """
    rho = lam / (c * mu)  # taux d'utilisation du port

    if rho >= 1.0:
        # Le port est saturé : le temps d'attente tend vers l'infini
        return float("inf")

    # Probabilité que le système soit vide (P0)
    sum_terms = sum((c * rho) ** n / factorial(n) for n in range(c))
    last_term = (c * rho) ** c / (factorial(c) * (1 - rho))
    p0 = 1.0 / (sum_terms + last_term)

    # Nombre moyen de navires en attente (Lq)
    lq = p0 * ((c * rho) ** c * rho) / (factorial(c) * (1 - rho) ** 2)

    # Temps d'attente moyen (Wq = Lq / lambda)
    wq = lq / lam
    return wq


def compute_theoretical_congestion(empirical_df: pd.DataFrame,
                                   avg_service_hours: float = 48.0,
                                   default_berths: int = 3) -> pd.DataFrame:
    """
    Pour chaque port, applique le modèle M/M/c avec :
    - lambda = taux d'arrivée observé (navires/jour)
    - mu = 1 / temps de service moyen (en jours)
    - c = nombre de postes à quai (par défaut 3)

    Teste aussi des scénarios : "Et si le trafic augmentait de +20% ou +50% ?"
    """
    mu = 24.0 / avg_service_hours  # Navires servis par jour par poste

    results = []
    for _, row in empirical_df.iterrows():
        lam = row["avg_arrivals_per_month"] / 30.0  # Arrivées par jour

        if lam <= 0:
            continue

        # Scénario actuel
        wq_current = mmc_wait_time(lam, mu, default_berths)

        # Scénario +20% de trafic
        wq_plus20 = mmc_wait_time(lam * 1.2, mu, default_berths)

        # Scénario +50% de trafic
        wq_plus50 = mmc_wait_time(lam * 1.5, mu, default_berths)

        results.append({
            "port": row["port"],
            "arrivals_per_day": round(lam, 3),
            "berths": default_berths,
            "utilization_pct": round(100 * lam / (default_berths * mu), 1),
            "mmc_wait_current_hours": round(wq_current * 24, 2),
            "mmc_wait_plus20pct_hours": round(wq_plus20 * 24, 2),
            "mmc_wait_plus50pct_hours": round(wq_plus50 * 24, 2),
        })

    return pd.DataFrame(results).sort_values("mmc_wait_current_hours", ascending=False)


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  ANALYSE DE LA CONGESTION PORTUAIRE")
    print("=" * 60)

    od_matrix = pd.read_csv(PROCESSED_DIR / "od_matrix.csv")

    # --- 1. Approche empirique ---
    print("\n--- 1. CONGESTION EMPIRIQUE (donnees AIS) ---\n")
    empirical = compute_empirical_congestion(od_matrix)
    print(f"Ports analyses : {len(empirical)}")
    print(f"\nTop 15 ports les plus congestionnés :")
    top_cols = ["port", "total_arrivals", "wait_time_mean_hours",
                "wait_time_median_hours", "wait_time_p90_hours", "pct_trips_with_wait"]
    print(empirical[top_cols].head(15).to_string(index=False))

    # Sauvegarder
    empirical.to_csv(PROCESSED_DIR / "congestion_empirical.csv", index=False)
    print(f"\nSauvegarde : congestion_empirical.csv")

    # --- 2. Approche théorique M/M/c ---
    print(f"\n--- 2. CONGESTION THEORIQUE (modele M/M/c) ---\n")
    print(f"Hypotheses : temps de service = 48h, postes a quai = 3")
    theoretical = compute_theoretical_congestion(empirical)
    print(f"\nTop 15 ports (temps d'attente theorique) :")
    print(theoretical.head(15).to_string(index=False))

    # Sauvegarder
    theoretical.to_csv(PROCESSED_DIR / "congestion_theoretical.csv", index=False)
    print(f"\nSauvegarde : congestion_theoretical.csv")

    # --- 3. Comparaison empirique vs théorique ---
    print(f"\n--- 3. COMPARAISON EMPIRIQUE vs THEORIQUE ---\n")
    comparison = empirical[["port", "wait_time_mean_hours"]].merge(
        theoretical[["port", "mmc_wait_current_hours"]], on="port", how="inner"
    )
    comparison.columns = ["port", "attente_reelle_h", "attente_mmc_h"]
    comparison = comparison.sort_values("attente_reelle_h", ascending=False)
    print(comparison.head(15).to_string(index=False))

    print(f"\n{'=' * 60}")
    print(f"  FIN DE L'ANALYSE")
    print(f"{'=' * 60}")

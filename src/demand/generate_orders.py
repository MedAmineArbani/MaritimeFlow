"""
generate_orders.py
Étape 5 : Génération de la demande d'exportation et de la flotte.
Crée des instances synthétiques mais réalistes pour alimenter le modèle MILP.
- Flotte : navires déduits de vessel_info.csv avec capacité et coûts extrapolés.
- Commandes : expéditions de phosphate générées aléatoirement sur le réseau O-D.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

# Hypothèses de marché (inspirées de la littérature et des cours du BDI)
# DWT (Deadweight Tonnage) est la capacité d'emport.
VESSEL_CLASSES = [
    {"name": "Handysize", "min_length": 0, "max_length": 180, "avg_dwt": 25000, "daily_cost_usd": 12000, "speed_knots": 13},
    {"name": "Handymax", "min_length": 180, "max_length": 200, "avg_dwt": 50000, "daily_cost_usd": 15000, "speed_knots": 14},
    {"name": "Panamax", "min_length": 200, "max_length": 230, "avg_dwt": 75000, "daily_cost_usd": 18000, "speed_knots": 14},
    {"name": "Capesize", "min_length": 230, "max_length": 999, "avg_dwt": 150000, "daily_cost_usd": 25000, "speed_knots": 15},
]

def assign_vessel_class(length: float) -> dict:
    """Associe un navire à une classe commerciale selon sa longueur."""
    if pd.isna(length):
        return VESSEL_CLASSES[1]  # Par défaut Handymax
    for c in VESSEL_CLASSES:
        if c["min_length"] <= length < c["max_length"]:
            return c
    return VESSEL_CLASSES[1]

def generate_fleet(vessel_info: pd.DataFrame, n_vessels: int = 50) -> pd.DataFrame:
    """Sélectionne un échantillon de navires et extrapole leurs caractéristiques commerciales."""
    # Filtrer les navires valides
    valid_vessels = vessel_info.dropna(subset=["Length"]).copy()
    
    # Prendre un échantillon
    if len(valid_vessels) > n_vessels:
        fleet = valid_vessels.sample(n=n_vessels, random_state=42).copy()
    else:
        fleet = valid_vessels.copy()
        
    fleet["vessel_class"] = fleet["Length"].apply(lambda l: assign_vessel_class(l)["name"])
    fleet["capacity_tonnes"] = fleet["Length"].apply(lambda l: assign_vessel_class(l)["avg_dwt"])
    fleet["daily_charter_cost"] = fleet["Length"].apply(lambda l: assign_vessel_class(l)["daily_cost_usd"])
    fleet["design_speed_knots"] = fleet["Length"].apply(lambda l: assign_vessel_class(l)["speed_knots"])
    
    # Pour simplifier le MILP, on suppose que les navires sont disponibles au début de l'horizon
    # et situés dans un "pool" de départ (idéalement le port de chargement principal)
    fleet["available_from"] = pd.Timestamp("2024-01-01")
    
    return fleet

def generate_orders(od_matrix: pd.DataFrame, n_orders: int = 30) -> pd.DataFrame:
    """Génère un carnet de commandes synthétique basé sur le réseau observé."""
    # Trouver les paires origine-destination fréquentes
    routes = od_matrix.groupby(["origin_port", "destination_port"]).size().reset_index(name="count")
    routes = routes[routes["origin_port"] != routes["destination_port"]]
    
    # Échantillonner des routes aléatoirement, pondérées par leur fréquence
    # pour simuler des flux commerciaux réalistes
    weights = routes["count"] / routes["count"].sum()
    selected_routes = routes.sample(n=n_orders, weights=weights, replace=True, random_state=42)
    
    orders = []
    base_date = pd.Timestamp("2024-01-05")
    
    for i, (_, row) in enumerate(selected_routes.iterrows()):
        # Volume de la commande (ex: 20k à 60k tonnes)
        volume = int(np.random.uniform(20000, 60000))
        
        # Fenêtre de livraison : date au plus tôt (er) et au plus tard (lr)
        # On étale les commandes sur un horizon d'un mois
        days_offset = np.random.randint(0, 30)
        early_time = base_date + pd.Timedelta(days=days_offset)
        # Fenêtre de 10 jours pour livrer
        late_time = early_time + pd.Timedelta(days=10)
        
        orders.append({
            "order_id": f"ORD_{i+1:03d}",
            "origin_port": row["origin_port"],
            "destination_port": row["destination_port"],
            "volume_tonnes": volume,
            "early_time": early_time,
            "late_time": late_time,
            "penalty_per_day": 5000  # Pénalité de retard (USD/jour)
        })
        
    return pd.DataFrame(orders)

if __name__ == "__main__":
    print("=" * 60)
    print("  GÉNÉRATION DE LA FLOTTE ET DES COMMANDES")
    print("=" * 60)
    
    # Charger les données existantes
    od_matrix = pd.read_csv(PROCESSED_DIR / "od_matrix.csv")
    vessel_info = pd.concat([pd.read_csv(f) for f in INTERIM_DIR.glob("vessel_info_*.csv")], ignore_index=True).drop_duplicates(subset="MMSI")
    
    # 1. Générer la flotte (on va simuler 20 navires pour un problème de petite taille pour le MILP)
    fleet = generate_fleet(vessel_info, n_vessels=20)
    print(f"\nFlotte générée : {len(fleet)} navires")
    print("Répartition par classe :")
    print(fleet["vessel_class"].value_counts().to_string())
    
    fleet.to_csv(PROCESSED_DIR / "fleet_instances.csv", index=False)
    print(f"-> Sauvegardé dans data/processed/fleet_instances.csv")
    
    # 2. Générer les commandes (ex: 30 commandes à livrer sur un mois)
    orders = generate_orders(od_matrix, n_orders=30)
    print(f"\nCommandes générées : {len(orders)}")
    print(f"Volume total à transporter : {orders['volume_tonnes'].sum():,} tonnes")
    
    orders.to_csv(PROCESSED_DIR / "export_orders.csv", index=False)
    print(f"-> Sauvegardé dans data/processed/export_orders.csv")
    
    print("\n" + "=" * 60)
    print("  GÉNÉRATION TERMINÉE")
    print("=" * 60)

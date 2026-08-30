"""
fleet_simulation.py
Simulateur à événements discrets avec SimPy.
Teste la robustesse du plan optimisé (ALNS) face aux aléas de navigation
et de congestion portuaire, et le compare à une politique "Naïve".
"""

import simpy
import random
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Constantes de temps
SIM_START_DATE = pd.Timestamp("2024-01-01")

class MaritimeEnvironment:
    def __init__(self, env, ports_list):
        self.env = env
        # Chaque port a 3 postes à quai (berths)
        self.ports = {p: simpy.Resource(env, capacity=3) for p in ports_list}
        self.late_deliveries = 0
        self.total_deliveries = 0
        self.total_demurrage_cost = 0

def simulate_voyage(env, ship_name, order, maritime_env, policy_name):
    """Processus SimPy décrivant le voyage d'un navire pour livrer une commande."""
    
    # 1. Trajet vers le port d'origine (Ballast leg) + Chargement
    # Aléa météo : le trajet prend entre 2 et 5 jours
    travel_to_origin = random.uniform(2, 5)
    yield env.timeout(travel_to_origin)
    
    # Temps de chargement fixe de 2 jours
    yield env.timeout(2)
    
    # 2. Trajet vers la destination (Laden leg)
    # Aléa météo lourd : durée moyenne 10 jours, écart-type 3 jours (loi Normale)
    base_travel_time = max(5, random.gauss(10, 3))
    yield env.timeout(base_travel_time)
    
    # 3. Arrivée au port de destination et Congestion
    dest_port = order['destination_port']
    arrival_time = env.now
    
    # Demande d'accès à un poste à quai (file d'attente M/M/c)
    port_resource = maritime_env.ports[dest_port]
    with port_resource.request() as req:
        yield req  # Le navire attend que le poste se libère
        
        wait_time = env.now - arrival_time
        
        # Déchargement (temps de service exponentiel, moyenne 2 jours)
        service_time = random.expovariate(1.0 / 2.0)
        yield env.timeout(service_time)
        
    # 4. Évaluation du délai
    delivery_date = SIM_START_DATE + pd.Timedelta(days=env.now)
    late_time = pd.to_datetime(order['late_time'])
    
    maritime_env.total_deliveries += 1
    
    if delivery_date > late_time:
        maritime_env.late_deliveries += 1
        days_late = (delivery_date - late_time).days
        # Pénalité : 5000 USD par jour de retard
        penalty = days_late * 5000
        maritime_env.total_demurrage_cost += penalty
        
def run_simulation(policy="optimized"):
    """Prépare et lance la simulation pour une politique donnée."""
    env = simpy.Environment()
    
    orders_df = pd.read_csv(PROCESSED_DIR / "export_orders.csv")
    fleet_df = pd.read_csv(PROCESSED_DIR / "fleet_instances.csv")
    
    # Liste unique de tous les ports
    all_ports = set(orders_df['destination_port'].unique()).union(set(orders_df['origin_port'].unique()))
    
    maritime_env = MaritimeEnvironment(env, all_ports)
    
    if policy == "optimized":
        # Politique Optimisée : On lit le plan d'affectation ALNS
        try:
            plan_df = pd.read_csv(PROCESSED_DIR / "alns_solution.csv")
        except FileNotFoundError:
            print("Erreur: Le plan ALNS n'existe pas. Lancez alns_solver.py d'abord.")
            return None
            
        # Pour chaque navire, on lance les voyages prévus l'un après l'autre
        for ship_name, group in plan_df.groupby("Navire"):
            def ship_process(env, ship_name, assignments, maritime_env):
                for _, assignment in assignments.iterrows():
                    # Trouver les détails de la commande
                    order = orders_df[orders_df['order_id'] == assignment['Commande']].iloc[0]
                    yield env.process(simulate_voyage(env, ship_name, order, maritime_env, policy))
                    
            env.process(ship_process(env, ship_name, group, maritime_env))
            
    elif policy == "naive":
        # Politique Naïve : Premier arrivé, premier servi
        # On affecte aléatoirement les commandes aux navires capables
        orders_list = orders_df.to_dict('records')
        random.seed(42)  # Pour la reproductibilité
        random.shuffle(orders_list)
        
        # On regroupe bêtement les commandes par navire (sans optimisation mathématique)
        naive_assignments = {row['vessel_class']: [] for _, row in fleet_df.iterrows()}
        vessels = fleet_df.to_dict('records')
        
        for order in orders_list:
            # Trouver un navire aléatoire qui a la capacité
            capable_vessels = [v for v in vessels if v['capacity_tonnes'] >= order['volume_tonnes']]
            if capable_vessels:
                chosen_vessel = random.choice(capable_vessels)
                naive_assignments[chosen_vessel['vessel_class']].append(order)
                
        for ship_name, assigned_orders in naive_assignments.items():
            def ship_process(env, ship_name, assigned_orders, maritime_env):
                for order in assigned_orders:
                    yield env.process(simulate_voyage(env, ship_name, order, maritime_env, policy))
            
            if assigned_orders:
                env.process(ship_process(env, ship_name, assigned_orders, maritime_env))

    # Lancer la simulation pendant 200 jours max pour être sûr que tout arrive
    env.run(until=200)
    
    # Calcul des KPIs
    otd_rate = ((maritime_env.total_deliveries - maritime_env.late_deliveries) / max(1, maritime_env.total_deliveries)) * 100
    
    return {
        "Politique": policy.capitalize(),
        "Commandes livrées": maritime_env.total_deliveries,
        "Livraisons en retard": maritime_env.late_deliveries,
        "Taux de Service (OTD)": f"{otd_rate:.1f}%",
        "Coût des pénalités (USD)": f"{maritime_env.total_demurrage_cost:,.0f}"
    }

if __name__ == "__main__":
    print("=" * 60)
    print("  SIMULATION DE LA ROBUSTESSE DES PLANS (SIMPY)")
    print("=" * 60)
    
    print("\nSimulation du Plan Optimisé (ALNS) avec aléas...")
    res_opt = run_simulation(policy="optimized")
    
    print("\nSimulation du Plan Naïf (Aléatoire/FCFS) avec aléas...")
    res_naive = run_simulation(policy="naive")
    
    print("\n" + "=" * 60)
    print("  RÉSULTATS DU BENCHMARK DE ROBUSTESSE")
    print("=" * 60)
    
    results = pd.DataFrame([res_opt, res_naive])
    print(results.to_string(index=False))

"""
alns_solver.py
Métaheuristique ALNS (Adaptive Large Neighborhood Search) pour le routage de flotte.
Implémentation didactique :
- Solution initiale gloutonne
- Destruction aléatoire (Random Removal)
- Réparation gloutonne (Greedy Insertion)
- Recuit simulé pour l'acceptation
"""

import time
import random
import math
import copy
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

class Order:
    def __init__(self, id, volume):
        self.id = id
        self.volume = volume

class Vessel:
    def __init__(self, id, name, capacity, daily_cost):
        self.id = id
        self.name = name
        self.capacity = capacity
        # Coût estimé du trajet complet pour simplifier (10 jours)
        self.cost = daily_cost * 10
        self.assigned_orders = []
        
    @property
    def remaining_capacity(self):
        return self.capacity - sum(o.volume for o in self.assigned_orders)

def compute_cost(vessels):
    """Calcule le coût total d'une solution (somme des coûts des navires utilisés)."""
    return sum(v.cost for v in vessels if len(v.assigned_orders) > 0)

def initial_solution(orders, vessels):
    """Heuristique gloutonne : assigne les plus grosses commandes aux navires les moins chers capables."""
    orders_sorted = sorted(orders, key=lambda o: o.volume, reverse=True)
    vessels_sorted = sorted(vessels, key=lambda v: v.cost)
    
    unassigned = []
    
    for order in orders_sorted:
        assigned = False
        for vessel in vessels_sorted:
            if vessel.remaining_capacity >= order.volume:
                vessel.assigned_orders.append(order)
                assigned = True
                break
        if not assigned:
            unassigned.append(order)
            
    return unassigned

def random_removal(vessels, degree=0.2):
    """Retire (désassigne) aléatoirement un pourcentage des commandes."""
    used_vessels = [v for v in vessels if len(v.assigned_orders) > 0]
    # Nombre total de commandes assignées
    total_assigned = sum(len(v.assigned_orders) for v in used_vessels)
    n_remove = max(1, int(total_assigned * degree))
    
    removed_orders = []
    
    # On retire aléatoirement n_remove commandes des navires
    for _ in range(n_remove):
        valid_vessels = [v for v in vessels if len(v.assigned_orders) > 0]
        if not valid_vessels:
            break
        v = random.choice(valid_vessels)
        o = random.choice(v.assigned_orders)
        v.assigned_orders.remove(o)
        removed_orders.append(o)
        
    return removed_orders

def greedy_insertion(removed_orders, vessels):
    """Réinsère les commandes en minimisant l'augmentation du coût (le moins cher d'abord)."""
    unassigned = []
    # On trie les commandes restantes par taille pour placer les plus dures d'abord
    removed_orders.sort(key=lambda o: o.volume, reverse=True)
    
    for order in removed_orders:
        best_vessel = None
        best_cost_increase = float('inf')
        
        for vessel in vessels:
            if vessel.remaining_capacity >= order.volume:
                # Augmentation du coût = 0 si navire déjà utilisé, sinon son coût fixe
                cost_increase = 0 if len(vessel.assigned_orders) > 0 else vessel.cost
                if cost_increase < best_cost_increase:
                    best_cost_increase = cost_increase
                    best_vessel = vessel
                    
        if best_vessel is not None:
            best_vessel.assigned_orders.append(order)
        else:
            unassigned.append(order)
            
    return unassigned

def solve_alns(fleet_df, orders_df, iterations=1000):
    print("Initialisation de l'ALNS...")
    
    # 1. Préparation des objets
    orders = [Order(row["order_id"], row["volume_tonnes"]) for _, row in orders_df.iterrows()]
    vessels = [Vessel(i, row["vessel_class"], row["capacity_tonnes"], row["daily_charter_cost"]) 
               for i, row in fleet_df.iterrows()]
               
    # 2. Solution Initiale
    start_time = time.time()
    unassigned = initial_solution(orders, vessels)
    current_cost = compute_cost(vessels)
    best_cost = current_cost
    best_vessels = copy.deepcopy(vessels)
    
    print(f"Coût initial : {current_cost:,.0f} USD ({len(unassigned)} commandes non assignées)")
    
    # Paramètres du Recuit Simulé (Simulated Annealing)
    temperature = 50000.0
    cooling_rate = 0.99
    
    # 3. Boucle ALNS
    for i in range(iterations):
        # Créer une copie de travail
        temp_vessels = copy.deepcopy(vessels)
        
        # Destruction
        removed = random_removal(temp_vessels, degree=0.2)
        
        # On essaie aussi de réinsérer celles qui n'avaient pas de place
        removed.extend(unassigned)
        
        # Réparation
        new_unassigned = greedy_insertion(removed, temp_vessels)
        
        new_cost = compute_cost(temp_vessels)
        
        # Pénalité énorme si des commandes ne sont pas assignées (pour forcer à tout livrer)
        penalty = len(new_unassigned) * 10_000_000
        total_new_cost = new_cost + penalty
        total_current_cost = current_cost + (len(unassigned) * 10_000_000)
        
        # Critère d'acceptation (Simulated Annealing)
        if total_new_cost < total_current_cost:
            # Meilleure solution, on l'accepte toujours
            vessels = temp_vessels
            current_cost = new_cost
            unassigned = new_unassigned
            
            # Mise à jour du "Meilleur absolu"
            if total_new_cost < (best_cost + (len(unassigned)*10_000_000)):
                best_cost = new_cost
                best_vessels = copy.deepcopy(temp_vessels)
        else:
            # Solution pire, on l'accepte avec une probabilité qui dépend de la température
            delta = total_new_cost - total_current_cost
            prob = math.exp(-delta / temperature)
            if random.random() < prob:
                vessels = temp_vessels
                current_cost = new_cost
                unassigned = new_unassigned
                
        # Refroidissement
        temperature *= cooling_rate
        
    solve_time = time.time() - start_time
    
    # 4. Affichage des résultats
    print(f"\n--- RÉSULTATS ALNS ---")
    print(f"Coût final trouvé : {best_cost:,.0f} USD")
    print(f"Temps de calcul   : {solve_time:.2f} secondes")
    
    assignments = []
    for v in best_vessels:
        for o in v.assigned_orders:
            assignments.append({
                "Navire": v.name,
                "Capacité (t)": v.capacity,
                "Commande": o.id,
                "Volume (t)": o.volume,
                "Coût (USD)": v.cost
            })
            
    res_df = pd.DataFrame(assignments)
    if not res_df.empty:
        print("\nPlan d'affectation (Top 10) :")
        print(res_df.head(10).to_string(index=False))
        res_df.to_csv(PROCESSED_DIR / "alns_solution.csv", index=False)

if __name__ == "__main__":
    fleet = pd.read_csv(PROCESSED_DIR / "fleet_instances.csv")
    orders = pd.read_csv(PROCESSED_DIR / "export_orders.csv")
    
    # On lance l'ALNS sur la totalité des navires et commandes
    solve_alns(fleet, orders, iterations=5000)

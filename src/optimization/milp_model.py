"""
milp_model.py
Modèle d'optimisation MILP pour l'affectation de flotte (Fleet Deployment).
Assigne les commandes d'exportation aux navires disponibles en minimisant
les coûts d'affrètement, tout en respectant les capacités.
"""

import pandas as pd
import pyomo.environ as pyo
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def build_and_solve_milp(fleet_df: pd.DataFrame, orders_df: pd.DataFrame):
    """
    Construit et résout le modèle MILP d'affectation de flotte.
    """
    print("Construction du modèle MILP...")
    model = pyo.ConcreteModel()

    # --- ENSEMBLES ---
    model.V = pyo.Set(initialize=fleet_df.index.tolist(), doc="Navires disponibles")
    model.R = pyo.Set(initialize=orders_df.index.tolist(), doc="Commandes à livrer")

    # --- PARAMÈTRES ---
    # Capacité des navires (tonnes)
    cap = {v: fleet_df.loc[v, "capacity_tonnes"] for v in model.V}
    # Coût journalier d'affrètement (estimé sur 10 jours de trajet pour simplifier)
    cost = {v: fleet_df.loc[v, "daily_charter_cost"] * 10 for v in model.V}
    # Volume des commandes
    vol = {r: orders_df.loc[r, "volume_tonnes"] for r in model.R}

    # --- VARIABLES DE DÉCISION ---
    # x[v, r] = 1 si le navire v traite la commande r
    model.x = pyo.Var(model.V, model.R, domain=pyo.Binary)
    # y[v] = 1 si le navire v est utilisé
    model.y = pyo.Var(model.V, domain=pyo.Binary)

    # --- FONCTION OBJECTIF ---
    # Minimiser le coût total d'affrètement des navires utilisés
    def obj_rule(m):
        return sum(cost[v] * m.y[v] for v in m.V)
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # --- CONTRAINTES ---
    # 1. Satisfaction de la demande : chaque commande doit être affectée à exactement 1 navire
    def demand_rule(m, r):
        return sum(m.x[v, r] for v in m.V) == 1
    model.demand_constraint = pyo.Constraint(model.R, rule=demand_rule)

    # 2. Capacité du navire : le volume affecté ne doit pas dépasser la capacité
    # Et relier x[v, r] à y[v] (si un navire transporte qqch, y[v] doit être à 1)
    def capacity_rule(m, v):
        return sum(vol[r] * m.x[v, r] for r in m.R) <= cap[v] * m.y[v]
    model.capacity_constraint = pyo.Constraint(model.V, rule=capacity_rule)

    # --- RÉSOLUTION ---
    print("Résolution en cours (solveur GLPK/CBC/HiGHS)...")
    try:
        # On essaie d'utiliser glpk ou cbc par défaut, ou appsi_highs si dispo
        solver = pyo.SolverFactory('appsi_highs') 
        if not solver.available():
            solver = pyo.SolverFactory('glpk')
            
        result = solver.solve(model, tee=True)
        
        print("\n--- RÉSULTATS DE L'OPTIMISATION ---")
        if (result.solver.status == pyo.SolverStatus.ok) and (result.solver.termination_condition == pyo.TerminationCondition.optimal):
            print(f"Solution OPTIMALE trouvée !")
            print(f"Coût total : {pyo.value(model.obj):,.0f} USD")
            
            # Afficher l'affectation
            assignments = []
            for v in model.V:
                if pyo.value(model.y[v]) > 0.5:  # Si navire utilisé
                    for r in model.R:
                        if pyo.value(model.x[v, r]) > 0.5:
                            assignments.append({
                                "Navire": fleet_df.loc[v, "vessel_class"],
                                "Capacité (t)": cap[v],
                                "Commande": orders_df.loc[r, "order_id"],
                                "Volume (t)": vol[r],
                                "Coût (USD)": cost[v]
                            })
            
            res_df = pd.DataFrame(assignments)
            print("\nPlan d'affectation :")
            print(res_df.to_string(index=False))
            res_df.to_csv(PROCESSED_DIR / "milp_solution.csv", index=False)
            
        else:
            print("Aucune solution optimale trouvée. Vérifiez que la flotte est suffisante pour la demande.")
            
    except Exception as e:
        print(f"Erreur lors de la résolution (solveur introuvable ?) : {e}")

if __name__ == "__main__":
    fleet = pd.read_csv(PROCESSED_DIR / "fleet_instances.csv")
    orders = pd.read_csv(PROCESSED_DIR / "export_orders.csv")
    
    # Pour tester rapidement, on prend un sous-ensemble (10 navires, 15 commandes)
    build_and_solve_milp(fleet.head(10), orders.head(15))

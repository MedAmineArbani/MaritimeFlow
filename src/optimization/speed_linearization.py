"""
speed_linearization.py
Module pour linéariser le coût de carburant cubique (C = k * v^3)
nécessaire pour le solveur MILP (qui n'accepte que des équations linéaires).
"""

import numpy as np

def generate_speed_breakpoints(v_min: float, v_max: float, n_segments: int = 5) -> tuple:
    """
    Génère les points de cassure (breakpoints) pour approximer v^3
    par une fonction affine par morceaux.
    
    Args:
        v_min: Vitesse minimale (nœuds)
        v_max: Vitesse maximale (nœuds)
        n_segments: Nombre de segments linéaires
        
    Returns:
        speeds: liste des vitesses (x)
        consumptions: liste des consommations (y = v^3)
    """
    # On crée des points de vitesse équidistants
    speeds = np.linspace(v_min, v_max, n_segments + 1)
    
    # La consommation théorique est proportionnelle au cube de la vitesse
    consumptions = speeds ** 3
    
    return speeds.tolist(), consumptions.tolist()

if __name__ == "__main__":
    speeds, cons = generate_speed_breakpoints(10.0, 16.0, 4)
    print("Points de linéarisation (Vitesse -> Conso) :")
    for s, c in zip(speeds, cons):
        print(f"  Vitesse {s:.1f} noeuds -> Conso relative {c:.0f}")

# MaritimeFlow

**MaritimeFlow** est une plateforme complète d'analyse et d'optimisation du routage maritime, basée sur le traitement massif de données AIS (Automatic Identification System).

L'objectif du projet est de reconstruire le réseau maritime réel à partir des signaux GPS bruts des navires, d'identifier les ports et zones de mouillage, de calculer les temps de parcours réels (matrice Origine-Destination), et d'optimiser les routes et vitesses de la flotte de cargos (via des modèles mathématiques et métaheuristiques).

---

## Architecture du Projet

Le pipeline de données et de modélisation est divisé en plusieurs modules fonctionnels :

### 1. Ingestion & Nettoyage (`src/ingestion/`)
- Traitement des données brutes NOAA (Golfe du Mexique) et flux continu AISstream.io (Maroc).
- Nettoyage des données : suppression des valeurs aberrantes (vitesses impossibles, valeurs sentinelles).
- Segmentation chronologique par navire (MMSI) pour construire des **trajectoires** cohérentes.
- Ré-échantillonnage des trajectoires pour alléger les calculs géospatiaux.

### 2. Détection des Ports (`src/ports/`)
- **Détection des "stay points"** : Identification des segments où un navire reste immobile pendant plusieurs heures (candidats pour des ports ou mouillages).
- **Clustering spatial (DBSCAN)** : Regroupement géographique des stay points pour identifier les véritables zones portuaires.
- **Validation** : Croisement des clusters détectés avec la base de données globale (ex: World Port Index) pour nommer et valider les ports.

### 3. Construction du Réseau (`src/network/`)
- Calcul de la **matrice Origine-Destination (O-D)** : Extraction des temps de trajet réels, distances, et vitesses moyennes observées entre les ports validés à partir des trajectoires extraites.

### 4. Modélisation & Optimisation (`src/forecasting/` & `src/optimization/`)
- **Prévisions (Forecasting)** : Modèles de Machine Learning (LightGBM) pour estimer l'ETA (Expected Time of Arrival) et modélisation de la congestion portuaire.
- **Optimisation** : 
  - Modélisation MILP (Mixed Integer Linear Programming) via Pyomo/OR-Tools pour l'allocation optimale de la flotte.
  - Solveur ALNS (Adaptive Large Neighbourhood Search) pour le passage à l'échelle sur de grandes instances.
  - Modélisation des coûts de carburant (linéarisation de la fonction cubique de la vitesse).

### 5. Simulation & Visualisation (`src/simulation/` & `dashboard/`)
- **Simulation** : Validation des plans de routage optimisés via une simulation à événements discrets (SimPy).
- **Dashboard** : Interface utilisateur interactive développée avec Streamlit pour visualiser le réseau maritime, les clusters de ports et la matrice O-D.

---

## Installation & Configuration

1. **Cloner le projet** et se placer à la racine :
   ```bash
   cd maritimeflow
   ```

2. **Créer et activer l'environnement virtuel** :
   ```bash
   python -m venv .venv
   # Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

---

## Utilisation (Pipeline)

Pour exécuter le traitement de bout en bout des données AIS, lancez les scripts dans cet ordre :

```powershell
# 1. Nettoyage des fichiers bruts NOAA
python src/ingestion/clean_ais.py

# 2. Construction et segmentation des trajectoires
python src/ingestion/build_trajectories.py

# 3. Ré-échantillonnage (allègement) des trajectoires
python src/ingestion/resample_trajectories.py

# 4. Identification des points d'arrêt géolocalisés
python src/ports/detect_stay_points.py

# 5. Clustering géographique (détection des terminaux/mouillages)
python src/ports/cluster_ports_dbscan.py

# 6. Validation et assignation des noms de ports (World Port Index)
python src/ports/validate_ports.py

# 7. Génération de la matrice des temps de trajet Origine-Destination
python src/network/od_matrix.py
```

Vous pouvez également automatiser ces étapes si un script `run_pipeline.py` est configuré à la racine du projet.

---

## Arborescence des Données (`data/`)

- `data/raw/` : Fichiers CSV/JSON bruts (données NOAA, flux AISstream). Les fichiers ZIP de la NOAA doivent être extraits dans `data/raw/noaa_gulf/`.
- `data/interim/` : Fichiers intermédiaires (données nettoyées, trajectoires resamplées, stay points).
- `data/processed/` : Données finales prêtes pour l'analyse (ports validés, matrice O-D, candidats de clusters).

*(Les dossiers de données sont exclus de Git via le `.gitignore` en raison du volume de données.)*

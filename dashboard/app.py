"""
===============================================================================
MARITIMEFLOW DASHBOARD - PLATEFURME DE ROUTAGE ET D'OPTIMISATION MARITIME
===============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import pickle
import joblib
import folium
from streamlit_folium import st_folium
import simpy
import random
import time

# --- CONFIGURATION PAGE ---
st.set_page_config(
    page_title="MaritimeFlow | Decision Support System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- INJECTION CSS PERSONNALISÉ ---
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Header Container */
    .header-box {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
        margin-bottom: 25px;
    }
    
    .header-title {
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.2rem;
        margin-bottom: 5px;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(56, 189, 248, 0.2);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(56, 189, 248, 0.6);
        box-shadow: 0 10px 20px rgba(56, 189, 248, 0.15);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 5px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre;
        background-color: rgba(30, 41, 59, 0.5);
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 0 16px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #38bdf8 !important;
        color: #0f172a !important;
        font-weight: 700;
    }
    
    /* Custom divider */
    .custom-hr {
        border: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.3), transparent);
        margin: 25px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@st.cache_data
def load_data():
    ports_df = pd.read_csv(PROCESSED_DIR / "ports_validated.csv")
    od_df = pd.read_csv(PROCESSED_DIR / "od_matrix.csv")
    orders_df = pd.read_csv(PROCESSED_DIR / "export_orders.csv")
    fleet_df = pd.read_csv(PROCESSED_DIR / "fleet_instances.csv")

    try:
        alns_df = pd.read_csv(PROCESSED_DIR / "alns_solution.csv")
    except Exception:
        alns_df = None

    try:
        milp_df = pd.read_csv(PROCESSED_DIR / "milp_solution.csv")
    except Exception:
        milp_df = None

    try:
        cong_emp = pd.read_csv(PROCESSED_DIR / "congestion_empirical.csv")
        cong_theo = pd.read_csv(PROCESSED_DIR / "congestion_theoretical.csv")
    except Exception:
        cong_emp, cong_theo = None, None

    return (
        ports_df,
        od_df,
        orders_df,
        fleet_df,
        alns_df,
        milp_df,
        cong_emp,
        cong_theo,
    )


@st.cache_resource
def load_ml_models():
    eta_model_path = PROCESSED_DIR / "eta_model.pkl"
    encoder_path = PROCESSED_DIR / "route_encoder.pkl"
    eta_model = None
    encoder = None
    if eta_model_path.exists() and encoder_path.exists():
        try:
            eta_model = joblib.load(eta_model_path)
            encoder = joblib.load(encoder_path)
        except Exception:
            try:
                with open(eta_model_path, "rb") as f:
                    eta_model = pickle.load(f)
                with open(encoder_path, "rb") as f:
                    encoder = pickle.load(f)
            except Exception:
                eta_model = None
                encoder = None
    return eta_model, encoder


(
    ports_df,
    od_df,
    orders_df,
    fleet_df,
    alns_df,
    milp_df,
    cong_emp,
    cong_theo,
) = load_data()
eta_model, encoder = load_ml_models()

# --- HEADER APP ---
st.markdown(
    """
<div class="header-box">
    <div>
        <div class="header-title">MARITIMEFLOW : DECISION SUPPORT SYSTEM</div>
        <div style="color: #94a3b8; font-size: 1.05rem;">
            Plateforme d'Optimisation des Routes Maritimes, Détection de Ports & Simulation Stochastique de Flotte
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation Globale")
menu_selection = st.sidebar.radio(
    "Modules Métier",
    [
        "Vue Synthétique & KPIs",
        "Cartographie & DBSCAN",
        "Prédiction ETA (LightGBM)",
        "Congestion Portuaire (M/M/c)",
        "Optimisation Flotte (ALNS/MILP)",
        "Simulation Robustesse (SimPy)",
    ],
)

st.sidebar.markdown("<div class='custom-hr'></div>", unsafe_allow_html=True)
st.sidebar.subheader("Filtres Globaux")
selected_validated_only = st.sidebar.checkbox(
    "Ports Validés Uniquement", value=True
)

if selected_validated_only:
    filtered_ports = ports_df[ports_df["is_validated_port"] == True]
else:
    filtered_ports = ports_df

st.sidebar.markdown(
    """
---
**Projet PFA / Stage OCP**  
*Ingénierie des Systèmes Maritimes*  
Données : AIS USA (NOAA) 2024
"""
)

# =============================================================================
# MODULE 1 : VUE SYNTHÉTIQUE & KPIS
# =============================================================================
if menu_selection == "Vue Synthétique & KPIs":
    st.subheader("Vue d'Ensemble du Pipeline & Indicateurs Clés")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value">{len(filtered_ports):,}</div>
            <div class="metric-label">Ports Validés Détectés</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value">{len(od_df):,}</div>
            <div class="metric-label">Trajets (Matrice OD)</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-value">{len(orders_df):,}</div>
            <div class="metric-label">Commandes d'Exportation</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-value" style="color: #4ade80;">96.9%</div>
            <div class="metric-label">Taux OTD (Plan ALNS)</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='custom-hr'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("### Architecture du Pipeline End-to-End")
        st.markdown(
            r"""
        L'architecture globale de MaritimeFlow résout la chaîne de valeur complète depuis le signal brut jusqu'au plan d'affectation :
        1. **Acquisition & Nettoyage AIS** : Traitement par batchs de 2 mois (gestion de la mémoire RAM 16 Go).
        2. **Segmentation & Resampling** : Découpage par gap de 6h et ré-échantillonnage temporel à 5 min.
        3. **Détection de Stay Points & Clustering** : Extraction des zones d'arrêt (vitesse = 0 pendant > 2h) et algorithme **DBSCAN** ($\epsilon=3$ km).
        4. **Construction Matrice Origine-Destination (O-D)** : Match des clusters avec le World Port Index (< 10 km).
        5. **Modélisation Prédictive ETA & Congestion** : LightGBM pour l'ETA et files d'attente M/M/c pour la saturation.
        6. **Optimisation Métaheuristique (ALNS)** : Affectation flotte-commandes avec fenêtres de temps et coûts de demurrage.
        7. **Simulation de Robustesse (SimPy)** : Évaluation sous incertitude météo et temps d'attente.
        """
        )

    with col_right:
        st.markdown("### Répartition des Capacités de Flotte")
        if fleet_df is not None:
            fig_fleet = px.pie(
                fleet_df,
                names="vessel_class",
                values="capacity_tonnes",
                title="Capacité de Transport par Type de Navire",
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Tealgrn,
            )
            fig_fleet.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#f8fafc",
            )
            st.plotly_chart(fig_fleet, width="stretch")

# =============================================================================
# MODULE 2 : CARTOGRAPHIE & DBSCAN
# =============================================================================
elif menu_selection == "Cartographie & DBSCAN":
    st.subheader("Cartographie Interactive des Ports & Clusters DBSCAN")
    st.caption(
        "Visualisation géographique des hubs maritimes identifiés par analyse spatio-temporelle des stay points AIS."
    )

    col_map, col_details = st.columns([2.5, 1])

    with col_map:
        m = folium.Map(
            location=[29.5, -90.0],
            zoom_start=6,
            tiles="CartoDB dark_matter",
        )

        for _, row in filtered_ports.iterrows():
            is_val = row.get("is_validated_port", True)
            name = (
                row.get("matched_port_name")
                if pd.notnull(row.get("matched_port_name"))
                else f"Cluster #{int(row['port_cluster'])}"
            )

            color = "#38bdf8" if is_val else "#f43f5e"
            radius = np.sqrt(row["n_stay_points"]) / 5 + 3

            popup_html = f"""
            <div style="color: #0f172a; font-family: sans-serif;">
                <b>{name}</b><br>
                Cluster ID: {int(row['port_cluster'])}<br>
                Stay Points: {int(row['n_stay_points'])}<br>
                Durée Moyenne: {row['avg_duration_hours']:.1f}h<br>
                Validé WPI: {'Oui' if is_val else 'Non'}
            </div>
            """

            folium.CircleMarker(
                location=[row["centroid_lat"], row["centroid_lon"]],
                radius=min(radius, 18),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=250),
            ).add_to(m)

        st_folium(m, width="100%", height=520)

    with col_details:
        st.markdown("### Top Ports par Volume d'Arrêts")
        top_ports = filtered_ports.sort_values(
            by="n_stay_points", ascending=False
        ).head(10)

        fig_top = px.bar(
            top_ports,
            x="n_stay_points",
            y="matched_port_name",
            orientation="h",
            labels={
                "n_stay_points": "Nombre d'Arrêts",
                "matched_port_name": "Port",
            },
            color="avg_duration_hours",
            color_continuous_scale="Viridis",
        )
        fig_top.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f8fafc",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_top, width="stretch")

# =============================================================================
# MODULE 3 : PRÉDICTION ETA (LIGHTGBM)
# =============================================================================
elif menu_selection == "Prédiction ETA (LightGBM)":
    st.subheader("Simulateur Prédictif du Temps de Trajet (ETA)")
    st.caption(
        "Ce module utilise un modèle Gradient Boosting (LightGBM) entraîné sur les caractéristiques géométriques et saisonnières des routes."
    )

    if eta_model is None or encoder is None:
        st.warning(
            "Modèle ETA introuvable dans data/processed. Veuillez ré-exécuter le script de formation."
        )
    else:
        col_form, col_res = st.columns([1, 1.2])

        with col_form:
            st.markdown("### Paramètres de la Traversée")

            valid_routes = list(encoder.classes_)
            selected_route = st.selectbox(
                "Sélectionnez la Route (Origine -> Destination)", valid_routes
            )

            col_a, col_b = st.columns(2)
            with col_a:
                vessel_len = st.slider("Longueur du Navire (m)", 50, 400, 220)
                vessel_width = st.slider("Largeur du Navire (m)", 10, 60, 32)
            with col_b:
                vessel_draft = st.slider("Tirant d'eau (Draft) (m)", 3.0, 18.0, 10.5)
                month = st.select_slider(
                    "Mois de Départ", options=list(range(1, 13)), value=6
                )

            distance_est = st.number_input(
                "Distance Estimée de la Route (km)",
                min_value=10.0,
                max_value=15000.0,
                value=450.0,
                step=50.0,
            )

            btn_predict = st.button(
                "Calculez l'ETA Prédictif", width="stretch"
            )

        with col_res:
            st.markdown("### Résultat du Modèle Prédictif")
            if btn_predict:
                route_enc = encoder.transform([selected_route])[0]
                day_of_week = 2

                input_features = pd.DataFrame(
                    [
                        {
                            "distance_km": distance_est,
                            "Length": vessel_len,
                            "Width": vessel_width,
                            "Draft": vessel_draft,
                            "month": month,
                            "day_of_week": day_of_week,
                            "route_encoded": route_enc,
                        }
                    ]
                )

                raw_pred = eta_model.predict(input_features)[0]
                predicted_hours = float(np.expm1(raw_pred)) if raw_pred < 15 else float(raw_pred)
                predicted_days = predicted_hours / 24.0

                st.markdown(
                    f"""
                <div style="background: rgba(56, 189, 248, 0.1); border: 2px solid #38bdf8; border-radius: 16px; padding: 24px; text-align: center; margin-top: 10px;">
                    <div style="font-size: 1rem; color: #94a3b8;">TEMPS DE TRANSIT ESTIMÉ (ETA)</div>
                    <div style="font-size: 3.5rem; font-weight: 700; color: #38bdf8;">{predicted_hours:.1f} hrs</div>
                    <div style="font-size: 1.2rem; color: #f8fafc;">~ {predicted_days:.1f} Jours de Mer</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                st.markdown("#### Métriques Globales du Modèle")
                mcol1, mcol2 = st.columns(2)
                mcol1.metric(
                    label="Précision MAE (Erreur Moyenne)", value="18.4 heures"
                )
                mcol2.metric(label="Score R² (Variance Expliquée)", value="0.74")

# =============================================================================
# MODULE 4 : CONGESTION PORTUAIRE (M/M/c)
# =============================================================================
elif menu_selection == "Congestion Portuaire (M/M/c)":
    st.subheader("Analyse Théorique vs Empirique de la Congestion Portuaire")
    st.caption(
        "Comparaison des temps d'attente observés en mer avec le modèle stochastique de théorie des files d'attente (M/M/c)."
    )

    if cong_emp is not None and cong_theo is not None:
        merged_cong = pd.merge(
            cong_emp,
            cong_theo,
            on="port",
        )

        plot_df = merged_cong.copy().head(15)
        plot_df["mmc_wait_plot"] = plot_df["mmc_wait_current_hours"].replace(
            [np.inf, -np.inf], np.nan
        )

        fig_cong = go.Figure()
        fig_cong.add_trace(
            go.Bar(
                x=plot_df["port"],
                y=plot_df["wait_time_mean_hours"],
                name="Temps d'Attente Réel (AIS Empirique)",
                marker_color="#38bdf8",
            )
        )
        fig_cong.add_trace(
            go.Bar(
                x=plot_df["port"],
                y=plot_df["mmc_wait_plot"],
                name="Temps d'Attente Théorique (File M/M/c)",
                marker_color="#f43f5e",
            )
        )

        fig_cong.update_layout(
            title="Temps d'Attente Moyen aux Postes à Quai (Heures)",
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f8fafc",
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_cong, width="stretch")

        with st.expander("Pourquoi observe-t-on un écart entre le modèle M/M/c et le Réel ?"):
            st.markdown(
                """
            - **Hypothèse de Poisson** : Le modèle M/M/c suppose des arrivées aléatoires indépendantes. En pratique, les navires suivent des schedules ou arrivent par vagues.
            - **Facteurs exogènes non capturés** : Congestion liée aux conditions météo, contraintes de marées, disponibilité des remorqueurs et pilotes.
            - **Attente hors port** : Certains stay points sont des zones de mouillage commercial (attente d'ordre d'affrètement) et non de la pure congestion portuaire.
            """
            )
    else:
        st.info("Données de congestion non disponibles dans data/processed.")

# =============================================================================
# MODULE 5 : OPTIMISATION DE LA FLOTTE (ALNS / MILP)
# =============================================================================
elif menu_selection == "Optimisation Flotte (ALNS/MILP)":
    st.subheader("Planification & Optimisation de la Flotte")
    st.caption(
        "Résolution du problème d'affectation navire-commande avec fenêtres de temps (VRPTW)."
    )

    tab_alns, tab_milp = st.tabs(["Métaheuristique ALNS", "Modèle Exact MILP"])

    with tab_alns:
        if alns_df is not None:
            st.markdown("### Plan d'Affectation Optimisé (ALNS)")
            st.dataframe(alns_df, width="stretch")

            fig_cost = px.bar(
                alns_df,
                x="Commande",
                y="Coût (USD)",
                color="Navire",
                title="Répartition des Coûts d'Affrètement par Commande",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_cost.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#f8fafc",
            )
            st.plotly_chart(fig_cost, width="stretch")
        else:
            st.warning("Fichier alns_solution.csv non trouvé.")

    with tab_milp:
        if milp_df is not None:
            st.markdown("### Solution Exacte MILP (Solveur Linéaire)")
            st.dataframe(milp_df, width="stretch")
        else:
            st.warning("Fichier milp_solution.csv non trouvé.")

# =============================================================================
# MODULE 6 : SIMULATION ÉVÉNEMENTIELLE (SIMPY)
# =============================================================================
elif menu_selection == "Simulation Robustesse (SimPy)":
    st.subheader("Simulation Événementielle Stochastique (SimPy)")
    st.caption(
        "Évaluation dynamique du plan sous aléas météorologiques et temps de service aléatoires aux ports."
    )

    col_sim_controls, col_sim_view = st.columns([1, 2])

    with col_sim_controls:
        st.markdown("### Configuration de la Simulation")
        n_days = st.slider("Horizon de Simulation (Jours)", 30, 200, 100)
        seed = st.number_input("Graine Aléatoire (Seed)", value=42)

        btn_run_sim = st.button("Lancer le Benchmark SimPy", width="stretch")

    with col_sim_view:
        if btn_run_sim:
            with st.spinner("Exécution de la simulation stochastique en cours..."):
                time.sleep(1)

                try:
                    from src.simulation.fleet_simulation import run_simulation

                    res_opt = run_simulation(policy="optimized")
                    res_naive = run_simulation(policy="naive")

                    bench_df = pd.DataFrame([res_opt, res_naive])

                    st.markdown("### Résultats du Benchmark de Robustesse")
                    st.table(bench_df)

                    st.success("Simulation terminée avec succès !")

                except Exception as e:
                    st.error(f"Erreur lors de l'exécution SimPy : {e}")

# --- FOOTER ---
st.markdown("<div class='custom-hr'></div>", unsafe_allow_html=True)
st.markdown(
    """
<div style="text-align: center; color: #64748b; font-size: 0.85rem;">
    MaritimeFlow DSS © 2026 | Conçu avec Streamlit, Folium, Plotly, LightGBM & SimPy
</div>
""",
    unsafe_allow_html=True,
)

import streamlit as st
import requests
from PIL import Image
import io
import json
import os
import networkx as nx
import matplotlib.pyplot as plt
import math
import streamlit.components.v1 as components

# Configuration de la page
st.set_page_config(page_title="Projet MLOps - Reconnaissance d'Oiseaux", layout="wide")

# URLs des APIs
USER_API_URL = os.getenv("USER_API_URL", "http://user_api:5000")
ADMIN_API_URL = os.getenv("ADMIN_API_URL", "http://admin_api:5100")

# Fonction pour charger et redimensionner l'image avec une meilleure qualité
def load_and_resize_image(image_path, new_width):
    image = Image.open(image_path)
    width, height = image.size
    new_height = int(height * (new_width / width))
    return image.resize((new_width, new_height), Image.LANCZOS)

# Fonction pour créer le graphe MLOps
def create_mlops_pipeline_graph():
    G = nx.DiGraph()

    nodes = [
        ("Préparation\ndes données", "#FFB3BA"),
        ("Entraînement\ndu modèle", "#BAFFC9"),
        ("Inférence\ndu modèle", "#BAE1FF"),
        ("Déploiement", "#FFFFBA"),
        ("Interaction\nutilisateur", "#FFD700"),
        ("Traitement\nnouvelles données", "#DDA0DD"),
        ("Prédiction", "#98FB98"),
        ("Suivi des\nperformances", "#FFA07A"),
        ("Détection\nde drift", "#FF6347"),
        ("Système\nd'alerte", "#20B2AA"),
        ("Mise à jour\nBDD", "#87CEFA"),
        ("Déclenchement\nréentraînement", "#FFA500"),
        ("MLflow", "#7B68EE")
    ]

    G.add_nodes_from([node[0] for node in nodes])

    edges = [
        ("Préparation\ndes données", "Entraînement\ndu modèle"),
        ("Entraînement\ndu modèle", "Inférence\ndu modèle"),
        ("Inférence\ndu modèle", "Déploiement"),
        ("Déploiement", "Interaction\nutilisateur"),
        ("Interaction\nutilisateur", "Traitement\nnouvelles données"),
        ("Traitement\nnouvelles données", "Prédiction"),
        ("Prédiction", "Suivi des\nperformances"),
        ("Suivi des\nperformances", "Détection\nde drift"),
        ("Détection\nde drift", "Système\nd'alerte"),
        ("Traitement\nnouvelles données", "Mise à jour\nBDD"),
        ("Détection\nde drift", "Déclenchement\nréentraînement"),
        ("Déclenchement\nréentraînement", "MLflow"),
        ("MLflow", "Entraînement\ndu modèle"),
        ("Mise à jour\nBDD", "Préparation\ndes données")
    ]

    G.add_edges_from(edges)

    return G, nodes

def custom_layout(G, center=(0.5, 0.5), radius=0.45):
    pos = {}
    n = len(G.nodes())
    for i, node in enumerate(G.nodes()):
        angle = 2 * math.pi * i / n
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        pos[node] = (x, y)
    return pos

def draw_graph(G, nodes):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    pos = custom_layout(G)

    nx.draw(G, pos, ax=ax, with_labels=False, node_color=[node[1] for node in nodes],
            node_size=3000, arrowsize=20, edge_color='gray',
            width=1, arrows=True)

    # Ajout des labels
    for node, (x, y) in pos.items():
        ax.text(x, y, node, fontsize=8, ha='center', va='center', wrap=True)

    plt.title("Pipeline MLOps", fontsize=20, pad=20)
    plt.axis('off')
    return fig

# Ajouter du CSS personnalisé
st.markdown(
    """
    <style>
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }}
    .main .block-container {{
        max-width: 1200px;
        margin: auto;
    }}
    .stButton>button {{
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }}
    .stButton>button:hover {{
        background-color: #45a049;
    }}
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
        text-align: center;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar pour la navigation
with st.sidebar:
    # Ajout du logo centré
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("logo.jpeg", width=100)

    st.title("Navigation")
    page = st.selectbox("Choisissez une page", ["Présentation du Projet", "Schéma MLOps", "Architecture SVG"])

    st.markdown("---")
    st.info("Ce projet est développé dans le cadre d'un cours MLOps. Il démontre l'intégration de diverses technologies pour créer un système de reconnaissance d'oiseaux robuste et scalable, avec la participation active des utilisateurs.")

# Contenu principal
if page == "Présentation du Projet":
    st.markdown("<h1 style='text-align: center;'>Projet MLOps - Reconnaissance d'Oiseaux</h1>", unsafe_allow_html=True)

    # Chargement et affichage de l'image de couverture
    try:
        image = load_and_resize_image("oiseau_cover.jpg", 400)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Reconnaissance d'Oiseaux", use_column_width=True)
    except FileNotFoundError:
        st.warning("Image de couverture non trouvée. Veuillez vous assurer que 'oiseau_cover.jpg' est présent dans le répertoire du script.")

    # Création des onglets
    tabs = st.tabs(["Introduction", "Contexte", "Solution", "Architecture", "Participation des Utilisateurs"])

    with tabs[0]:
        st.header("Introduction et Présentation")
        st.write("""
        Bienvenue dans notre projet MLOps de reconnaissance d'oiseaux. Ce projet innovant vise à :
        - Identifier automatiquement les espèces d'oiseaux à partir d'images avec une haute précision
        - Utiliser des techniques avancées de deep learning, notamment EfficientNetB0
        - Appliquer les meilleures pratiques MLOps pour un déploiement robuste, scalable et maintenable
        - Impliquer activement les utilisateurs dans l'amélioration continue du modèle

        Notre solution combine l'intelligence artificielle de pointe, les principes MLOps, et la participation communautaire pour créer un outil puissant et évolutif.
        """)

    with tabs[1]:
        st.header("Contexte et Problématique")
        st.write("""
        La biodiversité aviaire fait face à des défis sans précédent. Notre projet répond à ces enjeux en offrant :
        - Une identification rapide et précise des espèces d'oiseaux
        - Un outil participatif permettant aux utilisateurs de contribuer à l'enrichissement des données
        - Une plateforme d'apprentissage continu, s'adaptant aux nouvelles espèces et variations
        """)

    with tabs[2]:
        st.header("Solution")
        st.write("""
        Notre solution MLOps complète et participative comprend :
        1. API Utilisateur pour soumettre des images et recevoir des prédictions
        2. Système de contribution permettant aux utilisateurs d'enrichir le dataset
        3. Processus automatisé d'intégration des nouvelles données et de mise à jour du modèle
        4. Mécanisme de création de nouvelles classes pour les espèces non identifiées
        5. Plateforme communautaire pour l'identification collaborative des espèces inconnues
        """)

    with tabs[3]:
        st.header("Architecture")
        st.write("""
        Notre architecture basée sur Docker assure une portabilité et une scalabilité exceptionnelles, tout en facilitant la contribution des utilisateurs :
        - Conteneurs spécialisés pour chaque composant du système
        - Intégration fluide des contributions des utilisateurs dans le pipeline de données
        - Mécanismes de validation et d'intégration des nouvelles espèces
        - Système de stockage et de traitement des images non identifiées
        """)

    with tabs[4]:
        st.header("Participation des Utilisateurs")
        st.write("""
        Notre projet se distingue par son approche participative :
        1. Les utilisateurs peuvent soumettre leurs propres photos d'oiseaux
        2. Si l'espèce est reconnue, l'image enrichit le dataset existant
        3. Pour les nouvelles espèces, une nouvelle classe est créée automatiquement
        4. Les images d'espèces inconnues sont stockées pour une identification communautaire
        5. Ce processus permet une amélioration continue du modèle et une extension de sa couverture
        """)

elif page == "Schéma MLOps":
    st.markdown("<h1 style='text-align: center;'>Diagramme de la Pipeline MLOps</h1>", unsafe_allow_html=True)

    G, nodes = create_mlops_pipeline_graph()
    fig = draw_graph(G, nodes)

    # Convertir le graphique en image pour permettre le zoom
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    st.image(buf, use_column_width=True)

    # Ajouter un bouton pour télécharger l'image en haute résolution
    btn = st.download_button(
        label="Télécharger l'image en haute résolution",
        data=buf,
        file_name="mlops_pipeline.png",
        mime="image/png"
    )

    st.write("### Description détaillée des composants de la pipeline MLOps")
    descriptions = {
        "Préparation des données": "Processus de nettoyage, normalisation et augmentation du dataset, incluant l'intégration des nouvelles images soumises par les utilisateurs.",
        "Entraînement du modèle": "Utilisation d'EfficientNetB0 avec des couches personnalisées, optimisé pour la classification des oiseaux, incluant l'adaptation aux nouvelles classes.",
        "Inférence du modèle": "Déploiement du modèle entraîné pour des prédictions en temps réel sur les nouvelles images soumises.",
        "Déploiement": "Mise en production du modèle via une API REST, avec mise à jour régulière basée sur les nouvelles données.",
        "Interaction utilisateur": "Interface permettant aux utilisateurs de soumettre des images, recevoir des prédictions, et contribuer à l'enrichissement du dataset.",
        "Traitement nouvelles données": "Intégration et prétraitement des nouvelles images soumises par les utilisateurs, y compris la gestion des nouvelles espèces.",
        "Prédiction": "Génération de prédictions sur les nouvelles images soumises par les utilisateurs.",
        "Suivi des performances": "Monitoring continu de l'accuracy et d'autres métriques clés, crucial avec l'ajout constant de nouvelles données.",
        "Détection de drift": "Analyse des changements dans la distribution des données ou les performances du modèle, particulièrement important avec l'évolution du dataset.",
        "Système d'alerte": "Notification des administrateurs en cas de dégradation des performances ou de drift détecté.",
        "Mise à jour BDD": "Intégration des nouvelles images validées et création de nouvelles classes dans la base de données d'entraînement.",
        "Déclenchement réentraînement": "Lancement automatique d'un nouveau cycle d'entraînement basé sur les nouvelles données et les résultats du monitoring.",
        "MLflow": "Tracking des expériences, gestion des versions des modèles et des artefacts, crucial pour suivre l'évolution du modèle avec les nouvelles contributions."
    }

    for node, description in descriptions.items():
        with st.expander(f"{node}"):
            st.write(description)

elif page == "Architecture SVG":
    st.markdown("<h1 style='text-align: center;'>Architecture du Projet</h1>", unsafe_allow_html=True)

    # Charger et afficher l'image SVG avec zoom
    try:
        with open("Architecture.svg", "r", encoding='utf-8') as svg_file:
            svg_content = svg_file.read()

        # Corriger les symboles
        svg_content = svg_content.replace('DonnÃ©es', 'Données')
        svg_content = svg_content.replace('ModÃ¨les', 'Modèles')
        svg_content = svg_content.replace('traitÃ©es', 'traitées')
        svg_content = svg_content.replace('archivÃ©', 'archivé')

        # Utiliser un composant HTML personnalisé pour le zoom
        components.html(f"""
        <div id="svg-container" style="width: 100%; height: 600px; border: 1px solid #ddd; overflow: hidden;">
            {svg_content}
        </div>
        <script src="https://unpkg.com/panzoom@9.4.0/dist/panzoom.min.js"></script>
        <script>
            const element = document.getElementById('svg-container');
            const svgElement = element.querySelector('svg');
            svgElement.style.width = '100%';
            svgElement.style.height = '100%';
            panzoom(svgElement, {{
                maxZoom: 5,
                minZoom: 0.5,
                bounds: true,
                boundsPadding: 0.1
            }});
        </script>
        """, height=650)

        st.info("Utilisez la molette de la souris pour zoomer/dézoomer. Cliquez et faites glisser pour vous déplacer dans l'image.")
    except FileNotFoundError:
        st.error("Le fichier Architecture.svg n'a pas été trouvé. Assurez-vous qu'il est présent dans le répertoire du script.")

    # Ajout d'explications détaillées sur l'architecture
    st.write("""
    ### Explication détaillée de l'Architecture MLOps

    Notre architecture MLOps intègre activement les contributions des utilisateurs :

    1. **Gestion des données** :
       - Responsable de l'acquisition, du nettoyage et de l'augmentation des données, y compris les nouvelles images soumises par les utilisateurs.
       - Gère la création de nouvelles classes pour les espèces non répertoriées.

    2. **Entraînement** :
       - Orchestre l'entraînement du modèle EfficientNetB0, s'adaptant aux nouvelles classes et données.
       - Utilise MLflow pour le suivi des expériences et la gestion des versions, crucial avec l'évolution constante du dataset.

    3. **Production** :
       - Héberge le modèle optimisé pour des prédictions en temps réel sur les nouvelles images soumises.
       - Se met à jour régulièrement pour intégrer les améliorations basées sur les contributions des utilisateurs.

    4. **API** :
       - Fournit des endpoints pour la soumission d'images, la récupération des prédictions, et la gestion des contributions utilisateurs.
       - Gère l'authentification et les autorisations pour sécuriser les contributions.

    5. **Interface** :
       - Interface Streamlit intuitive permettant aux utilisateurs de soumettre des images, voir les prédictions, et contribuer au dataset.
       - Offre des visualisations des performances du modèle et de l'évolution du dataset.

    6. **Monitoring** :
       - Surveille en temps réel les performances du modèle, particulièrement important avec l'ajout constant de nouvelles données.
       - Détecte les drifts potentiels causés par l'évolution du dataset.

    7. **MLflow** :
       - Centralise la gestion des expériences, des modèles et des métriques.
       - Crucial pour suivre l'évolution du modèle avec l'intégration continue de nouvelles données et classes.

    Cette architecture supporte efficacement le flux de travail participatif, permettant une amélioration continue du modèle grâce aux contributions des utilisateurs.
    """)

# Pied de page
st.sidebar.markdown("---")
st.sidebar.info("Développé par :\n- Maxence REMY-HAROCHE\n- Guillaume RUIZ\n- Yoni EDERY")

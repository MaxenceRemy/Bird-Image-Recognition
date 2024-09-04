import time
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

if 'specie' not in st.session_state:
    st.session_state.specie = 0

if 'selected_specie' not in st.session_state:
    st.session_state.selected_specie = "Sélectionnez une espèce..."

if 'success' not in st.session_state:
    st.session_state.success = False


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
    page = st.selectbox("Choisissez une page", ["Présentation du Projet", "Interface Utilisateur (APIs)", "Schéma MLOps", "Architecture SVG", "MLflow"])

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

elif page == "Interface Utilisateur (APIs)":
    st.title("Interface Utilisateur (APIs)")
    
    # Sélection de l'API
    api_choice = st.radio("Choisissez une API", ("Utilisateur", "Admin"))

    if api_choice == "Utilisateur":
        st.subheader("API Utilisateur")
        
        # Authentification
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        if st.button("Se connecter"):
            try:
                response = requests.post(f"{USER_API_URL}/token",
                                         data={"username": username, "password": password})
                if response.status_code == 200:
                    st.success(f"Connecté en tant que {username}")
                    st.session_state.user_token = response.json()["access_token"]
                else:
                    st.error("Échec de l'authentification")
            except requests.RequestException:
                st.error("Impossible de se connecter à l'API. Assurez-vous que les conteneurs Docker sont en cours d'exécution.")


        if st.session_state.success == True:
            st.toast("Merci pour votre précieuse contribution !")
            st.session_state.success = False


        # Prédiction
        if 'user_token' in st.session_state:
            uploaded_file = st.file_uploader("Choisissez une image d'oiseau", type=["jpg", "png"])
            col1, col2, col3, col4 = st.columns([0.2, 0.2, 0.2, 0.22], vertical_alignment = "bottom")
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                with col1:
                    st.image(image, caption="Image téléchargée")
                
                try:
                    files = {"file": ("image.jpg", uploaded_file.getvalue(), "image/jpeg")}
                    headers = {
                        "Authorization": f"Bearer {st.session_state.user_token}",
                        "api-key": "abcd1234"
                    }
                    response = requests.post(f"{USER_API_URL}/predict", files=files, headers=headers)
                    if response.status_code == 200:
                        prediction = response.json()
                        with col2:
                            response = requests.get(f"{USER_API_URL}/get_class_image", params={'classe': prediction['predictions'][0]}, headers=headers)
                            st.image(response.content)
                            st.markdown(f"""<p style="font-size: 20px;"> {prediction['predictions'][0]} </p>""", unsafe_allow_html = True)
                            st.markdown(f"""<p style="color: #079e20; font-size: 20px;"> {str(round(prediction['scores'][0] * 100, 2)) + "%"} </p>""", unsafe_allow_html = True)
                        with col3:
                            response = requests.get(f"{USER_API_URL}/get_class_image", params={'classe': prediction['predictions'][1]}, headers=headers)
                            st.image(response.content)
                            st.markdown(f"""<p style="font-size: 20px;"> {prediction['predictions'][1]} </p>""", unsafe_allow_html = True)
                            st.markdown(f"""<p style="color: #d1ae29; font-size: 20px;"> {str(round(prediction['scores'][1] * 100, 2)) + "%"} </p>""", unsafe_allow_html = True)
                        with col4:
                            response = requests.get(f"{USER_API_URL}/get_class_image", params={'classe': prediction['predictions'][2]}, headers=headers)
                            st.image(response.content)
                            st.markdown(f"""<p style="font-size: 20px;"> {prediction['predictions'][2]} </p>""", unsafe_allow_html = True)
                            st.markdown(f"""<p style="color: #b26a19; font-size: 20px;"> {str(round(prediction['scores'][2] * 100, 2)) + "%"} </p>""", unsafe_allow_html = True)


                        st.subheader("Une des prédictions est-elle correcte ?")
                        col1, col2, col3 = st.columns([0.1, 0.4, 0.5])
                        with col1: 
                            if st.button("Oui"):
                                st.session_state.specie = 1
                        with col2:
                            if st.button("Non, mais je connais l'espèce correcte"):
                                st.session_state.specie = 2
                        
                        with col3:
                            if st.button("Je ne suis pas sûr"):
                                data = {"species": "NA", "image_name": prediction['filename'], "is_unknown": True}
                                response = requests.post(f"{ADMIN_API_URL}/add_image", headers=headers, data=data)
                                st.session_state.success = True
                                st.session_state.specie = 0
                                st.rerun()

                       

                        if st.session_state.specie == 1 and st.session_state.success == False:
                            st.session_state.selected_specie = st.selectbox("Sélectionnez l'espèce correcte :", ["Sélectionnez une espèce..."] + prediction['predictions'])
                        if st.session_state.specie == 2 and st.session_state.success == False:
                            response = requests.get(f"{USER_API_URL}/get_species", headers=headers)
                            species_list = response.json()
                            st.session_state.selected_specie = st.selectbox("Sélectionnez l'espèce correcte :", ["Sélectionnez une espèce..."] + species_list['species'])
                        if st.session_state.selected_specie != "Sélectionnez une espèce...":
                            data = {"species": st.session_state.selected_specie, "image_name": prediction['filename'], "is_unknown": False}
                            response = requests.post(f"{ADMIN_API_URL}/add_image", headers=headers, data=data)
                            st.session_state.success = True
                            st.session_state.specie = 0
                            st.session_state.selected_specie = "Sélectionnez une espèce..."
                            st.rerun()
                    else:
                        st.error("Échec de la prédiction")
                except requests.RequestException:
                    st.error("Impossible de communiquer avec l'API d'inférence")




    else:
        st.subheader("API Admin")
        
        # Authentification
        admin_username = st.text_input("Nom d'administrateur")
        admin_password = st.text_input("Mot de passe administrateur", type="password")
        if st.button("Se connecter (Admin)"):
            try:
                response = requests.post(f"{ADMIN_API_URL}/token",
                                         data={"username": admin_username, "password": admin_password})
                if response.status_code == 200:
                    st.success(f"Connecté en tant qu'administrateur {admin_username}")
                    st.session_state.admin_token = response.json()["access_token"]
                else:
                    st.error("Échec de l'authentification admin")
            except requests.RequestException:
                st.error("Impossible de se connecter à l'API admin. Assurez-vous que les conteneurs Docker sont en cours d'exécution.")

        if 'admin_token' in st.session_state:
            # Ajout d'un utilisateur
            st.subheader("Ajouter un utilisateur")
            new_username = st.text_input("Nouveau nom d'utilisateur")
            new_password = st.text_input("Nouveau mot de passe", type="password")
            is_admin = st.checkbox("Est administrateur")
            if st.button("Ajouter l'utilisateur"):
                try:
                    headers = {
                        "Authorization": f"Bearer {st.session_state.admin_token}",
                        "api-key": "abcd1234"
                    }
                    data = {"new_username": new_username, "user_password": new_password, "is_admin": is_admin}
                    response = requests.post(f"{ADMIN_API_URL}/add_user", headers=headers, data=data)
                    if response.status_code == 200:
                        st.success(f"Utilisateur {new_username} ajouté avec succès")
                    else:
                        st.error("Échec de l'ajout de l'utilisateur")
                except requests.RequestException:
                    st.error("Impossible de communiquer avec l'API admin")

            # Lancer l'entraînement
            if st.button("Lancer l'entraînement"):
                try:
                    headers = {
                        "Authorization": f"Bearer {st.session_state.admin_token}",
                        "api-key": "abcd1234"
                    }
                    response = requests.get(f"{ADMIN_API_URL}/train", headers=headers)
                    if response.status_code == 200:
                        st.info(response.json())
                    else:
                        st.error("Échec du lancement de l'entraînement")
                except requests.RequestException:
                    st.error("Impossible de communiquer avec l'API d'entraînement")

            # Afficher les résultats
            if st.button("Afficher les résultats de l'entraînement"):
                try:
                    headers = {
                        "Authorization": f"Bearer {st.session_state.admin_token}",
                        "api-key": "abcd1234"
                    }
                    response = requests.get(f"{ADMIN_API_URL}/results", headers=headers)
                    if response.status_code == 200:
                        results = response.json()
                        st.write(f"Résultats de l'entraînement : {results}")
                    else:
                        st.error("Impossible d'obtenir les résultats de l'entraînement")
                except requests.RequestException:
                    st.error("Impossible de communiquer avec l'API admin")

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

elif page == "MLflow":
    st.title("MLflow Dashboard")
    
    if 'admin_token' in st.session_state:
        try:
            headers = {
                "Authorization": f"Bearer {st.session_state.admin_token}",
                "api-key": "abcd1234"
            }
            response = requests.get(f"{ADMIN_API_URL}/results", headers=headers, timeout=10)
            if response.status_code == 200:
                results = response.json()
                st.write("### Résultats de l'entraînement le plus récent")
                st.json(results)
                
                # Ajouter un lien vers l'interface MLflow
                st.markdown(f"[Ouvrir l'interface MLflow complète](http://localhost:5200)")
            else:
                st.error("Impossible d'obtenir les résultats de l'entraînement")
                st.error(f"Code d'état : {response.status_code}")
                st.error(f"Détails de l'erreur : {response.text}")
        except requests.RequestException as e:
            st.error(f"Impossible de communiquer avec l'API admin. Erreur : {str(e)}")
            st.info("Vérifiez que tous les conteneurs sont en cours d'exécution et que les ports sont correctement configurés.")
    else:
        st.warning("Veuillez vous connecter en tant qu'administrateur pour accéder aux résultats MLflow.")
        st.info("Allez dans l'onglet 'Interface Utilisateur (APIs)' et connectez-vous en tant qu'admin.")

# Pied de page
st.sidebar.markdown("---")
st.sidebar.info("Développé par :\n- Maxence REMY-HAROCHE\n- Guillaume RUIZ\n- Yoni EDERY")
import streamlit as st
import requests
from PIL import Image
import io
import json

# Configuration de la page
st.set_page_config(page_title="Projet MLOps - Reconnaissance d'Oiseaux", layout="wide")

# Fonction pour charger et redimensionner l'image
def load_and_resize_image(image_path, new_width):
    image = Image.open(image_path)
    width, height = image.size
    new_height = int(height * (new_width / width))
    return image.resize((new_width, new_height))

# Chargement et affichage de l'image de couverture
try:
    image = load_and_resize_image("oiseau_cover.jpg", 400)
    st.image(image, caption="Reconnaissance d'Oiseaux")
except FileNotFoundError:
    st.warning("Image de couverture non trouvée. Veuillez vous assurer que 'oiseau_cover.jpg' est présent dans le répertoire du script.")

# Titre principal
st.title("Projet MLOps - Reconnaissance d'Oiseaux")

# Création des onglets
tabs = st.tabs(["Introduction", "Contexte", "Solution", "Architecture", "Interface Utilisateur (APIs)"])

with tabs[0]:
    st.header("Introduction et Présentation")
    st.write("""
    Bienvenue dans notre projet MLOps de reconnaissance d'oiseaux. 
    Ce projet vise à identifier automatiquement les espèces d'oiseaux à partir d'images 
    en utilisant des techniques de deep learning et des pratiques MLOps.
    """)

with tabs[1]:
    st.header("Contexte et Problématique")
    st.write("""
    La biodiversité aviaire est menacée par divers facteurs environnementaux. 
    Une identification précise et rapide des espèces d'oiseaux est cruciale pour la conservation 
    et la recherche ornithologique. Notre solution répond à ce besoin en fournissant un outil 
    d'identification automatique basé sur l'intelligence artificielle.
    """)

with tabs[2]:
    st.header("Solution")
    st.write("""
    Notre solution comprend plusieurs composants clés :
    1. API Utilisateur : Permet aux utilisateurs d'envoyer des images et de recevoir des prédictions.
    2. API Admin : Offre des fonctionnalités de gestion et de contrôle du système.
    3. Prétraitement : Prépare les données pour l'entraînement et l'inférence.
    4. Inférence : Utilise le modèle entraîné pour faire des prédictions sur de nouvelles images.
    5. Entraînement : Met à jour régulièrement le modèle avec de nouvelles données.
    """)

with tabs[3]:
    st.header("Architecture")
    st.write("""
    Notre architecture est basée sur des conteneurs Docker pour assurer la portabilité et la scalabilité :
    - user_api : Gère les interactions avec les utilisateurs finaux.
    - admin_api : Fournit des fonctionnalités d'administration.
    - inference : Effectue les prédictions sur les nouvelles images.
    - preprocessing : Prépare les données pour l'entraînement et l'inférence.
    - training : Entraîne et met à jour le modèle de reconnaissance.

    Tous ces conteneurs partagent un volume Docker commun pour la persistance des données.
    """)

with tabs[4]:
    st.header("Interface Utilisateur (APIs)")
    
    # Sélection de l'API
    api_choice = st.radio("Choisissez une API", ("Utilisateur", "Admin"))

    if api_choice == "Utilisateur":
        st.subheader("API Utilisateur")
        
        # Authentification
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        if st.button("Se connecter"):
            try:
                response = requests.post("http://user_api:5000/token", 
                                         data={"username": username, "password": password})
                if response.status_code == 200:
                    st.success(f"Connecté en tant que {username}")
                    st.session_state.user_token = response.json()["access_token"]
                else:
                    st.error("Échec de l'authentification")
            except requests.RequestException:
                st.error("Impossible de se connecter à l'API. Assurez-vous que les conteneurs Docker sont en cours d'exécution.")

        # Prédiction
        if 'user_token' in st.session_state:
            uploaded_file = st.file_uploader("Choisissez une image d'oiseau", type=["jpg", "png"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Image téléchargée", use_column_width=True)
                if st.button("Prédire"):
                    try:
                        files = {"file": ("image.jpg", uploaded_file.getvalue(), "image/jpeg")}
                        headers = {
                            "Authorization": f"Bearer {st.session_state.user_token}",
                            "api-key": "abcd1234"
                        }
                        response = requests.post("http://user_api:5000/predict", files=files, headers=headers)
                        if response.status_code == 200:
                            prediction = response.json()
                            st.success(f"Prédiction : {prediction['prediction']}, Score : {prediction['score']}")
                        else:
                            st.error("Échec de la prédiction")
                    except requests.RequestException:
                        st.error("Impossible de communiquer avec l'API d'inférence")

            # Liste des espèces
            if st.button("Obtenir la liste des espèces"):
                try:
                    headers = {
                        "Authorization": f"Bearer {st.session_state.user_token}",
                        "api-key": "abcd1234"
                    }
                    response = requests.get("http://user_api:5000/get_species", headers=headers)
                    if response.status_code == 200:
                        species = response.json()["species"]
                        st.write(species)
                    else:
                        st.error("Impossible d'obtenir la liste des espèces")
                except requests.RequestException:
                    st.error("Impossible de communiquer avec l'API")

    else:
        st.subheader("API Admin")
        
        # Authentification
        admin_username = st.text_input("Nom d'administrateur")
        admin_password = st.text_input("Mot de passe administrateur", type="password")
        if st.button("Se connecter (Admin)"):
            try:
                response = requests.post("http://admin_api:5100/token", 
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
                    response = requests.post("http://admin_api:5100/add_user", headers=headers, data=data)
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
                    response = requests.get("http://admin_api:5100/train", headers=headers)
                    if response.status_code == 200:
                        st.info("Entraînement lancé. Veuillez patienter...")
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
                    response = requests.get("http://admin_api:5100/results", headers=headers)
                    if response.status_code == 200:
                        results = response.json()
                        st.write(f"Résultats de l'entraînement : {results}")
                    else:
                        st.error("Impossible d'obtenir les résultats de l'entraînement")
                except requests.RequestException:
                    st.error("Impossible de communiquer avec l'API admin")

st.sidebar.title("À propos")
st.sidebar.info("Ce projet est développé dans le cadre d'un cours MLOps. Il démontre l'intégration de diverses technologies pour créer un système de reconnaissance d'oiseaux robuste et scalable.")
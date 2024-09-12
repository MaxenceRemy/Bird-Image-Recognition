import streamlit as st
import requests
from PIL import Image
import os
import streamlit.components.v1 as components

# Configuration de la page
st.set_page_config(page_title="Projet MLOps - Reconnaissance d'oiseaux", layout="wide")

# URLs des APIs
USER_API_URL = os.getenv("USER_API_URL", "http://user_api:5000")
ADMIN_API_URL = os.getenv("ADMIN_API_URL", "http://admin_api:5100")

# Définition des variables de session utilisées durant la prédicition
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


# Sidebar pour la navigation
with st.sidebar:
    # Ajout du logo centré
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("logo.jpeg", width=100)

    st.title("Navigation")
    page = st.selectbox(
        "Choisissez une page",
        [
            "Présentation du projet",
            "Technologies",
            "Schémas",
            "Résultats de l'entraînement",
            "Interface utilisateur (APIs)",
            "Conclusion"
        ]
    )

    st.markdown("---")

    mystyle = """
        <style>
            p {
                text-align: justify;
            }
        </style>
        """
    st.markdown(mystyle, unsafe_allow_html=True)
    st.info("Ce projet est développé dans le cadre d'une formation MLOps. Il démontre l'intégration de diverses \
            technologies pour créer un système de reconnaissance d'oiseaux robuste et scalable, \
            avec la participation active des utilisateurs.")

# Contenu principal
if page == "Présentation du projet":

    st.markdown("<h1 style='text-align: center;'>Projet MLOps - Reconnaissance d'oiseaux</h1>", unsafe_allow_html=True)

    # Chargement et affichage de l'image de couverture
    try:
        image = load_and_resize_image("oiseau_cover.jpg", 400)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Reconnaissance d'oiseaux", use_column_width=True)
    except FileNotFoundError:
        st.warning("Image de couverture non trouvée. \
                   Veuillez vous assurer que 'oiseau_cover.jpg' est présent dans le répertoire du script.")

    st.write("""
    Bienvenue dans notre projet MLOps de reconnaissance d'oiseaux.
    - Application de reconnaissance d'oiseaux
    - Suite d’un projet de cursus Data Scientist
    - Problématiques environnementales
    - Coopération et intérêt partagé
    """)

if page == "Technologies":
    st.title("Technologies")
    st.write("")

    logo_list = [
        "python_logo.png",
        "tensorflow_logo.png",
        "mlflow_logo.png",
        "docker_logo.png",
        "github_logo.png"
    ]

    texts = [
        "Python fut utilisé pour rédiger l'intégralité des scripts nous permettant d'exécuter une pipeline CI/CD \
            complète, automatisant ainsi l'intégration continue et le déploiement de notre application avec une \
                grande efficacité.",
        "Notre modèle de classification a été développé avec TensorFlow, couplé à un modèle pré-entrainé \
            EfficientNetB0, qui assure la partie convolutive de notre réseau de neurones, \
                permettant une meilleure précision et une optimisation des performances lors de l'entraînement.",
        "Dans le cadre d'un suivi rigoureux des cycles d'entraînement de notre modèle, MLflow assure un historique \
            permanent des différentes versions de notre modèle, facilitant ainsi la gestion, la comparaison et \
                l'amélioration continue des résultats obtenus.",
        "Nous avons utilisé Docker pour isoler les différentes parties de notre application dans des conteneurs qui \
            fonctionnent ensemble de manière fluide, assurant ainsi non seulement la portabilité de notre application \
                sur diverses plateformes, mais aussi une gestion simplifiée de ses composants.",
        "GitHub nous offre une plateforme idéale pour travailler en groupe dans les meilleures conditions, \
            nous permettant également de vérifier le bon fonctionnement des différentes fonctions de notre projet, \
                tout en garantissant la collaboration entre les membres de l'équipe et le respect strict de la \
                    convention PEP 8."
    ]

    # On affiche les textes avec leur logos
    for i, text in enumerate(texts):
        col1, col2 = st.columns([1, 8])

        with col1:
            st.image(logo_list[i], width=65)

        with col2:
            st.write(f"###### {texts[i]}")

        st.markdown("---")

elif page == "Schémas":

    # On permet de sélectionner le schéma à afficher
    choix_schema = st.radio(
        "Choisissez un schéma :",
        [
            "Interaction utilisateur",
            "Architecture",
            'Pipeline'
        ]
    )


    if choix_schema == "Interaction utilisateur":
        st.markdown("<h1 style='text-align: center;'>Pipeline utilisateur</h1>", unsafe_allow_html=True)

        # On cahrge et affiche l'image SVG avec possibilité de zoomer
        try:
            with open("pipeline_user.svg", "r", encoding='utf-8') as svg_file:
                svg_content = svg_file.read()

        except FileNotFoundError:
            st.error("Le fichier 'pipeline_user.svg' n'a pas été trouvé. \
                     Assurez-vous qu'il est présent dans le répertoire du script.")

        # On utilise un composant HTML personnalisé pour le zoom
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

        st.info("Utilisez la molette de la souris pour zoomer/dézoomer. \
                Cliquez et faites glisser pour vous déplacer dans l'image.")

        st.write("""
        ### Explication détaillée de la pipeline utilisateur

        Notre architecture MLOps intègre activement les contributions des utilisateurs :

        1. **Upload de l'image** :
        - L'image est d'abord envoyée par l'utilisateur sur le serveur, \
                 où elle est conservée dans un fichier temporaire.

        2. **Inférence** :
        - La prédiction de classe est effectuée en quelques centaines de millisecondes.
        - Le résultat comprends les 3 classes les plus probables ainsi que leurs scores.
        - Sont aussi affichées les images des classes en question pour aider l'utilisateur.

        3. **Feedback** :
        - Si l'utilisateur indique que la prédiction est correcte, on ajoute son image au dataset.
        - Si il indique que la prédiction est fausse mais connaît l'espèce, \
                 il l'indique et on ajoute son image au dataset.
        - Si il ne connaît pas l'image, on ajoute son image dans un dossier qui sera traité plus tard.

        Cette architecture supporte efficacement le flux de travail participatif, \
                 permettant une amélioration continue du modèle grâce aux contributions des utilisateurs.
        """)

    elif choix_schema == "Architecture":
        st.markdown("<h1 style='text-align: center;'>Architecture du projet</h1>", unsafe_allow_html=True)

        # On charge et affiche l'image SVG avec zoom
        try:
            with open("architecture.svg", "r", encoding='utf-8') as svg_file:
                svg_content = svg_file.read()

        except FileNotFoundError:
            st.error("Le fichier 'architecture.svg' n'a pas été trouvé. \
                     Assurez-vous qu'il est présent dans le répertoire du script.")

        # On corrige les symboles
        svg_content = svg_content.replace('DonnÃ©es', 'Données')
        svg_content = svg_content.replace('ModÃ¨les', 'Modèles')
        svg_content = svg_content.replace('traitÃ©es', 'traitées')
        svg_content = svg_content.replace('archivÃ©', 'archivé')

        # On utilise un composant HTML personnalisé pour le zoom
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

        st.info("Utilisez la molette de la souris pour zoomer/dézoomer. \
                Cliquez et faites glisser pour vous déplacer dans l'image.")

        st.write("""
        ### Explication détaillée de l'Architecture MLOps

        Notre architecture MLOps intègre activement les contributions des utilisateurs :

        1. **Gestion des données** :
        - Entièrement autonome.
        - Responsable de l'acquisition, du nettoyage et de l'augmentation des données, \
                 y compris les nouvelles images soumises par les utilisateurs.
        - Gère la création de nouvelles classes pour les espèces non répertoriées.

        2. **Entraînement** :
        - Orchestre l'entraînement du modèle EfficientNetB0, s'adaptant aux nouvelles classes et données.
        - Utilise MLflow pour le suivi des expériences et la gestion des versions, \
                 crucial avec l'évolution constante du dataset.

        3. **Production** :
        - Héberge le modèle optimisé pour des prédictions en temps réel sur les nouvelles images soumises.
        - Se met à jour régulièrement pour intégrer les améliorations basées sur les contributions des utilisateurs.

        4. **API client** :
        - Fournit des endpoints pour la soumission d'images, la récupération des prédictions, \
                 et la gestion des contributions utilisateurs.
        - Gère l'authentification et les autorisations pour sécuriser les contributions.

        5. **API administrative** :
        - Fournit des endpoints pour la gestion des données, des utilisateurs et des entraînements, \
                 pour la comparaison des métriques et le choix du modèle en production.

        6. **Interface** :
        - Interface Streamlit intuitive permettant aux utilisateurs de soumettre des images, voir les prédictions, \
                 et contribuer au dataset.
        - Offre des visualisations des performances du modèle et de l'évolution du dataset.

        7. **Monitoring** :
        - Surveille en temps réel les performances du modèle, \
                 particulièrement important avec l'ajout constant de nouvelles données.
        - Détecte les drifts potentiels causés par l'évolution du dataset.

        8. **MLflow** :
        - Centralise la gestion des expériences, des modèles et des métriques.
        - Crucial pour suivre l'évolution du modèle avec l'intégration continue de nouvelles données et classes.

        Cette architecture supporte efficacement le flux de travail participatif, \
                 permettant une amélioration continue du modèle grâce aux contributions des utilisateurs.
        """)

    elif choix_schema == "Pipeline":
        st.markdown("<h1 style='text-align: center;'>Pipeline MLOPS</h1>", unsafe_allow_html=True)

        # On charge et on affiche l'image SVG avec zoom
        try:
            with open("pipeline_mlops.svg", "r", encoding='utf-8') as svg_file:
                svg_content = svg_file.read()

        except FileNotFoundError:
            st.error("Le fichier 'pipeline_mlops.svg' n'a pas été trouvé. \
                     Assurez-vous qu'il est présent dans le répertoire du script.")

        # On utilise un composant HTML personnalisé pour le zoom
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

        st.info("Utilisez la molette de la souris pour zoomer/dézoomer. \
                Cliquez et faites glisser pour vous déplacer dans l'image.")

        st.write("""
        ### Explication détaillée de la Pipeline MLOps

        Notre pipeline permet une actualisation des données constantes ainsi \
                 qu'un modèle toujours performant et évolutif.

        1. **Arrivée de nouvelles données** :
        Les données sont téléchargées et corrigées dans un dossier dataset_raw (données brutes) puis déplacées \
                 dans un dossier dataset_clean une fois ayant passé l'étape du traitement des données (preprocessing).
        Tout nouvelle image arrive d'abord dans dataset_raw et se retrouve plus tard dans dataset_clean \
                 lors du preprocessing, qu'elle soit ajoutée par l'utilisateur ou qu'elle provienne de Kaggle.
        Un processus automatique chaque jour vérifie la disponibilité d'un nouveau dataset sur Kaggle. \
                 Si une version plus récente est disponible, elle est téléchargée.
        Dès qu'un nouveau dataset est téléchargé, le preprocessing des données se lance directement.

        2. **Traitement des données** :
        Lors du lancement d'un preproccessing, les labels du dataset Kaggle sont corrigés avec les noms corrects \
                 et qui correspondent à la liste qui permet d'ajouter une nouvelle espèce.
        Aussi, les dossiers train/test/valid sont répartis avec un équilibre de 70%/15%/15%.
        Le script s'assure de suivre les nouvelles images et classes :
        - Lorsqu'un certain nombre de nouvelles images (au moins l'équivalent de 1% de la totalité du dataset) \
                 est ajouté par les utilisateurs, un preprocessing est lancé.
        - Lorsqu'une nouvelle classe apparaît, elle ne passe jamais à travers le script de preprocessing, \
                 même si déclenché par les événements ci-dessus, tant qu'elle n'a pas suffisamment d'images.
        On considère qu'une classe est complète lorsqu'elle dispose au moins du même nombre d'images que la classe \
                 la plus petite de notre dataset. Lorsque la condition est remplie, \
                 un preprocessing se lance et cette classe se voit intégrée.
        Dès qu'un preprocessing est lancé, cela signifie que les données ont changé de manière suffisante pour \
                 déclencher une alerte aux administrateurs et les encourager à réentraîner le modèle.

        3. **Entraînement du modèle** :
        Un entraînement du modèle peut-être déclenché manuellement par un administrateur, par exemple car il a reçu \
                 une alerte indiquant une dérive du modèle ou l'arrivée de nombreuses nouvelles données.
        Lorsqu'un entraînement est terminé, l'administrateur est notifié et les informations relatives au nouveau \
                 modèle sont enregistrées dans MLflow.
        Aussi, une matrice de confusion couplée avec un rapport de classification est sauvegardée pour faire état de \
                 la performance du modèle à sa création.

        4. **MLflow** :
        Durant et après un entraînement, MLflow s'assure du suivi des métriques et de l'enregistrement des modèles \
                 ainsi que de la matrice évoquée plus haut.
        Il dispose d'une interface permettant d'afficher toutes les informations relatives aux entraînements \
                 et de faire les comparaisons nécessaires entre les modèles.
        Son intérêt est également de servir de moyen d'archive des modèles générés.

        5. **Évaluation du modèle** :
        Une fois un entraînement terminé, l'administrateur peut afficher une comparaison entre le modèle en \
                 production et celui qui vient d'être entraîné.
        Pour cela, il dispose des deux métriques les plus importantes, à savoir la validation_accuracy et la \
                 validation_loss, mais également d'une liste des noms et scores des 10 classes sur lesquelles le \
                 modèle est le moins performant, permettant de faciliter la prise de décision pour changer de modèle.

        6. **Déploiement** :
        Pour déployer le modèle, l'administrateur peut choisir parmi tous les modèles dans MLflow et simplement \
                 indiquer lequel passer en production.
        Le script chargé de l'inférence pour les utilisateurs charge en quelques secondes le nouveau modèle, \
                 sans interruption de service \
                 (seulement une attente si prédiction demandée en même temps que le changement de modèle).

        7. **Monitoring des performances** :
        Grâce à la matrice de confusion enregistrée avec chaque modèle, tous les jours, \
                 une nouvelle matrice est générée et comparée avec l'originale du modèle.
        Grâce à cela, il est possible de détecter un drift du modèle,\
                  soit une perte de performance sur certaines classes qu'il connaît à cause de nouvelles images qui \
                 peuvent être trop différentes de son entraînement.
        Un rapport est envoyé tous les jours avec les 10 meilleures et pires classes ainsi que les classes qui \
                 dérivent (positivement ou négativement, s'il y en a).

        8. **Gestion des conflits**
        Pour éviter des conflits entre les scripts de monitoring, de preprocessing et de training, \
                 chacun indique en permanence son état aux autres.
        Cela permet à l'un d'attendre que l'autre est fini pour se lancer \
                 et évite des erreurs ou corruption de données.
        """)

elif page == "Résultats de l'entraînement":
    st.title("Résultats et comparaison des modèles")

    # On affiche cette page seulement si l'utilisateur est connecté
    if 'admin_token' in st.session_state:
        try:
            headers = {
                "Authorization": f"Bearer {st.session_state.admin_token}",
                "api-key": "abcd1234"
            }

            # On récupère les résultats via l'API
            response = requests.get(f"{ADMIN_API_URL}/results", headers=headers, timeout=10)
            results = response.json()

            # On mets en page le JSON
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Dernier modèle entraîné :")
                st.markdown(f"""<p style="font-size: 18px; display: inline;">Identifiant du run MLFlow : </p> <p style="font-size: 18px; color: #0da0e4; display: inline;"> {results['latest_run_id']}</p>""", unsafe_allow_html = True)
                st.markdown(f"""<p style="font-size: 18px; display: inline;">Précision (sur le set de validation) : </p> <p style="font-size: 18px; color: #0da0e4; display: inline;"> {results['latest_run_val_accuracy']}</p>""", unsafe_allow_html = True)
                st.markdown(f"""<p style="font-size: 18px; display: inline;">Perte (sur le set de validation) : </p> <p style="font-size: 18px; color: #0da0e4; display: inline;"> {results['latest_run_val_loss']}</p>""", unsafe_allow_html = True)
                st.markdown(f"""<b style="font-size: 18px; display: inline;"> </br> Les 10 pires classes : </b>""", unsafe_allow_html = True)
                for classe, score in results['latest_run_worst_f1_scores'].items():
                    st.write(f"""<p style="font-size: 18px; display: inline;">{classe}</p> : <p style="font-size: 18px; color: #0da0e4; display: inline;">{score} </p>""", unsafe_allow_html = True)
            with col2:
                st.subheader("Modèle actuellement utilisé :")
                st.markdown(f"""<p style="font-size: 18px; display: inline;">Identifiant du run MLFlow : </p> <p style="font-size: 18px; color: #0da0e4; display: inline;"> {results['main_model_run_id']}</p>""", unsafe_allow_html = True)
                st.markdown(f"""<p style="font-size: 18px; display: inline;">Précision (sur le set de validation) : </p> <p style="font-size: 18px; color: #0da0e4; display: inline;"> {results['main_model_val_accuracy']}</p>""", unsafe_allow_html = True)
                st.markdown(f"""<p style="font-size: 18px; display: inline;">Perte (sur le set de validation) : </p> <p style="font-size: 18px; color: #0da0e4; display: inline;"> {results['main_model_val_loss']}</p>""", unsafe_allow_html = True)
                st.markdown(f"""<b style="font-size: 18px; display: inline;"> </br> Les 10 pires classes : </b>""", unsafe_allow_html = True)
                for classe, score in results['main_model_worst_f1_scores'].items():
                    st.write(f"""<p style="font-size: 18px; display: inline;">{classe}</p> : <p style="font-size: 18px; color: #0da0e4; display: inline;">{score} </p>""", unsafe_allow_html = True)

            # On ajoute un lien vers l'interface MLflow
            st.markdown("</br>[Accéder à plus de détails sur l'interface MLFlow](http://localhost:5200)", unsafe_allow_html = True)

        except Exception as e:
            st.error(f"Impossible de communiquer avec l'API administrateur : {str(e)}")
            st.info("Vérifiez que tous les conteneurs sont en cours d'exécution \
                    et que les ports sont correctement configurés.")
    else:
        st.warning("Veuillez vous connecter en tant qu'administrateur pour accéder aux résultats MLflow.")
        st.info("Allez dans l'onglet 'Interface utilisateur (APIs)' et connectez-vous en tant qu'admin.")

elif page == "Interface utilisateur (APIs)":
    st.title("Interface utilisateur (APIs)")

    # Permet de choisir à quelle API se connecter
    api_choice = st.radio("Choisissez une API", ("Utilisateur", "Admin"))

    if api_choice == "Utilisateur":
        st.subheader("API Utilisateur")

        # Système d'authentification
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        if st.button("Se connecter"):
            try:
                # On vérifie les indentifiants auprès de l'API
                response = requests.post(f"{USER_API_URL}/token",
                                         data={"username": username, "password": password})
                if response.status_code == 200:
                    st.success(f"Connecté en tant que {username}")
                    st.session_state.user_token = response.json()["access_token"]
                else:
                    st.error("Échec de l'authentification.")
            except Exception as e:
                st.error("Impossible de se connecter à l'API utilisateur. \
                         Assurez-vous que les conteneurs Docker sont en cours d'exécution.")

        # Permet d'afficher un message de succès après la contribution de l'utilisateur
        # lors d'une prédiction au moment du rechargement de la page
        if st.session_state.success:
            st.toast("Merci pour votre précieuse contribution !")
            st.session_state.success = False

        # Système de prédiction si l'utilisateur ou l'administrateur est bien connecté
        if 'user_token' in st.session_state:
            uploaded_file = st.file_uploader("Choisissez une image d'oiseau", type=["jpg", "png"])
            col1, col2, col3, col4 = st.columns([0.2, 0.2, 0.2, 0.22], vertical_alignment="bottom")
            # Si une image est chargée, on lance la prédiction
            if uploaded_file is not None:
                # On ouvre l'image pour l'afficher
                image = Image.open(uploaded_file)
                with col1:
                    st.image(image, caption="Image téléchargée")

                try:
                    files = {"file": ("image.jpg", uploaded_file.getvalue(), "image/jpeg")}
                    headers = {
                        "Authorization": f"Bearer {st.session_state.user_token}",
                        "api-key": "abcd1234"
                    }
                    # On fait une requête vers l'API
                    response = requests.post(f"{USER_API_URL}/predict", files=files, headers=headers)
                    
                    prediction = response.json()
                except Exception as e:
                    st.error(f"Impossible de communiquer avec l'API d'inférence : {response.json()}")

                try:
                    # On affiche dans les 3 colonnes restantes les prédictions ainsi que leurs
                    # images associées
                    with col2:
                        # Permet de récupérer l'image associée à la classe prédite
                        response = requests.get(f"{USER_API_URL}/get_class_image",
                                                params={'classe': prediction['predictions'][0]},
                                                headers=headers)
                        st.image(response.content)
                        st.markdown(f"""<p style="font-size: 20px;"> {prediction['predictions'][0]} </p>""",
                                    unsafe_allow_html=True)
                        st.markdown(f"""<p style="color: #079e20; font-size: 20px;"> \
                                    {str(round(prediction['scores'][0] * 100, 2)) + "%"} </p>""",
                                    unsafe_allow_html=True)
                    with col3:
                        # Permet de récupérer l'image associée à la classe prédite
                        response = requests.get(f"{USER_API_URL}/get_class_image",
                                                params={'classe': prediction['predictions'][1]},
                                                headers=headers)
                        st.image(response.content)
                        st.markdown(f"""<p style="font-size: 20px;"> {prediction['predictions'][1]} </p>""",
                                    unsafe_allow_html=True)
                        st.markdown(f"""<p style="color: #d1ae29; font-size: 20px;"> \
                                    {str(round(prediction['scores'][1] * 100, 2)) + "%"} </p>""",
                                    unsafe_allow_html=True)
                    with col4:
                        # Permet de récupérer l'image associée à la classe prédite
                        response = requests.get(f"{USER_API_URL}/get_class_image",
                                                params={'classe': prediction['predictions'][2]},
                                                headers=headers)
                        st.image(response.content)
                        st.markdown(f"""<p style="font-size: 20px;"> {prediction['predictions'][2]} </p>""",
                                    unsafe_allow_html=True)
                        st.markdown(f"""<p style="color: #b26a19; font-size: 20px;"> \
                                    {str(round(prediction['scores'][2] * 100, 2)) + "%"} </p>""",
                                    unsafe_allow_html=True)
                
                    # On affiche les boutons pour récupérer l'avis de l'utilisateur
                    st.subheader("Une des prédictions est-elle correcte ?")
                    col1, col2, col3 = st.columns([0.1, 0.4, 0.5])
                    with col1:
                        if st.button("Oui"):
                            st.session_state.specie = 1
                    with col2:
                        if st.button("Non, mais je connais l'espèce correcte"):
                            st.session_state.specie = 2

                    with col3:
                        # Si l'utilisateur n'est pas sûr, on demande à l'API d'ajouter l'image
                        # dans les classes inconnues
                        if st.button("Je ne suis pas sûr"):
                            data = {"species": "NA", "image_name": prediction['filename'], "is_unknown": True}
                            # On demande à l'API d'ajouter l'image
                            response = requests.post(f"{ADMIN_API_URL}/add_image", headers=headers, data=data)
                            st.session_state.success = True
                            st.session_state.specie = 0
                            st.rerun()

                except Exception as e:
                    st.error(f"Impossible de récupérer les images associées aux classes : {response.json()}")



                try:
                    # Si l'utilisateur dit Oui, on affiche la liste des espèces
                    if st.session_state.specie == 1 and not st.session_state.success:
                        st.session_state.selected_specie = st.selectbox(
                            "Sélectionnez l'espèce correcte :",
                            ["Sélectionnez une espèce..."] + prediction['predictions']
                        )

                    # Si l'utilisateur connaît la bonne espèce, on affiche la liste des 11 000 espèces
                    if st.session_state.specie == 2 and not st.session_state.success:
                        # On demande à l'API d'ajouter l'image
                        response = requests.get(f"{USER_API_URL}/get_species", headers=headers)
                        species_list = response.json()
                        st.session_state.selected_specie = st.selectbox(
                            "Sélectionnez l'espèce correcte :",
                            ["Sélectionnez une espèce..."] + species_list['species']
                        )

                except Exception as e:
                    st.error(f"Impossible de récupérer la liste des espèces...")
                
                # Si l'utilisateur a fait sa sélection, on demande à l'API d'ajouter l'image
                if st.session_state.selected_specie != "Sélectionnez une espèce...":
                    data = {"species": st.session_state.selected_specie,
                            "image_name": prediction['filename'],
                            "is_unknown": False
                            }
                    response = requests.post(f"{ADMIN_API_URL}/add_image", headers=headers, data=data)
                    st.session_state.success = True
                    st.session_state.specie = 0
                    st.session_state.selected_specie = "Sélectionnez une espèce..."
                    st.rerun()
                    
               

    else:
        st.subheader("API Admin")

        # Système d'authentification
        admin_username = st.text_input("Nom d'administrateur")
        admin_password = st.text_input("Mot de passe administrateur", type="password")

        if st.button("Se connecter (Admin)"):
            try:
                # On vérifie les indentifiants auprès de l'API
                response = requests.post(f"{ADMIN_API_URL}/token",
                                         data={"username": admin_username, "password": admin_password})
                if response.status_code == 200:
                    st.info(f"Connecté en tant qu'administrateur avec l'utilisateur : {admin_username}")
                    st.session_state.admin_token = response.json()["access_token"]
                else:
                    st.error(f"Impossible de se connecter à l'API administrateur, erreur {response.status_code}")
            except Exception as e:
                st.error("Impossible de se connecter à l'API administrateur. \
                         Assurez-vous que les conteneurs Docker sont en cours d'exécution.")

        # Si l'utilisateur est bien connecté, on affiche l'interface
        if 'admin_token' in st.session_state:

            # Permet d'ajouter un utilisateur
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
                    # On demande à l'API d'ajouter l'utilisateur
                    response = requests.post(f"{ADMIN_API_URL}/add_user", headers=headers, data=data)

                    st.info(response.json())
                    
                except Exception as e:
                    st.error(f"Impossible de communiquer avec l'API administrateur : {e}")

            st.write("---")

            # Lancer l'entraînement
            if st.button("Lancer l'entraînement"):
                try:
                    headers = {
                        "Authorization": f"Bearer {st.session_state.admin_token}",
                        "api-key": "abcd1234"
                    }
                    # On demande à l'API de lancer l'entraînement
                    response = requests.get(f"{ADMIN_API_URL}/train", headers=headers)

                    st.info(response.json())
                except Exception as e:
                    st.error(f"Impossible de communiquer avec l'API d'entraînement : {e}")


            st.write("---")

            st.write("Changer le modèle en production")
            run_id = st.text_input("Indiquez le run_id du modèle")
            if st.button("Valider"):
                try:
                    headers = {
                        "Authorization": f"Bearer {st.session_state.admin_token}",
                        "api-key": "abcd1234"
                    }
                    data = {"run_id": run_id}
                    # On demande à l'API de changer le modèle en production
                    response = requests.post(f"{ADMIN_API_URL}/switchmodel", headers=headers, data=data)
                    st.info(response.json())
                except Exception as e:
                    st.error(f"Impossible de communiquer avec l'API d'inférence : {e}")

elif page == "Conclusion":

    st.markdown("<h1 style='text-align: center;'>Projet MLOps - Reconnaissance d'oiseaux</h1>", unsafe_allow_html=True)

    # On charge et affiche l'image de couverture
    try:
        image = load_and_resize_image("oiseau_cover.jpg", 400)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Reconnaissance d'oiseaux", use_column_width=True)
    except FileNotFoundError:
        st.warning("Image de couverture non trouvée. \
                   Veuillez vous assurer que 'oiseau_cover.jpg' est présent dans le répertoire du script.")

    st.write("""
            ### Conclusion

            Possibilités d'améliorations :

            - **Traitement des images inconnues**
            - **Volume sauvegardé dans le cloud**
            - **Affichage de meilleures images à l'inférence**
            - **Applications Android et IOS**

            Pour aller plus loin :

            - **Reconaissance du chant des oiseaux**


            """)

# Pied de page
st.sidebar.markdown("---")
st.sidebar.info("Développé par :\n- Maxence REMY-HAROCHE\n- Guillaume RUIZ\n- Yoni EDERY")

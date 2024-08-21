# Projet de Reconnaissance d'Oiseaux avec MLOps

Ce projet implémente un système de reconnaissance d'oiseaux basé sur des images, avec une infrastructure MLOps complète pour la gestion des données, l'entraînement, le déploiement et le monitoring du modèle.

## Structure du Projet

- [Application](./app/) Modules principaux de l'application
    - [Models](./app/models/) Classe dédiée à la prédiction
    - [Utils](./app/utils/) Scripts utilitaires de l'application
- [Data](./data/) Données relatives au dataset d'entrainement du modèle
- [Documentation](./docs/) Documentation détaillée du projet
- [Logs](./logs/) Historique des logs générés pendant l'exécution des différents scripts
- [MLRuns](./mlruns/) Historique des runs MLflow
- [Models](./models/) Historique des modèles
- [Monitoring](./monitoring/) Scripts pour le monitoring et la détection de drift
- [Preprocessing](./preprocessing/) Scripts de traitement des données
- [Temp images](./tempImage/) Stockage temporairement des images dans le cadre du fonctionnement de l'API et du modèle
- [Scripts](./scripts/) Scripts dédiés à la pipeline
- [Tests](./tests/) Tests unitaires et d'intégration
    - [Tests d'intégration](./tests/integration/) Tests d'intégration
    - [Tests unitaires](./tests/unit/) Tests unitaires
- [Training](./training/) Scripts pour l'entraînement du modèle

## Installation

1. Clonez ce repository
2. Installez les dépendances : `pip install -r requirements.txt`
3. Ajouter à la racine du répértoire du projet le fichier de configuration Kaggle (kaggle.json) : Connectez-vous à votre compte kaggle.com puis "Settings > API > Create New Token"

## Utilisation

- Pour exécuter la pipeline complète : `python scripts/pipeline.py`
    - Pour télécharger le dataset : `python scripts/downloadDataset.py`
    - Pour effectuer le traitement obligatoire des données : `python preprocessing/preprocess_dataset.py`
    - Pour entraîner le modèle : `python training/train_model.py`
- Pour lancer l'interface Streamlit : `streamlit run streamlit_app.py`

## Documentation

Pour une documentation plus détaillée, veuillez consulter le dossier `docs/`.

## Contribution

Les pull requests sont les bienvenues. Pour des changements majeurs, veuillez d'abord ouvrir une issue pour discuter de ce que vous aimeriez changer.
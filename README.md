# Projet de Reconnaissance d'Oiseaux avec MLOps

Ce projet implémente un système de reconnaissance d'oiseaux basé sur des images, avec une infrastructure MLOps complète pour la gestion des données, l'entraînement, le déploiement et le monitoring du modèle.

## Structure du Projet

- [Application](./app/README.md) Modules principaux de l'application
    - [Models](./app/models/README.md) Classe dédiée à la prédiction
    - [Utils](./app/utils/README.md) Scripts utilitaires de l'application
- [Data](./data/README.md) Données relatives au dataset d'entrainement du modèle
- [Documentation](./docs/README.md) Documentation détaillée du projet
- [Inference](./inference/README.md) Fonction de prédiction utilisée dans les phases de tests
- [Logs](./logs/README.md) Historique des logs générés pendant l'exécution des différents scripts
- [Mlruns](./mlruns/README.md) Historique des runs MLflow
- [Models](./models/README.md) Historique des modèles
- [Monitoring](./monitoring/README.md) Scripts pour le monitoring et la détection de drift
- [Preprocessing](./preprocessing/README.md) Scripts de traitement des données
- [Temp images](./tempImage/README.md) Stockage temporairement des images dans le cadre du fonctionnement de l'API et du modèle
- [Scripts](./scripts/README.md) Scripts dédiés à la pipeline
- [Tests](./tests/README.md) Tests unitaires et d'intégration
    - [Tests d'intégration](./tests/integration/README.md) Tests d'intégration
    - [Tests unitaires](./tests/unit/README.md) Tests unitaires
- [Training](./training/README.md) Scripts pour l'entraînement du modèle

## Installation

1. Clonez ce repository
2. Installez les dépendances : `pip install -r requirements.txt`
3. Ajouter à la racine du répértoire du projet le fichier de configuration Kaggle (kaggle.json) : Connectez-vous à votre compte kaggle.com puis "Settings > API > Create New Token"

## Utilisation

- Pour télécharger le dataset : `python scripts/downloadDataset.py`
- Pour effectuer le traitement obligatoire des données : `python preprocessing/preprocess_dataset.py`
- Pour entraîner le modèle : `python training/train_model.py`
- Pour lancer l'interface Streamlit : `streamlit run streamlit_app.py`
- Pour exécuter la pipeline complète : `python scripts/pipeline.py`

## Documentation

Pour une documentation plus détaillée, veuillez consulter le dossier `docs/`.

## Contribution

Les pull requests sont les bienvenues. Pour des changements majeurs, veuillez d'abord ouvrir une issue pour discuter de ce que vous aimeriez changer.
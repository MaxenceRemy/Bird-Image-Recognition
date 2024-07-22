# Projet de Reconnaissance d'Oiseaux avec MLOps

Ce projet implémente un système de reconnaissance d'oiseaux basé sur des images, avec une infrastructure MLOps complète pour l'entraînement, le déploiement et le monitoring du modèle.

## Structure du Projet

- `app/`: Contient les modules principaux de l'application
- `monitoring/`: Scripts pour le monitoring et la détection de drift
- `scripts/`: Scripts utilitaires et de test
- `training/`: Scripts pour l'entraînement du modèle
- `docs/`: Documentation détaillée du projet

## Installation

1. Clonez ce repository
2. Installez les dépendances : `pip install -r requirements.txt`

## Utilisation

- Pour entraîner le modèle : `python training/train_model.py`
- Pour exécuter la pipeline complète : `python scripts/pipeline.py`
- Pour lancer l'interface Streamlit : `streamlit run streamlit_app.py`

## Documentation

Pour une documentation plus détaillée, veuillez consulter le dossier `docs/`.

## Contribution

Les pull requests sont les bienvenues. Pour des changements majeurs, veuillez d'abord ouvrir une issue pour discuter de ce que vous aimeriez changer.
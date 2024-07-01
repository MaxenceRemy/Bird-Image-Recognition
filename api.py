from fastapi import FastAPI, HTTPException, Header, File, UploadFile
from typing import List
import json

app = FastAPI(
    title="Reconnaissance des oiseaux",
    description="API pour identifier l'espèce d'un oiseau à partir d'une photo.",
    version="0.1"
)

@app.get("/", summary="Page d'accueil")
def get_status():
    """
    Permet de faire une simple requête vers l’API afin de vérifier que l’on obtient bien un status code “200”. Elle retourne l’erreur 503 si le service n’est pas démarré
    """
    # Vous pouvez vérifier l'état de votre service ici
    response = {"status": "Service is running"}
    return response

@app.post("/predict", summary="Faire une prédiction")
async def get_prediction(file: UploadFile = File(...)):
    """
    Permet de faire une prédiction de classe à partir d’une image envoyée à l’API.
    Format de l’entrée : .jpg, .jpeg, .png
    Format de réponse : JSON, contenant le score et la classe + image d’exemple de la classe prédite
    """
    # Vous pouvez charger votre modèle et faire une prédiction ici
    # N'oubliez pas de prétraiter l'image avant de faire une prédiction
    response = {"prediction": "Classe prédite", "score": "Score de prédiction"}
    return response

@app.post("/add_image", summary="Ajouter une image")
async def add_image(file: UploadFile = File(...), label: str = Header(None)):
    """
    Permet d’ajouter une image dans la base de données et la classe déjà existante associée
    Format de l’entrée: .jpg, .jpeg, .png + label
    """
    # Vous pouvez ajouter l'image à votre base de données ici
    response = {"status": "Image ajoutée avec succès"}
    return response

@app.get("/get_species", summary="Obtenir la liste des espèces")
def get_species():
    """
    Permet d’obtenir la liste des espèces d’oiseaux 
    Format de réponse : JSON, contenant la liste des espèces
    """
    # Vous pouvez obtenir la liste des espèces de votre base de données ici
    response = {"species": ["Espèce 1", "Espèce 2", "Espèce 3"]}
    return response

from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile, Header, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
import jwt
import os
import json
from dotenv import load_dotenv
import logging
# from app.models.predictClass import predictClass
from fastapi.responses import FileResponse
import requests
import hashlib
import time
import pandas as pd

# Charger les variables d'environnement
load_dotenv()

volume_path = 'volume_data'

dataset_clean_path = os.path.join(volume_path, 'dataset_clean')
dataset_raw_path = os.path.join(volume_path, 'dataset_raw')
log_folder = os.path.join(volume_path, "logs")
os.makedirs(log_folder, exist_ok = True)
logging.basicConfig(filename=os.path.join(log_folder, "user_api.log"), level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s', 
                    datefmt='%d/%m/%Y %I:%M:%S %p')



# Variables d'environnement
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ADMIN_PASSWORD = os.getenv("USER_PASSWORD")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

users_folder = os.path.join(volume_path, 'authorized_users')
users_path = os.path.join(users_folder, 'authorized_users.json')
if not os.path.exists(users_path):
    time.sleep(5) # on attends que le container d'API ajoute les utilisateurs

# Charger les utilisateurs autorisés depuis le fichier JSON
def load_authorized_users():
    with open(users_path, 'r') as f:
        return json.load(f)
    
app = FastAPI(
    title="Reconnaissance des oiseaux",
    description="API pour identifier l'espèce d'un oiseau à partir d'une photo.",
    version="0.1"
)

# on précharge Tensorflow et Cudnn (pour Nvidia) en important la classe et en faisant l'inférence d'une image
# classifier = predictClass()
# temp_image_path ='7.jpg'
# classifier.predict(temp_image_path)

# Modèle Pydantic pour le token
class Token(BaseModel):
    access_token: str
    token_type: str

# Fonction pour créer un token JWT
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Fonction pour vérifier le token JWT
def verify_token(token: str = Depends(OAuth2PasswordBearer(tokenUrl="/token"))):
    try:
        logging.info(f"Tentative de décodage du token: {token}")
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None or username not in load_authorized_users():
            logging.warning("Le token ne contient pas de 'sub' valide")
            raise HTTPException(status_code=401, detail="Could not validate credentials")
        logging.info(f"Token validé pour l'utilisateur: {username}")
        return username
    except jwt.PyJWTError as e:
        logging.error(f"Erreur lors de la validation du token: {str(e)}")
        raise HTTPException(status_code=401, detail="Could not validate credentials")

# Fonction pour vérifier la clé API
def verify_api_key(api_key: str = Header(..., alias="api-key")):
    if api_key != API_KEY:
        logging.warning("Tentative d'accès avec une clé API invalide")
        raise HTTPException(status_code=403, detail="Invalid API Key")
    logging.info("Clé API validée")
    return api_key


# Route pour obtenir un token
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = load_authorized_users().get(form_data.username)
    if user is None or form_data.password != user[1]:
        logging.warning(f"Tentative de connexion échouée pour l'utilisateur: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username}, expires_delta=access_token_expires
    )
    logging.info(f"Connexion réussie pour l'utilisateur: {form_data.username}")
    return {"access_token": access_token, "token_type": "bearer"}

# Route racine
@app.get("/")
async def root(api_key: str = Depends(verify_api_key), username: str = Depends(verify_token)):
    return {"message": "Bienvenue sur l'API de reconnaissance d'oiseaux", "user": username}

# Route pour faire une prédiction
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key),
    username: str = Depends(verify_token)
):
    logging.info(f"Prédiction demandée par l'utilisateur: {username}")
    try:
        content = await file.read()
        file_name = hashlib.sha256(content).hexdigest() + ".jpg"
        folder_path = os.path.join(volume_path, 'temp_images')
        os.makedirs(folder_path, exist_ok = True)
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "wb") as image_file:
            image_file.write(content)

        response = requests.get(f'http://inference:5500/predict', params={'file_name': file_name})
        return response.json()
    
    except Exception as e:
        logging.error(f"Erreur lors de la prédiction: {str(e)}")
        logging.exception("Traceback complet:")
        raise HTTPException(status_code=500, detail=str(e))


# Route pour obtenir la liste des espèces
@app.get("/get_species")
async def get_species(api_key: str = Depends(verify_api_key), username: str = Depends(verify_token)):
    df = pd.read_csv(os.path.join(dataset_raw_path, 'birds_list.csv'))
    species_list = sorted(df["English"].tolist())
    return {"species": species_list}

# Route pour télécharger une image
@app.get("/get_class_image")
async def get_class_image(
    classe: str,
    api_key: str = Depends(verify_api_key),
    username: str = Depends(verify_token)
):
    dossier_classe = os.path.join(dataset_clean_path, 'test', classe)
    for name in os.listdir(dossier_classe):
        image_path = os.path.join(dossier_classe, name)
        return FileResponse(image_path, media_type='image/jpeg', filename=f"{classe}_image.jpg")
    raise HTTPException(status_code=404, detail="Image not found")
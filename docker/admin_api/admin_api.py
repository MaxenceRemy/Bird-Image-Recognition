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
import requests
import shutil

# Charger les variables d'environnement
load_dotenv()

volume_path = 'volume_data'

dataset_raw_path = os.path.join(volume_path, 'dataset_raw')
unknown_images_path = os.path.join(volume_path, 'unknown_images')
os.makedirs(unknown_images_path, exist_ok = True)
log_folder = os.path.join(volume_path, "logs")
os.makedirs(log_folder, exist_ok = True)
logging.basicConfig(filename=os.path.join(log_folder, "admin_api.log"), level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s', 
                    datefmt='%d/%m/%Y %I:%M:%S %p')

# Le code suivant permet d'ajouter une liste par défaut contenant les admins autorisés
# seulement si il n'y a pas déjà un tel fichier sur le volume
users_folder = os.path.join(volume_path, 'authorized_users')
users_path = os.path.join(users_folder, 'authorized_users.json')
if os.path.exists(users_path):
    os.remove('authorized_users.json')
else:
    os.makedirs(users_folder, exist_ok = True)
    shutil.copy('authorized_users.json', os.path.join(users_folder, 'authorized_users.json'))

# Variables d'environnement
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Charger les utilisateurs autorisés depuis le fichier JSON
def load_authorized_users():
    with open(users_path, 'r') as f:
        return json.load(f)

AUTHORIZED_USERS = load_authorized_users()

app = FastAPI(
    title="Reconnaissance des oiseaux",
    description="API pour identifier l'espèce d'un oiseau à partir d'une photo.",
    version="0.1"
)

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
        user = AUTHORIZED_USERS.get(username)
        if user is None or not user[0]:
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

# Fonction pour mettre à jour le fichier JSON des utilisateurs autorisés
def update_authorized_users(users):
    with open(users_path, 'w') as f:
        json.dump(users, f, indent=4)

# Route pour obtenir un token
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = AUTHORIZED_USERS.get(form_data.username)
    if user is None or not user[0] or form_data.password != user[1]:
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


# Route pour ajouter une image
@app.post("/add_image")
async def add_image(
    species: str = Form(...),
    image_name: str = Form(...),
    is_unknown: bool = Form(False),
    api_key: str = Depends(verify_api_key),
    username: str = Depends(verify_token)
):
    try:
        temp_folder = os.path.join(volume_path, 'temp_images')
        file_path = os.path.join(temp_folder, image_name)
        if is_unknown:
            os.rename(file_path, f"{unknown_images_path}/{image_name}")
            return {"status": "Image added to unknow images folder"}
            
        else:
            class_path = os.path.join(dataset_raw_path, f"train/{species}")
            if not os.path.exists(class_path):
                os.makedirs(class_path, exist_ok = True)
            os.rename(file_path, f"{class_path}/{image_name}")
            return {"status": f"Image added to existing species '{species}'"}
    except Exception as e:
        logging.error(f"Error adding image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))




# Nouvelle route pour ajouter un utilisateur
@app.post("/add_user")
async def add_user(
    new_username: str = Form(...),
    user_password: str = Form(...),
    is_admin: bool = Form(False),
    api_key: str = Depends(verify_api_key),
    current_user: str = Depends(verify_token)
):
    global AUTHORIZED_USERS
    if new_username in AUTHORIZED_USERS:
        raise HTTPException(status_code=400, detail="User already exists")
    
    if is_admin:
        AUTHORIZED_USERS[new_username] = [True, user_password]
    else:
        AUTHORIZED_USERS[new_username] = [False, user_password]
    update_authorized_users(AUTHORIZED_USERS)
    
    logging.info(f"Nouvel utilisateur ajouté par {current_user}: {new_username}")
    return {"status": "User added successfully"}

# Route pour obtenir la liste des utilisateurs autorisés
@app.get("/get_users")
async def get_users(
    api_key: str = Depends(verify_api_key),
    current_user: str = Depends(verify_token)
):
    return {"authorized_users": AUTHORIZED_USERS}


@app.get("/train")
async def train(
    api_key: str = Depends(verify_api_key),
    current_user: str = Depends(verify_token)
):
        
    try:
        response = requests.get(f'http://training:5500/train')
        return response.json()
    
    except requests.RequestException as e:
        logging.error(f'Failed to communicate with the training container: {e}')
        raise HTTPException(status_code=500, detail="Internal server error")
    
@app.post("/switchmodel")
async def switch_model(
    run_id: str = Form(...),
    api_key: str = Depends(verify_api_key),
    current_user: str = Depends(verify_token)
):
    try:
        response = requests.post('http://inference:5500/switchmodel', data = {'run_id': run_id})
        return response.json()
    
    except requests.RequestException as e:
        logging.error(f'Failed to communicate with the inference container: {e}')
        raise HTTPException(status_code=500, detail="Internal server error")
    
@app.get("/results")
async def results(
    api_key: str = Depends(verify_api_key),
    current_user: str = Depends(verify_token)
):
    logging.info(f"Requête /results reçue de l'utilisateur: {current_user}")
    try:
        logging.info("Tentative de communication avec le conteneur training")
        response = requests.get(f'http://training:5500/results', timeout=10)
        logging.info(f"Réponse reçue du conteneur training: status={response.status_code}")
        if response.status_code == 200:
            results = response.json()
            logging.info("Résultats récupérés avec succès")
            return results
        else:
            logging.error(f"Erreur lors de la récupération des résultats: {response.text}")
            raise HTTPException(status_code=response.status_code, detail=response.text)
    except requests.RequestException as e:
        logging.error(f'Erreur de communication avec le conteneur training: {str(e)}')
        raise HTTPException(status_code=500, detail=f"Erreur de communication avec le conteneur training: {str(e)}")
    except Exception as e:
        logging.error(f'Erreur inattendue: {str(e)}')
        raise HTTPException(status_code=500, detail=f"Erreur inattendue: {str(e)}")
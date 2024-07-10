from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile, Header
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
import jwt
import os
from dotenv import load_dotenv
import logging
from predictClass import predictClass
from fastapi.responses import FileResponse

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(filename='api.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Variables d'environnement
API_KEY = os.getenv("API_KEY")
API_USERNAME = os.getenv("API_USERNAME")
API_PASSWORD = os.getenv("API_PASSWORD")
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI(
    title="Reconnaissance des oiseaux",
    description="API pour identifier l'espèce d'un oiseau à partir d'une photo.",
    version="0.1"
)

# on précharge Tensorflow et Cudnn (pour Nvidia) en important la classe et en faisant l'inférence d'une image
classifier = predictClass()
temp_image_path = os.path.join("../data/test", 'FAIRY BLUEBIRD', '7.jpg')
classifier.predict(temp_image_path)

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
        if username is None:
            logging.warning("Le token ne contient pas de 'sub'")
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
    if form_data.username != API_USERNAME or form_data.password != API_PASSWORD:
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
        # Créer le dossier tempImage s'il n'existe pas
        os.makedirs("tempImage", exist_ok=True)
        
        image_path = "tempImage/image.png"
        logging.info(f"Sauvegarde de l'image à: {image_path}")
        with open(image_path, "wb") as image_file:
            content = await file.read()
            image_file.write(content)
        
        logging.info("Début de la prédiction")
        meilleure_classe, highest_score = classifier.predict(image_path)
        logging.info(f"Prédiction terminée: {meilleure_classe}, score: {highest_score}")
        return {"prediction": meilleure_classe, "score": highest_score}
    except Exception as e:
        logging.error(f"Erreur lors de la prédiction: {str(e)}")
        logging.exception("Traceback complet:")
        raise HTTPException(status_code=500, detail=str(e))

# Route pour ajouter une image
@app.post("/add_image")
async def add_image(
    file: UploadFile = File(...),
    label: str = Header(None),
    api_key: str = Depends(verify_api_key),
    username: str = Depends(verify_token)
):
    # Implémentez ici la logique pour ajouter l'image
    return {"status": "Image ajoutée avec succès"}

# Route pour obtenir la liste des espèces
@app.get("/get_species")
async def get_species(api_key: str = Depends(verify_api_key), username: str = Depends(verify_token)):
    # Implémentez ici la logique pour obtenir la liste des espèces
    return {"species": ["Espèce 1", "Espèce 2", "Espèce 3"]}

# Route pour télécharger une image
@app.get("/get_class_image")
async def get_class_image(
    classe: str,
    api_key: str = Depends(verify_api_key),
    username: str = Depends(verify_token)
):
    dossier_classe = os.path.join("../data/test", classe)
    for name in os.listdir(dossier_classe):
        image_path = os.path.join(dossier_classe, name)
        return FileResponse(image_path, media_type='image/jpeg', filename=f"{classe}_image.jpg")
    raise HTTPException(status_code=404, detail="Image not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

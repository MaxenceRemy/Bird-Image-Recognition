from fastapi import FastAPI, HTTPException, Header, File, UploadFile, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials, OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.openapi.models import OAuthFlows as OAuthFlowsModel
from fastapi.security.oauth2 import OAuth2
from fastapi.security.api_key import APIKeyHeader
from typing import List
import json
import secrets

app = FastAPI(
    title="Reconnaissance des oiseaux",
    description="API pour identifier l'espèce d'un oiseau à partir d'une photo.",
    version="0.1"
)

# Authentification de base
security = HTTPBasic()

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, "username")
    correct_password = secrets.compare_digest(credentials.password, "password")
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# Clés API
API_KEY = "1234567asdfgh"
API_KEY_NAME = "access_token"
api_key_header = APIKeyHeader(name=API_KEY_NAME)

def verify_api_key(api_key_header: str = Depends(api_key_header)):
    if api_key_header != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return api_key_header

# Tokens Web JSON (JWT)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def verify_token(token: str = Depends(oauth2_scheme)):
    if token != "myfaketoken":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid JWT Token"
        )
    return token

# OAuth
class OAuth2PasswordBearerWithCookie(OAuth2):
    def __init__(
        self,
        tokenUrl: str,
        scheme_name: str = None,
        scopes: dict = None,
        auto_error: bool = True,
    ):
        if not scopes:
            scopes = {}
        flows = OAuthFlowsModel(password={"tokenUrl": tokenUrl, "scopes": scopes})
        super().__init__(flows=flows, scheme_name=scheme_name, auto_error=auto_error)

oauth2_scheme = OAuth2PasswordBearerWithCookie(tokenUrl="/token")

@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username != "alice" or form_data.password != "wonderland":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"access_token": "myfaketoken", "token_type": "bearer"}


# Pour utiliser HTTPS, on doit configurer un serveur HTTPS pour notre application FastAPI.
# on peut le faire en utilisant un serveur comme Uvicorn avec un certificat SSL.

@app.get("/", summary="Page d'accueil")
def get_status(username: str = Depends(verify_credentials), api_key: str = Depends(verify_api_key), token: str = Depends(verify_token)):
    response = {"status": "Service is running"}
    return response

@app.post("/predict", summary="Faire une prédiction")
async def get_prediction(file: UploadFile = File(...), username: str = Depends(verify_credentials), api_key: str = Depends(verify_api_key), token: str = Depends(verify_token)):
    response = {"prediction": "Classe prédite", "score": "Score de prédiction"}
    return response

@app.post("/add_image", summary="Ajouter une image")
async def add_image(file: UploadFile = File(...), label: str = Header(None), username: str = Depends(verify_credentials), api_key: str = Depends(verify_api_key), token: str = Depends(verify_token)):
    response = {"status": "Image ajoutée avec succès"}
    return response

@app.get("/get_species", summary="Obtenir la liste des espèces")
def get_species(username: str = Depends(verify_credentials), api_key: str = Depends(verify_api_key), token: str = Depends(verify_token)):
    response = {"species": ["Espèce 1", "Espèce 2", "Espèce 3"]}
    return response

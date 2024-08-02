import os
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import shutil

def get_dataset(dataset, destination):

    # Initialisation de l'API de Kaggle
    api = KaggleApi()
    api.authenticate() # il faut régler les variables d'authentification (voir documentation Kaggle)
    
    # On télécharge le fichier
    api.dataset_download_files(dataset, path = destination, unzip = True)
    print("Fichier téléchargé et extrait!")

    # On supprime les fichiers temporaires
    os.remove(os.path.join(destination, "birds.csv"))
    os.remove(os.path.join(destination, "EfficientNetB0-525-(224 X 224)- 98.97.h5"))


dataset = "gpiosenka/100-bird-species"
destination = "./data/dataset_raw"
get_dataset(dataset, destination)

import os
import random
import shutil
import time
import logging
import csv
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS
from UnderSampling import UnderSamplerImages
from SizeManager import SizeManager
import os
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import shutil
import smtplib, ssl
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks
import requests
import json
import schedule
from CleanDB import CleanDB
from DatasetCorrection import DatasetCorrection


volume_path = 'volume_data'
dataset_raw_path = os.path.join(volume_path, 'dataset_raw')
dataset_clean_path = os.path.join(volume_path, 'dataset_clean')
dataset_version_path = os.path.join(dataset_raw_path, 'dataset_version.json')
classes_tracking_path = os.path.join(dataset_raw_path, 'classes_tracking.json')

os.makedirs(dataset_raw_path, exist_ok = True)
os.makedirs(dataset_clean_path, exist_ok = True)

state_folder = os.path.join(volume_path, "containers_state")
os.makedirs(state_folder, exist_ok = True)
state_path  = os.path.join(state_folder, "preprocessing_state.txt")
training_state_path  = os.path.join(state_folder, "training_state.txt")
with open(state_path, "w") as file:
    file.write("0")
log_folder = os.path.join(volume_path, "logs")
os.makedirs(log_folder, exist_ok = True)
logging.basicConfig(filename=os.path.join(log_folder, "preprocessing.log"), level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s', 
                    datefmt='%d/%m/%Y %I:%M:%S %p')
training_logs_file = os.path.join(log_folder, "training_results.txt")

def save_json(filepath, dict):
    with open(filepath, 'w') as file:
        json.dump(dict, file, indent=4)

def get_new_classes(base_dict):
    updated_list = os.listdir(os.path.join(dataset_raw_path, 'train'))
    new_classes_to_track = []
        
    for classe in updated_list:
        if classe not in base_dict['originals'] and classe not in base_dict['new']:
            new_classes_to_track.append(classe)
    return new_classes_to_track

def start_cleaning(base_dict, updated_dict):

    with open(training_state_path, "r") as file:
        state = file.read()
    while state == "1":
        with open(training_state_path, "r") as file:
            state = file.read()
        time.sleep(5)

    with open(state_path, "w") as file:
        file.write("1")

    cleanDB = CleanDB(dataset_clean_path, treshold = False)

    shutil.rmtree(dataset_clean_path)
    os.makedirs(dataset_clean_path)
    time.sleep(5)
    shutil.copytree(dataset_raw_path, dataset_clean_path, dirs_exist_ok = True)
    logging.info(f"Copie des fichiers terminée !")
    os.remove(os.path.join(dataset_clean_path, 'dataset_version.json'))
    os.remove(os.path.join(dataset_clean_path, 'classes_tracking.json'))
    os.remove(os.path.join(dataset_clean_path, 'birds.csv'))
    os.remove(os.path.join(dataset_clean_path, 'birds_list.csv'))
    new_classes_to_track = get_new_classes(base_dict)
    if new_classes_to_track:
        for classe in new_classes_to_track:
            shutil.rmtree(os.path.join(dataset_clean_path, 'train', classe))

    logging.info(f"Lancement du nettoyage des données !")
    cleanDB.cleanAll()
    logging.info(f"Nettoyage des données terminé !")
    time.sleep(5)
    with open(state_path, "w") as file:
        file.write("0")

def auto_update_dataset(dataset_name, destination, base_dict = False, updated_dict = False, first_launch = False):
     
     # Initialisation de l'API de Kaggle
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()

    datasets = kaggle_api.dataset_list(search=dataset_name, sort_by='hottest') # Renvoie la liste des datasets qui correspondent aux critères de recherche
    for dataset in datasets:
        if dataset.ref == dataset_name: # Pour le bon dataset
            online_dataset_version = dataset.lastUpdated # On récupère la version du dataset sur Kaggle       
            dataset_info = { # On sauvegarde les informations pour créer notre fichier dataset_version.json
            'dataset_name': dataset.ref,
            'last_updated': online_dataset_version.strftime("%Y-%m-%d %H:%M:%S")
            }

    if first_launch == False:

        if os.path.exists(dataset_version_path):
            with open(dataset_version_path, 'r') as file:
                current_dataset_version = datetime.strptime(json.load(file).get('last_updated'), "%Y-%m-%d %H:%M:%S") # On récupère la version de notre dataset
                if current_dataset_version >= online_dataset_version:
                    logging.info(f"Le dataset est à jour, on ne le télécharge pas.")
                    print(f"Le dataset est à jour, on ne le télécharge pas.")
                    return
                else:
                    logging.info(f"La version du dataset nécéssite une mise à jour.")
                    print(f"La version du dataset nécéssite une mise à jour.")

    # On télécharge le fichier
    temp_destination = os.path.join(destination, 'temp')
    kaggle_api.dataset_download_files(dataset_name, path = temp_destination, unzip = True)
    with open(dataset_version_path, 'w') as file:
        json.dump(dataset_info, file, indent=4)

    # On supprime les fichiers temporaires
    # os.remove(os.path.join(destination, "birds.csv"))
    os.remove(os.path.join(temp_destination, "EfficientNetB0-525-(224 X 224)- 98.97.h5"))

    """
    Correction des incoherences des données, et création du fichier CSV de modèle d'espèces
    """
    datasetCorrection = DatasetCorrection(db_to_clean = temp_destination, test_mode = False)
    datasetCorrection.full_correction()

    shutil.copytree(temp_destination, destination, dirs_exist_ok = True)
    shutil.rmtree(temp_destination)

    # on créer ou charge le fichier de tracking du dataset
    if not os.path.exists(classes_tracking_path):

        logging.info("Création du fichier de tracking du dataset car aucun n'est présent.")

        classes_tracking_base = {
            'originals': os.listdir(os.path.join(dataset_raw_path, 'train')),
            'originals_count': [],
            'new': [],
            'new_count': []
        }

        for folder in os.listdir(os.path.join(dataset_raw_path, 'train')):
            classes_tracking_base['originals_count'].append(len(os.listdir(os.path.join(dataset_raw_path, 'train', folder))))

        save_json(classes_tracking_path, classes_tracking_base)

        classes_tracking_updated = classes_tracking_base.copy()

    if base_dict == False or updated_dict == False:
        start_cleaning(classes_tracking_base, classes_tracking_updated)
    else:
        start_cleaning(base_dict, updated_dict)



if not os.path.exists(dataset_version_path):
    logging.info("Téléchargement et preprocessing du dataset car aucun n'est présent.")
    auto_update_dataset(dataset_name='gpiosenka/100-bird-species', destination=dataset_raw_path, first_launch=True)
    

with open(classes_tracking_path, 'r') as file:
        classes_tracking_base = json.load(file)
        logging.info("Chargement des données de tracking du dataset")

classes_tracking_updated = classes_tracking_base.copy()

schedule.every().day.at("02:00").do(auto_update_dataset, 'gpiosenka/100-bird-species', dataset_raw_path, classes_tracking_base, classes_tracking_updated)

while True:
    
    classes_tracking_updated['originals_count'] = []
    for classe in classes_tracking_updated['originals']:
        classes_tracking_updated['originals_count'].append(len(os.listdir(os.path.join(dataset_raw_path, 'train', classe))))

    classes_tracking_updated['new_count'] = []
    for classe in classes_tracking_updated['new']:
        classes_tracking_updated['new_count'].append(len(os.listdir(os.path.join(dataset_raw_path, 'train', classe))))

    original_classes_added_images = sum(classes_tracking_updated['originals_count']) - sum(classes_tracking_base['originals_count'])
    new_classes_added_images = sum(classes_tracking_updated['new_count']) - sum(classes_tracking_base['new_count'])
    total_added_images = original_classes_added_images + new_classes_added_images
    
    minimum_nbr_images = 0.01 * (sum(classes_tracking_base['originals_count']) + sum(classes_tracking_base['new_count']))
    if total_added_images > minimum_nbr_images:
        logging.info(f"{total_added_images} nouvelles images ajoutées, lancement du cleaning")
        start_cleaning(classes_tracking_base, classes_tracking_updated)
        classes_tracking_base = classes_tracking_updated.copy()
        save_json(classes_tracking_path, classes_tracking_base)
        time.sleep(5)

    new_classes_to_track = get_new_classes(classes_tracking_base)

    for classe in new_classes_to_track:
        nbr_images = len(os.listdir(os.path.join(dataset_raw_path, 'train', classe)))
        if nbr_images >= min(classes_tracking_base['originals_count']):
            print(f"Nouvelle classe nommée {classe} à ajouter !")
            classes_tracking_updated['new'].append(classe)
            classes_tracking_updated['new_count'].append(len(os.listdir(os.path.join(dataset_raw_path, 'train', classe))))
            logging.info("Nouvelle classe ajoutée, lancement du cleaning")
            start_cleaning(classes_tracking_base, classes_tracking_updated)
            classes_tracking_base = classes_tracking_updated.copy()
            save_json(classes_tracking_path, classes_tracking_base)
            time.sleep(5)

    schedule.run_pending()
    time.sleep(1)
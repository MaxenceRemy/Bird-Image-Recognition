import os
import numpy as np
from fastapi import FastAPI, HTTPException, Body
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
import logging
import tensorflow as tf
import time
import json

# On lance le serveur FastAPI
app = FastAPI()

# On créer les différents chemins
volume_path = 'volume_data'
log_folder = os.path.join(volume_path, "logs")
mlruns_path = os.path.join(volume_path, 'mlruns')

# On créer le dossier si nécessaire
os.makedirs(log_folder, exist_ok = True)

# On configure le logging pour les informations et les erreurs
logging.basicConfig(filename=os.path.join(log_folder, "inference.log"), level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s', 
                    datefmt='%d/%m/%Y %I:%M:%S %p')

# Cette variable s'incrémente dès que le temps d'inférence est trop long
too_long_inference = 0
    
class predictClass:
    def __init__(self, model_path, img_size=(224, 224)):
        self.img_size = img_size
        self.model_path = model_path
        
        # Configurer GPU si disponible
        self.configure_gpu()
        
        try:
            self.model = load_model(os.path.join(model_path, 'saved_model.h5'))
            with open(os.path.join(model_path, 'classes.json'), 'r') as file:
                self.class_names = json.load(file)
            # self.num_classes = len(self.class_names)
            logging.info("Modèle chargé avec succès.")
        except Exception as e:
            logging.error(f"Erreur lors de l'initialisation : {str(e)}")
            raise

    def configure_gpu(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"GPU(s) configuré(s) pour une utilisation dynamique de la mémoire.")
            except RuntimeError as e:
                logging.error(f"Erreur lors de la configuration du GPU : {e}")

    def predict(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"L'image {image_path} n'existe pas.")
        
        try:
            img = image.load_img(image_path, target_size=self.img_size)
            img_array = image.img_to_array(img)
            img_array_expanded_dims = np.expand_dims(img_array, axis=0)
            img_ready = preprocess_input(img_array_expanded_dims)
            
            prediction = self.model.predict(img_ready)
            
            # highest_score_index = int(np.argmax(prediction))
            # meilleure_classe = self.class_names[str(highest_score_index)]
            # highest_score = float(np.max(prediction))

            meilleurs_scores = np.flip(np.sort(prediction[0])[-3:])
            meilleures_classes_index = np.flip(np.argsort(prediction[0])[-3:])
            meilleures_classes = []
            for index in meilleures_classes_index:
                meilleures_classes.append(self.class_names[str(index)])
    
            
            logging.info(f"Prédiction effectuée : classe = {meilleures_classes}, score = {meilleurs_scores}")
            return meilleures_classes, meilleurs_scores
        except Exception as e:
            logging.error(f"Erreur lors de la prédiction : {str(e)}")
            raise

    # def get_class_names(self):
    #     return self.class_names
    
def load_classifier(run_id):
    model_path = os.path.join(volume_path, f'mlruns/157975935045122495/{run_id}/artifacts/model/')
    classifier = predictClass(model_path = model_path)
    classifier.predict('./load_image.jpg')
    return classifier
    
prod_model_id_path = os.path.join(mlruns_path, 'prod_model_id.txt')
while not os.path.exists(prod_model_id_path):
    time.sleep(1)

with open(prod_model_id_path, 'r') as file:
    run_id = file.read()

classifier = load_classifier(run_id)
    
@app.get("/")
def read_root():
    return {"Status": "OK"}

@app.get("/predict")
async def predict(file_name: str):
    try:
        global too_long_inference 
        start_time = time.time()
        temp_folder = os.path.join(volume_path, 'temp_images')
        image_path = os.path.join(temp_folder, file_name)
        logging.info("Début de la prédiction")
        meilleures_classes, meilleurs_scores = classifier.predict(image_path)
        logging.info(f"Prédiction terminée: {meilleures_classes}, score: {meilleurs_scores}")
        time.sleep(1)
        end_time = time.time()
        if ((end_time - start_time) > 1):
            too_long_inference += 1
        if too_long_inference > 3:
            too_long_inference = 0
            return {"predictions": meilleures_classes, "scores": meilleurs_scores.tolist(), "filename": file_name, 'too_long_inference': "Yes"}
        else:
            return {"predictions": meilleures_classes, "scores": meilleurs_scores.tolist(), "filename": file_name, 'too_long_inference': "No"}
    
    except Exception as e:
        logging.error(f'Failed to open the image and/or do the inference: {e}')
        raise HTTPException(status_code=500, detail="Internal server error")
    
@app.post("/switchmodel")
async def switch_model(run_id: str = Body(...)):
    try:
        global classifier
        run_id = run_id.removeprefix('run_id=')
        with open(os.path.join(mlruns_path, 'prod_model_id.txt'), 'w') as file:
            file.write(run_id)
            
        classifier = load_classifier(run_id)
        logging.info(f"Changement de modèle effectué !")
        return {f"Le nouveau modèle utilisé provient maintenant du run id suivant : {run_id}"}
    
    except Exception as e:
        logging.error(f'Failed to open the image and/or do the inference: {e}')
        raise HTTPException(status_code=500, detail="Internal server error")
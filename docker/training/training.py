import os
import pandas as pd
import mlflow
import mlflow.keras
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from timeit import default_timer as timer
from fastapi import FastAPI, HTTPException, BackgroundTasks
import logging
import mlflow
from mlflow.tracking import MlflowClient
import shutil
import json

app = FastAPI()

volume_path = 'volume_data'
log_folder = os.path.join(volume_path, "logs")
state_folder = os.path.join(volume_path, "containers_state")
experiment_id = "157975935045122495"
os.makedirs(state_folder, exist_ok = True)
state_path = os.path.join(state_folder, "training_state.txt")
preprocessing_state_path = os.path.join(state_folder, "preprocessing_state.txt")
with open(state_path, "w") as file:
    file.write("0")
dataset_folder = os.path.join(volume_path, "dataset_clean")
os.makedirs(log_folder, exist_ok = True)
logging.basicConfig(filename=os.path.join(log_folder, "training.log"), level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s', 
                    datefmt='%d/%m/%Y %I:%M:%S %p')

mlruns_path = os.path.join(volume_path, 'mlruns')
if not os.path.exists(mlruns_path):
    shutil.copytree('./mlruns', mlruns_path)
    shutil.copy('./prod_model_id.txt', mlruns_path)
else:
    shutil.rmtree('./mlruns')

mlflow.set_tracking_uri("file:///home/app/volume_data/mlruns")
client = MlflowClient()

class TimingCallback(Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)

def train_model():

    with open(state_path, "w") as file:
        file.write("1")

    mlflow.set_experiment("Bird Classification Training")

    with mlflow.start_run():
        # Définition du chemin vers le dataset

        mlflow.keras.autolog(log_models=False)
        
        train_path = os.path.join(dataset_folder, "train")
        valid_path = os.path.join(dataset_folder, "valid")
        test_path = os.path.join(dataset_folder, "test")

        # Vérification de l'existence des dossiers
        for path in [dataset_folder, train_path, valid_path, test_path]:
            if not os.path.exists(path):
                logging.error(f"Le dossier {path} n'existe pas.")
                raise FileNotFoundError(f"Le dossier {path} n'existe pas.")

        logging.info(f"Chemin d'entraînement : {train_path}")
        logging.info(f"Chemin actuel : {os.getcwd()}")

        # Définition de la batch size
        batch_size = 16
        mlflow.log_param("batch_size", batch_size)

        # Définition des callbacks
        reduce_learning_rate = ReduceLROnPlateau(monitor="val_loss", patience=2, min_delta=0.01, factor=0.1, cooldown=4, verbose=1)
        early_stopping = EarlyStopping(patience=5, min_delta=0.01, verbose=1, mode='min', monitor='val_loss')
        time_callback = TimingCallback()

        # Création des générateurs d'images avec augmentation des données plus agressive
        train_datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest')
        train_generator = train_datagen.flow_from_directory(train_path, target_size=(224, 224), batch_size=batch_size)
        valid_generator = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224, 224), batch_size=batch_size)
        test_generator = ImageDataGenerator().flow_from_directory(test_path, target_size=(224, 224), batch_size=batch_size)

        # Récupération du nombre de classes
        num_classes = train_generator.num_classes
        indices_classes_raw = train_generator.class_indices
        indices_classes = {}
        for classe in indices_classes_raw:
            indices_classes[indices_classes_raw[classe]] = classe
        classes_file_path = './classes.json'
        with open(classes_file_path, 'w') as json_file:
            json.dump(indices_classes, json_file)
        mlflow.log_artifact(classes_file_path, artifact_path="model")
        os.remove(classes_file_path)
        mlflow.log_param("num_classes", num_classes)

        # On se base sur le modèle pré-entrainé EfficientNetB0
        base_model = EfficientNetB0(weights='imagenet', include_top=False)

        # On dégèle les 20 dernières couches pour affiner le modèle
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        for layer in base_model.layers[-20:]:
            layer.trainable = True

        # On ajoute nos couches
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1280, activation='relu')(x) 
        x = Dropout(rate=0.2)(x)
        x = Dense(640, activation='relu')(x)  
        x = Dropout(rate=0.2)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        # Compilation du modèle avec un optimiseur Adam et un learning rate adaptatif
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc', 'mean_absolute_error'])

        # Entraînement du modèle
        training_history = model.fit(train_generator,
                            epochs=11,
                            steps_per_epoch=train_generator.samples//train_generator.batch_size,
                            validation_data=valid_generator,
                            validation_steps=valid_generator.samples//valid_generator.batch_size,
                            callbacks=[reduce_learning_rate, early_stopping, time_callback], 
                            verbose=1)

        # Évaluation du modèle sur le set de test
        test_loss, test_accuracy, test_mae = model.evaluate(test_generator)

        logging.info(f"Test accuracy: {test_accuracy}")
        logging.info(f"Final validation accuracy: {training_history.history['val_acc'][-1]}")

        # Sauvegarde du modèle au format h5
        model_save_path = f'saved_model.h5'
        model.save(model_save_path)
        mlflow.log_artifact(model_save_path, artifact_path="model")
        os.remove(model_save_path)
        # mlflow.tensorflow.log_model(model, artifact_path=f"model_{timestamp}")
        logging.info(f"Model sucessfuly saved in mlruns folder")

        generate_confusion_matrix(test_generator, model)

        mlflow.end_run()

        with open(state_path, "w") as file:
            file.write("0")


def generate_confusion_matrix(test_generator, model):
    """
    Génére la matrice de confusion (et métriques de recall, precision, et f1-score) pour le modèle
    """
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)

    # Obtenir les labels des classes réelles
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Créer la matrice de confusion
    conf_matrix = confusion_matrix(true_classes, predicted_classes)

    # Ajout des metriques au DataFrame
    confusion_df = pd.DataFrame(conf_matrix, index=class_labels, columns=class_labels)
    confusion_df = add_metrics(confusion_df)

    # Enregistrement de la matrice de confusion
    confusion_df.to_csv("./initial_confusion_matrix.csv")
    mlflow.log_artifact("./initial_confusion_matrix.csv")
    os.remove("./initial_confusion_matrix.csv")
  

def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajout au DataFrame des métriques de precision, recall et f1-score
    """
    df["Precision"] = df.apply(
        lambda row: np.where(df[row.name].sum() != 0, 
                            df.loc[row.name, row.name] / df[row.name].sum(), 
                            0), 
        axis=1)

    df["Recall"] = df.apply(
        lambda row: np.where(df.loc[row.name].sum() != 0, 
                            df.loc[row.name, row.name] / df.loc[row.name].sum(), 
                            0), 
        axis=1)

    df["f1-score"] = df.apply(
        lambda row: np.where((row["Precision"] + row["Recall"]) != 0,
                            (2 * row["Precision"] * row["Recall"]) / (row["Precision"] + row["Recall"]),
                            0),
        axis=1)

    return df

def get_worst_f1_scores(run_id : str):
    """
    Renvoie les f1-score et index les plus bas de la matrice de confusion d'une run
    """

    df = pd.read_csv(f"{mlruns_path}/{experiment_id}/{run_id}/artifacts/initial_confusion_matrix.csv",
                         index_col=0)
    worst_values = df.nsmallest(10, 'f1-score')
    index_and_values = worst_values.index, worst_values['f1-score']
    return index_and_values

@app.get("/")
def read_root():
    return {"Status": "OK"}

@app.get("/train")
async def train(background_tasks: BackgroundTasks):
    try:
        with open(preprocessing_state_path, "r") as file:
            preprocessing_state = file.read()
        with open(state_path, "r") as file:
            state = file.read()
        if preprocessing_state == "0" and state == "0" and len(os.listdir(dataset_folder)) > 1:
            background_tasks.add_task(train_model)
            return "Entraînement du modèle lancé, merci d'attende le mail indiquant le succès de la tâche."
        else:
            return "Un preprocessing ou en entraînement est en cours, merci de revenir plus tard."

    except Exception as e:
        logging.error(f'Failed to train the model: {e}')
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/results")
async def results():
    """
    Renvoie les métriques du dernier modèle et de celui en production
    """
    try:

        # Récuperer la run id du dernier modèle
        runs = client.search_runs(experiment_id)
        latest_run = runs[0]
        latest_run_id = latest_run.info.run_id

        # Récuperer la run id du modèle en production
        with open(os.path.join(mlruns_path, 'prod_model_id.txt'), 'r') as file:
            main_model_run_id = file.read()

        # Récupérer les métriques et pires f1-scores du dernier modèle
        latest_run_val_acc = latest_run.data.metrics.get('val_acc')
        latest_run_val_loss = latest_run.data.metrics.get('val_loss')
        latest_run_worst_f1_scores = get_worst_f1_scores(latest_run_id)

        # Récupérer les métriques et pires f1-scores du modèle en production
        main_model_run = client.get_run(main_model_run_id)
        main_model_val_acc = main_model_run.data.metrics.get('val_acc')
        main_model_val_loss = main_model_run.data.metrics.get('val_loss')
        main_model_worst_f1_scores = get_worst_f1_scores(main_model_run_id)

        return {
            "latest_run_id": latest_run_id,
            "latest_run_val_accuracy": latest_run_val_acc,
            "latest_run_val_loss": latest_run_val_loss,
            "latest_run_worst_f1_scores": zip(latest_run_worst_f1_scores[0], latest_run_worst_f1_scores[1]),
            "main_model_run_id": main_model_run_id,
            "main_model_val_accuracy": main_model_val_acc,
            "main_model_val_loss": main_model_val_loss,
            "main_model_worst_f1_scores": zip(main_model_worst_f1_scores[0], main_model_worst_f1_scores[1])
            }

    except Exception as e:
        logging.error(f"Echec de l'otention de l'id de la dernière run : {e}")
        raise HTTPException(status_code=500, detail="Erreur de serveur interne")

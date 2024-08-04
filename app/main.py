import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from timeit import default_timer as timer
from monitoring.drift_monitor import DriftMonitor
from monitoring.alert_system import AlertSystem
from app.utils.logger import setup_logger
from datetime import datetime
from contextlib import nullcontext

# Définition du chemin de base du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuration du logger
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = setup_logger('train_model', f'train_model_{timestamp}.log')

class TimingCallback(Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)

def train_model(start_mlflow_run=True):
    if start_mlflow_run:
        mlflow.set_experiment("Bird Classification Training")
        mlflow.tensorflow.autolog(disable=True)

    run_context = mlflow.start_run() if start_mlflow_run else nullcontext()

    with run_context:
        # Définition du chemin vers le dataset
        dataset_path = os.path.join(BASE_DIR, "data")
        train_path = os.path.join(dataset_path, "train")
        valid_path = os.path.join(dataset_path, "valid")
        test_path = os.path.join(dataset_path, "test")

        # Vérification de l'existence des dossiers
        for path in [dataset_path, train_path, valid_path, test_path]:
            if not os.path.exists(path):
                logger.error(f"Le dossier {path} n'existe pas.")
                raise FileNotFoundError(f"Le dossier {path} n'existe pas.")

        logger.info(f"Chemin d'entraînement : {train_path}")
        logger.info(f"Chemin actuel : {os.getcwd()}")

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
        mlflow.log_param("initial_learning_rate", 0.001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc', 'mean_absolute_error'])

        # Entraînement du modèle
        training_history = model.fit(train_generator,
                            epochs=1,
                            steps_per_epoch=train_generator.samples//train_generator.batch_size,
                            validation_data=valid_generator,
                            validation_steps=valid_generator.samples//valid_generator.batch_size,
                            callbacks=[reduce_learning_rate, early_stopping, time_callback],
                            verbose=1)

        # Évaluation du modèle sur le set de test
        test_loss, test_accuracy, test_mae = model.evaluate(test_generator)

        logger.info(f"Test accuracy: {test_accuracy}")
        logger.info(f"Final validation accuracy: {training_history.history['val_acc'][-1]}")

        # Sauvegarde du modèle au format SavedModel
        model_save_path = os.path.join(BASE_DIR, 'models', f'saved_model_{timestamp}')
        tf.saved_model.save(model, model_save_path)
        mlflow.log_artifact(model_save_path, artifact_path="model")
        logger.info(f"Model saved successfully at {model_save_path}")

        # Log des métriques manuellement
        mlflow.log_metric("test_accuracy", float(test_accuracy))
        mlflow.log_metric("test_loss", float(test_loss))
        mlflow.log_metric("test_mae", float(test_mae))
        mlflow.log_metric("final_val_accuracy", float(training_history.history['val_acc'][-1]))

        # Log de l'historique d'entraînement
        for epoch, (acc, val_acc, loss, val_loss) in enumerate(zip(
            training_history.history['acc'],
            training_history.history['val_acc'],
            training_history.history['loss'],
            training_history.history['val_loss']
        )):
            mlflow.log_metrics({
                f"accuracy_epoch_{epoch+1}": float(acc),
                f"val_accuracy_epoch_{epoch+1}": float(val_acc),
                f"loss_epoch_{epoch+1}": float(loss),
                f"val_loss_epoch_{epoch+1}": float(val_loss)
            }, step=epoch)

        # Log des temps d'exécution par époque
        for epoch, time in enumerate(time_callback.logs):
            mlflow.log_metric(f"epoch_{epoch+1}_time", float(time), step=epoch)

        # Vérification de drift et alerte
        drift_monitor = DriftMonitor()
        alert_system = AlertSystem()

        drift_detected, drift_details = drift_monitor.check_drift()
        if drift_detected:
            alert_message = f"Drift détecté après l'entraînement. Détails: {drift_details}"
            alert_system.send_alert("Alerte de Drift Post-Entraînement", alert_message)
            logger.warning(alert_message)

        # Comparaison avec le meilleur modèle précédent
        client = mlflow.tracking.MlflowClient()
        best_run = client.search_runs(
            experiment_ids=[run.info.experiment_id] if start_mlflow_run else [mlflow.active_run().info.experiment_id],
            filter_string="metrics.test_accuracy > 0",
            order_by=["metrics.test_accuracy DESC"],
            max_results=1
        )

        if best_run and float(best_run[0].data.metrics['test_accuracy']) > test_accuracy:
            performance_drop = float(best_run[0].data.metrics['test_accuracy']) - test_accuracy
            if performance_drop > 0.05:  # Seuil de 5% de dégradation
                alert_message = f"Dégradation des performances détectée. Ancienne accuracy: {float(best_run[0].data.metrics['test_accuracy'])}, Nouvelle accuracy: {test_accuracy}"
                alert_system.send_alert("Alerte de Dégradation des Performances", alert_message)
                logger.warning(alert_message)

if __name__ == "__main__":
    train_model()

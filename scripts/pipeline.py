import sys
import os
import threading
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'preprocessing'))

import mlflow
from monitoring.drift_monitor import DriftMonitor
from monitoring.performance_tracker import PerformanceTracker
from monitoring.alert_system import AlertSystem
from monitoring.system_monitor import SystemMonitor
from app.utils.logger import setup_logger, clean_old_logs
from app.utils.data_manager import DataManager
from app.models.predictClass import predictClass
from training.train_model import train_model
from preprocessing.preprocess_dataset import CleanDB

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger = setup_logger('pipeline', 'pipeline.log')

def preprocess_data(data_path):
    logger.info("Début du prétraitement des données")
    cleaner = CleanDB(data_path, treshold=False)
    cleaner.cleanAll()
    logger.info("Prétraitement des données terminé")

class SystemMonitorThread(threading.Thread):
    def __init__(self, duration):
        threading.Thread.__init__(self)
        self.duration = duration
        self.stop_event = threading.Event()
        self.monitor = SystemMonitor()

    def run(self):
        start_time = time.time()
        while not self.stop_event.is_set() and time.time() - start_time < self.duration:
            self.monitor.log_metrics()
            time.sleep(5)  # Log every 5 seconds

    def stop(self):
        self.stop_event.set()

def run_pipeline():
    clean_old_logs()

    mlflow.end_run()

    experiment_name = "Bird Classification Project"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id)
    
    with mlflow.start_run(run_name="Pipeline Run"):
        mlflow.set_tag("run_type", "pipeline")
        logger.info("Démarrage de la pipeline")

        monitor_thread = SystemMonitorThread(3600)  # Surveillez pendant 1 heure max
        monitor_thread.start()

        try:
            data_manager = DataManager()
            drift_monitor = DriftMonitor()
            performance_tracker = PerformanceTracker()
            alert_system = AlertSystem()

            data_path = os.path.join(BASE_DIR, "data")
            preprocess_data(data_path)

            logger.info("Début de l'entraînement du modèle")
            model, drift_detected_during_training = train_model(start_mlflow_run=False)
            logger.info("Fin de l'entraînement du modèle")

            if drift_detected_during_training:
                logger.warning("Drift détecté pendant l'entraînement")
                mlflow.log_param("drift_detected_during_training", True)

            predictor = predictClass()

            new_data = data_manager.load_new_data()
            all_classes = data_manager.get_class_names()

            predictions = {class_name: 0 for class_name in all_classes}
            new_species = set()
            unknown_images = []

            logger.info(f"Traitement de {len(new_data)} nouvelles images")

            for image_path, true_class in new_data:
                class_name, confidence = predictor.predict(image_path)
                
                if class_name in predictions:
                    predictions[class_name] += 1
                else:
                    new_species.add(class_name)
                    predictions[class_name] = 1

                if confidence < 0.5:  # Seuil arbitraire pour les images "inconnues"
                    unknown_images.append(image_path)

                performance_tracker.log_prediction(class_name, confidence, true_class=true_class)

            logger.info(f"Nombre total d'images traitées : {sum(predictions.values())}")
            mlflow.log_metric("total_images_processed", sum(predictions.values()))
            
            logger.info(f"Nouvelles espèces détectées : {len(new_species)}")
            mlflow.log_metric("new_species_detected", len(new_species))
            
            logger.info(f"Images non identifiées : {len(unknown_images)}")
            mlflow.log_metric("unknown_images", len(unknown_images))

            for class_name, count in predictions.items():
                if count > 0:
                    logger.info(f"  {class_name}: {count} images")
                    mlflow.log_metric(f"predictions_{class_name}", count)

            drift_detected, drift_details = drift_monitor.check_drift()
            drift_detected = drift_detected or drift_detected_during_training

            if drift_detected:
                logger.warning(f"Drift détecté: {drift_details}")
                alert_message = f"Drift détecté dans la pipeline de reconnaissance d'oiseaux.\n\nDétails : {drift_details}"
                alert_system.send_alert("Alerte de Drift", alert_message)
                mlflow.log_param("drift_detected", True)
                mlflow.log_param("drift_details", drift_details)
            else:
                logger.info("Aucun drift détecté")
                mlflow.log_param("drift_detected", False)

            overall_accuracy, class_accuracies = performance_tracker.get_performance_metrics()
            if overall_accuracy is not None:
                logger.info(f"Précision globale : {overall_accuracy:.4f}")
                mlflow.log_metric("overall_accuracy", overall_accuracy)
            else:
                logger.warning("Impossible de calculer la précision globale")
            
            logger.info("Précision par classe :")
            for class_name in all_classes:
                accuracy = class_accuracies.get(class_name)
                if accuracy is not None:
                    logger.info(f"  {class_name}: {accuracy:.4f}")
                    mlflow.log_metric(f"accuracy_{class_name}", accuracy)
                else:
                    logger.info(f"  {class_name}: Pas de données")

            logger.info("Pipeline terminée")

        finally:
            monitor_thread.stop()
            monitor_thread.join()

if __name__ == "__main__":
    run_pipeline()
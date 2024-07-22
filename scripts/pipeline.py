import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
from monitoring.drift_monitor import DriftMonitor
from monitoring.performance_tracker import PerformanceTracker
from monitoring.alert_system import AlertSystem
from app.utils.logger import setup_logger, clean_old_logs
from app.utils.data_manager import DataManager
from app.models.predictClass import predictClass
from training.train_model import train_model

logger = setup_logger('pipeline', 'pipeline.log')

def run_pipeline():
    # Nettoyage des anciens logs
    clean_old_logs()

    # Assurez-vous qu'aucune session MLflow n'est active
    mlflow.end_run()

    mlflow.set_experiment("Bird Classification Pipeline")
    
    with mlflow.start_run():
        logger.info("Démarrage de la pipeline")

        data_manager = DataManager()
        drift_monitor = DriftMonitor()
        performance_tracker = PerformanceTracker()
        alert_system = AlertSystem()
        predictor = predictClass()

        # Entraînement du modèle
        logger.info("Début de l'entraînement du modèle")
        train_model(start_mlflow_run=False)
        logger.info("Fin de l'entraînement du modèle")

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

if __name__ == "__main__":
    run_pipeline()
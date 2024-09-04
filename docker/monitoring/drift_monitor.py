import os
import numpy as np
import pandas as pd
import shutil
import json
import time
import logging
import schedule
from alert_system import AlertSystem
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

# region Configuration
volume_path = "/home/app/volume_data"
dataset_path = os.path.join(volume_path, "dataset_clean")
test_set_path = os.path.join(dataset_path, "test")
mlruns_path = os.path.join(volume_path, "mlruns")
log_folder = os.path.join(volume_path, "logs")
experiment_id = "157975935045122495"

os.makedirs(log_folder, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_folder, "drift_monitor.log"), level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p')
# endregion


class DriftMonitor:
    def __init__(self):
        logging.info("Début d'exécution du script de drift monitoring.")
        self.img_size = (224, 224)
        # Récupération du modèle et son emplacement
        with open(os.path.join(mlruns_path, "prod_model_id.txt"), "r") as file:
            self.run_id = file.read()
        self.alert_system = AlertSystem()
        self.artifacts_path = os.path.join(mlruns_path, f"{experiment_id}/{self.run_id}/artifacts/")
        self.model_path = os.path.join(mlruns_path, f"{experiment_id}/{self.run_id}/artifacts/model/")
        self.model = load_model(os.path.join(self.model_path, "saved_model.h5"))
        logging.info(f"Modèle traité : {self.model_path}/saved_model.h5")

    def make_current_model_confusion_matrix(self):
        """
        Créer une matrice de confusion du modèle en production avec les nouvelles données
        """
        self.exclude_unknown_classes()

        try:

            # Création du générateur d'images parcourant notre dossier test pour notre modèle
            datagen = ImageDataGenerator()
            test_generator = datagen.flow_from_directory(
                test_set_path,
                target_size=self.img_size,
                batch_size=32,
                class_mode="categorical",
                shuffle=False
            )

            logging.info("Prédiction et génération de la matrice de confusion.")

            # Prédire les classes pour l'ensemble du dataset
            predictions = self.model.predict(test_generator)
            predicted_classes = np.argmax(predictions, axis=1)

            # Obtenir les labels des classes réelles
            true_classes = test_generator.classes
            class_labels = list(test_generator.class_indices.keys())

            # Créer la matrice de confusion
            conf_matrix = confusion_matrix(true_classes, predicted_classes)

            # Ajout des metriques au DataFrame
            confusion_df = pd.DataFrame(conf_matrix, index=class_labels, columns=class_labels)
            confusion_df = self.add_metrics(confusion_df)

        finally:

            self.reinclude_unknown_classes()

        return confusion_df

    def exclude_unknown_classes(self):
        """
        Récupérer la liste des classes avec lesquelles le modèle a été entrainé
        """
        with open(os.path.join(self.model_path, "classes.json"), "r") as file:
            known_classes = list(json.load(file).values())  # Valeurs du dictionnaire = Liste des classes

        # Mettre de côté les classes sur lequelles le modèle n'a pas été entrainé
        for class_folder in os.listdir(test_set_path):
            if class_folder not in known_classes:
                logging.info(f"La classse {class_folder} n'est pas connue par le modèle, on la met de côté.")
                class_path = os.path.join(test_set_path, class_folder)
                class_new_path = os.path.join(dataset_path, "temp_data/test", class_folder)
                shutil.move(class_path, class_new_path)

    def reinclude_unknown_classes(self):
        """
        Réintégrer la liste des classes avec lesquelles le modèle a été entrainé
        """
        logging.info("On replace les classes mises de côté dans le dataset de test.")
        temp_test_folder = os.path.join(dataset_path, "temp_data/test")
        if os.path.exists(temp_test_folder):
            for classes in os.listdir(temp_test_folder):
                shutil.move(os.path.join(temp_test_folder, classes), os.path.join(test_set_path, classes))
            os.rmdir(temp_test_folder)

    def add_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajout au DataFrame des métriques de precision, recall et f1-score
        """
        df["Precision"] = df.apply(
            lambda row: df.loc[row.name, row.name] / df[row.name].sum(), axis=1)
        df["Recall"] = df.apply(
            lambda row: df.loc[row.name, row.name] / df.loc[row.name].sum(), axis=1)
        df["f1-score"] = df.apply(
            lambda row: (2 * row["Precision"] * row["Recall"]) / (row["Precision"] + row["Recall"]), axis=1)

        return df

    def compare_confusion_matrix(self, new_matrix: pd.DataFrame):
        """
        Ajoute au DataFrame des colonnes de comparaison des métriques avec la matrice de confusion initiale
        """
        initial_matrix = pd.read_csv(os.path.join(self.artifacts_path, "initial_confusion_matrix.csv"), index_col=0)

        new_matrix["diff Precision"] = new_matrix["Precision"] - initial_matrix["Precision"]
        new_matrix["diff Recall"] = new_matrix["Recall"] - initial_matrix["Recall"]
        new_matrix["diff f1-score"] = new_matrix["f1-score"] - initial_matrix["f1-score"]

        return new_matrix

    def get_best_f1_scores(self, df: pd.DataFrame):
        """
        Renvoie les f1-score les plaus haut de la matrice de confusion
        """
        best_values = df.nlargest(10, 'f1-score')
        index_and_values = best_values.index, best_values['f1-score']
        return index_and_values

    def get_worst_f1_scores(self, df: pd.DataFrame):
        """
        Renvoie les f1-score les plaus bas de la matrice de confusion
        """
        worst_values = df.nsmallest(10, 'f1-score')
        index_and_values = worst_values.index, worst_values['f1-score']
        return index_and_values

    def send_report_email(self, df: pd.DataFrame):
        """
        Envoie un email de résumé
        """
        logging.info("Envoie de l'email de rapport.")

        subject = f"Rapport de performance du modèle en production {self.run_id}."
        best_f1_scores = self.get_best_f1_scores(df)
        worst_f1_scores = self.get_worst_f1_scores(df)
        evol_f1_scores = df[np.abs(df["diff f1-score"]) > 0.03]

        best_scores_list = "\n".join(
            [f"{index} : {score}" for index, score in zip(best_f1_scores[0], best_f1_scores[1])])
        worst_scores_list = "\n".join(
            [f"{index} : {score}" for index, score in zip(worst_f1_scores[0], worst_f1_scores[1])])
        evolution_scores_list = "\n".join(
            [f"{index} : {score}" for index, score in zip(evol_f1_scores.index, evol_f1_scores["diff f1-score"])])

        message = f"""
        Une matrice de confusion a été généré pour le modèle {self.run_id} :

        Meilleures f1-score :
        {best_scores_list}

        Pires f1-score :
        {worst_scores_list}

        Evolution des f1-score :
        {evolution_scores_list}
        """
        self.alert_system.send_alert(subject=subject, message=message)


def main():

    drift_monitor = DriftMonitor()
    df = drift_monitor.make_current_model_confusion_matrix()
    df = drift_monitor.compare_confusion_matrix(df)
    drift_monitor.send_report_email(df)


if __name__ == "__main__":
    schedule.every().day.at("02:00").do(main)
    while True:
        schedule.run_pending()
        time.sleep(1)

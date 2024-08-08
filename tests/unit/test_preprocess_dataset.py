import unittest
import os
import shutil
import datetime
from PIL import Image
from app.utils.logger import setup_logger
from preprocessing import preprocess_dataset

# Configuration du logger
logger = setup_logger('test_preprocess_dataset', 'test_preprocess_dataset.log')

class TestPreprocessDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Initialisation de l'environnement de test.
        """
        logger.info(f"Début de test unitaire de preprocess_dataset.py")
        cls.processedDataTestFolder = f"./data/clean/test"
        cls.speciesImageFile = f"./data/clean/test/ZEBRA DOVE/1.jpg"


    def test_01_main(self):
        """
        Test du preprocessing du dataset
        Résultat attendu : Données nettoyées dans le dossier data/processed
        """
        logger.info(f"Test 01 : main")

        before_preprocess_time = datetime.datetime.now() # Date et heure avant l'étape de preprocessing

        preprocess_dataset.main() # Étape de preprocessing

        creation_time = datetime.datetime.fromtimestamp(os.path.getctime(self.processedDataTestFolder)) # Date et heure de création du nouveau dossier aux données nettoyées

        self.assertTrue(os.path.exists(self.processedDataTestFolder), f"Le répertoire {self.processedDataTestFolder} n'existe pas.") 
        self.assertTrue(before_preprocess_time <= creation_time, f"Le répertoire {self.processedDataTestFolder} actuel date d'avant l'étape de preprocessing.") 
        self.assertTrue(os.listdir(self.processedDataTestFolder), f"Le répertoire {self.processedDataTestFolder} est vide après l'étape de preprocessing.")
        self.assertTrue(os.path.isfile(f"{self.speciesImageFile}"), f"Le fichier {self.speciesImageFile} n'existe pas.") 
        with Image.open(f"{self.speciesImageFile}") as img:
            self.assertTrue(img.size == (224, 224), f"L'image {self.speciesImageFile} ne possède pas les dimensions attendues : {img.size} au lieu de (224, 224).")
        

    @classmethod
    def tearDownClass(cls):
        """
        Cloture de l'environnement de test
        """
        # Nettoyage du répertoire créé après le test
        shutil.rmtree(cls.processedDataTestFolder)

        logger.info(f"Fin de test unitaire de preprocess_dataset.py")


if __name__ == '__main__':
    unittest.main()

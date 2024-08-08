import unittest
import os
import shutil
from app.utils.logger import setup_logger
from scripts import downloadDataset

# Configuration du logger
logger = setup_logger('test_DownloadDataset', 'test_DownloadDataset.log')

class TestDownloadDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Initialisation de l'environnement de test.
        """
        logger.info(f"Début de test unitaire de downloadDataset.py")
        cls.mainFolder = f".data/raw"
        cls.trainFolder = f".data/raw/train"
        cls.testFolder = f".data/raw/test"
        cls.validFolder = f".data/raw/valid"
        cls.speciesFolder = f".data/raw/test/ZEBRA DOVE"


    def test_01_main(self):
        """
        Test du téléchargement du dataset Kaggle BIRDS xxx SPECIES
        Résultat attendu : Présence d'un dossier "archive", contenant un dossier "train", un dossier "test", un dossier "valid", un fichier "birds.csv" et un fichier "EfficientNet.h5"
        """
        logger.info(f"Test 01 : main")

        downloadDataset.main()

        self.assertTrue(os.path.exists(self.mainFolder), f"Le répertoire {self.mainFolder} n'existe pas.") 
        
        self.assertTrue(os.path.exists(self.trainFolder), f"Le répertoire {self.trainFolder} n'existe pas.")
        
        self.assertTrue(os.path.exists(self.testFolder), f"Le répertoire {self.testFolder} n'existe pas.")

        self.assertTrue(os.path.exists(self.validFolder), f"Le répertoire {self.validFolder} n'existe pas.")

        self.assertTrue(os.path.exists(self.speciesFolder), f"Le répertoire {self.speciesFolder} n'existe pas.")
        self.assertTrue(os.listdir(self.speciesFolder), f"Le répertoire {self.speciesFolder} est vide après le téléchargement.")


    @classmethod
    def tearDownClass(cls):
        """
        Cloture de l'environnement de test
        """
        # Nettoyage des fichiers téléchargés après le test
        shutil.rmtree(cls.mainFolder)

        logger.info(f"Fin de test unitaire de downloadDataset.py")

if __name__ == '__main__':
    unittest.main()

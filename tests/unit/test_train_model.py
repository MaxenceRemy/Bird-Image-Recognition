import unittest
import os
import datetime
from app.utils.logger import setup_logger
from training import train_model

# Configuration du logger
logger = setup_logger('test_train_model', 'test_train_model.log')

class TestTrainModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Initialisation de l'environnement de test.
        """
        logger.info(f"Début de test unitaire de train_model.py")


    def test_01_main(self):
        """
        Test de l'entrainement du modèle
        Résultat attendu : Création du fichier birds_model_XXXXXXXXXXXX.h5
        """
        logger.info(f"Test 01 : main")
        
        before_train_time = datetime.datetime.now() # Date et heure avant l'étape de preprocessing

        model_path = train_model.main(test_mode=True) # Entrainement du modèle

        creation_time = datetime.datetime.fromtimestamp(os.path.getctime(model_path)) # Date et heure de création du nouveau modèle

        self.assertTrue(os.path.isfile(f"{model_path}"), f"Le modèle {model_path} n'existe pas.") 
        self.assertTrue(before_train_time <= creation_time, f"Le modèle {model_path} date d'avant l'étape d'entrainement.") 


    @classmethod
    def tearDownClass(cls):
        """
        Cloture de l'environnement de test
        """
        # Suppression du modèle après le test
        os.remove(cls.modelFile)
        logger.info(f"Fin de test unitaire de train_model.py")


if __name__ == '__main__':
    unittest.main()

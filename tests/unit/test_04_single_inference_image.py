import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) # Ajoutez le chemin du projet au PYTHONPATH
import unittest
from app.utils.logger import setup_logger
from inference import single_image_inference

# Configuration du logger
logger = setup_logger('test_single_inference_image', 'test_single_inference_image.log')

class testSingleInferenceImage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Initialisation de l'environnement de test.
        """
        logger.info(f"Début de test unitaire de single_image_inference.py")

    def test_01_main(self):
        """
        Test d'inférence du modèle
        Résultat attendu : Une prédiction composée d'un nom d'espèce d'oiseau, et un score entre 0 et 1 
        """
        logger.info(f"Test 01 : main")

        # TODO : Actualiser cette fonction
        # species, score = single_image_inference() # Prédiction

        # self.assertTrue(bool(species.strip()), f"La classe prédite par le modèle est vide.") 
        # self.assertTrue( 0 <= score <= 1, f"Le score de la prédiction n'est pas compris dans l'intervalle [0;1].") 

    @classmethod
    def tearDownClass(cls):
        """
        Cloture de l'environnement de test
        """
        logger.info(f"Fin de test unitaire de single_image_inference.py")

if __name__ == '__main__':
    unittest.main()

import os
import sys
import unittest
from fastapi.testclient import TestClient
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) # Ajoutez le chemin du projet au PYTHONPATH
from app.utils.logger import setup_logger
from app.main import app

# Configuration du logger
logger = setup_logger('test_main', 'test_main.log')

class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logger.info(f"Début de test unitaire de main.py")
        cls.client = TestClient(app)
        load_dotenv() # Charger les variables d'environnement
        cls.API_KEY = os.getenv("API_KEY")
        cls.API_USERNAME = os.getenv("API_USERNAME")
        cls.ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
        cls.HEADERS = {"access_token": cls.API_KEY}
        cls.Token = ""

    def test_01_login(self):
        """
        Test de l'endpoint d'API "/token"
        Résultat attendu : Code de statut 200 et en retour un string "acess_token" et "token_type" = "bearer"
        """
        logger.info(f"Test 01 : login")
        response = self.client.post("/token", data={"username": self.API_USERNAME, "password": self.ADMIN_PASSWORD})
        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        self.assertIsInstance(json_response["access_token"], str)
        self.assertEqual(json_response["token_type"], "bearer")
        self.__class__.Token = json_response["access_token"]

    def test_02_get_status(self):
        """
        Test concluant de l'endpoint d'API "/"
        Résultat attendu : Code de statut 200
        """
        logger.info(f"Test 02 : get_status")
        response = self.client.get("/", headers=self.HEADERS, cookies={"Authorization": f"Bearer {self.Token}"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "Service is running"})

    def test_03_invalid_api_key(self):
        """
        Test de l'endpoint d'API "/" avec clé api erronée
        Résultat attendu : Code de statut 401 et en retour "detail" = "Invalid API Key"
        """
        logger.info(f"Test 03 : invalid_api_key")
        invalid_headers = {"access_token": "invalid_key"}
        response = self.client.get("/", headers=invalid_headers, cookies={"Authorization": f"Bearer {self.Token}"})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json(), {"detail": "Invalid API Key"})

    def test_04_invalid_token(self):
        """
        Test de l'endpoint d'API "/" avec token erroné
        Résultat attendu : Code de statut 401 et en retour "detail" = "Invalid JWT Token. Received: invalid_token"
        """
        logger.info(f"Test 04 : invalid_token")
        invalid_cookies = {"Authorization": "Bearer invalid_token"}
        response = self.client.get("/", headers=self.HEADERS, cookies=invalid_cookies)
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json(), {"detail": "Invalid JWT Token. Received: invalid_token"})

    def test_05_invalid_login(self):
        """
        Test de l'endpoint d'API "/token" avec des identifiants invalides
        Résultat attendu : Code de statut 401 et en retour "detail" = "Incorrect username or password"
        """
        logger.info(f"Test 05 : invalid_login")
        response = self.client.post("/token", data={"username": "invalid", "password": "invalid"})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json(), {"detail": "Incorrect username or password"})

    def test_06_predict(self):
        """
        Test de l'endpoint d'API "/predict"
        Résultat attendu : Code de statut 200 et en retour pour chaque image un score entre 0 et 1
        """
        logger.info(f"Test 06 : predict")
        # On parcourt le dossier contenant les images de tests
        image_folder = "./data/test_images"
        for image_filename in os.listdir(image_folder):
            image_path = os.path.join(image_folder, image_filename)
            expected_label = (os.path.splitext(image_filename)[0]).lower()
            
            with open(image_path, "rb") as f:
                files = {"file": (image_filename, f, "image/png")}
                response = self.client.post("/predict", files=files, headers=self.HEADERS, cookies={"Authorization": f"Bearer {self.Token}"})
                
                self.assertEqual(response.status_code, 200)
                response_json = response.json()
                predicted_label = str(response_json["prediction"]).lower() # Le nom du fichier (sans extension) est notre label attendu
                score = response_json["score"]
                self.assertEqual(predicted_label, expected_label, f"Expected {expected_label} but got {predicted_label}")
                self.assertTrue(0 <= score <= 1, f"Expected score between 0 and 1 but got {score}")

    def test_07_add_image(self):
        """
        Test de l'endpoint d'API "/add_image"
        Résultat attendu : Code de statut 200 et en retour "status" = "Image ajoutée avec succès"
        """
        logger.info(f"Test 07 : add_image")
        with open("./data/test_images/Iwi.png", "rb") as f:
            files = {"file": ("test_image.png", f, "image/png")}
            headers_with_label = {**self.HEADERS, "label": "test_label"}
            response = self.client.post("/add_image", files=files, headers=headers_with_label, cookies={"Authorization": f"Bearer {self.Token}"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "Image ajoutée avec succès"})

    def test_08_get_species(self):
        """
        Test de l'endpoint d'API "/get_species"
        Résultat attendu : Code de statut 200 et en retour la liste des espèces d'oiseaux 
        """
        logger.info(f"Test 08 : get_species")
        response = self.client.get("/get_species", headers=self.HEADERS, cookies={"Authorization": f"Bearer {self.Token}"})
        self.assertEqual(response.status_code, 200)
        json_response = response.json()
        self.assertIn("species", json_response)
        self.assertEqual(len(json_response["species"]), 11194)

    def test_09_get_class_image(self):
        """
        Test de l'endpoint d'API "/get_class_image"
        Résultat attendu : Code de statut 200 et en retour une image
        """
        logger.info(f"Test 09 : get_class_image")
        species = "SAND MARTIN"
        response = self.client.get("/get_class_image", params={"classe": species}, headers=self.HEADERS, cookies={"Authorization": f"Bearer {self.Token}"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertEqual(response.headers["filename"], species)

    @classmethod
    def tearDownClass(cls):
        """
        Cloture de l'environnement de test
        """
        logger.info(f"Fin de test unitaire de main.py")

if __name__ == '__main__':
    unittest.main()
    
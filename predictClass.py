import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
import logging

logging.basicConfig(level=logging.INFO)

class predictClass:
    def __init__(self, model_path="./weights/main_model.h5", test_path="./data/test", img_size=(224, 224)):
        self.img_size = img_size
        self.path_test = os.path.join(test_path)
        
        try:
            self.test_generator = ImageDataGenerator().flow_from_directory(
                self.path_test, target_size=self.img_size, batch_size=16)
            self.num_classes = self.test_generator.num_classes
            
            self.model = self.build_model()
            self.model.load_weights(model_path)
            logging.info("Modèle chargé avec succès.")
        except Exception as e:
            logging.error(f"Erreur lors de l'initialisation : {str(e)}")
            raise

    def build_model(self):
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=self.img_size + (3,))

        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1280, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(640, activation='relu')(x)
        x = Dropout(0.2)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers[-20:]:
            layer.trainable = True
        
        return model

    def predict(self, image_path):
        try:
            img = image.load_img(image_path, target_size=self.img_size)
            img_array = image.img_to_array(img)
            img_array_expanded_dims = np.expand_dims(img_array, axis=0)
            img_ready = preprocess_input(img_array_expanded_dims)
            
            prediction = self.model.predict(img_ready)

            highest_score_index = np.argmax(prediction)
            liste_classes = os.listdir(self.path_test)
            meilleure_classe = liste_classes[highest_score_index]
            highest_score = float(np.max(prediction))
            
            logging.info(f"Prédiction effectuée : classe = {meilleure_classe}, score = {highest_score}")
            return meilleure_classe, highest_score
        except Exception as e:
            logging.error(f"Erreur lors de la prédiction : {str(e)}")
            raise
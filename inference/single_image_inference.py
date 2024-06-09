import os
import numpy as np
import argparse
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model

# on importe le dossier contenant les classes à tester
path_test = os.path.join("./data", 'test')

# on récupère le nombre de classes
test_generator = ImageDataGenerator().flow_from_directory(path_test, target_size=(224, 224), batch_size=16)
num_classes = test_generator.num_classes

# Construction du modèle
def build_model(num_classes, rate_Droput=0.3):
    # on se base sur le modèle pré-entrainé EfficientNetB0
    base_model = EfficientNetB0(weights='imagenet', include_top=False)
   
    # on gèle les couches du modèle de base
    for layer in base_model.layers: 
        layer.trainable = False

    # on ajoute nos couches
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1280, activation='relu')(x) 
    x = Dropout(rate=rate_Droput)(x)
    x = Dense(640, activation='relu')(x)  
    x = Dropout(rate=rate_Droput)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers[-20:]:
        layer.trainable = True
    
    return model

model = build_model(num_classes, rate_Droput=0.2)


# on importe les poids du modèle EfficientNet
model.load_weights("./weights/main_model.h5")

# on importe l'image, on la convertit en tableau puis on effectue le pré-traitement
image_path = os.path.join(path_test, 'FAIRY BLUEBIRD', '7.jpg') # vous pouvez ici changer l'image
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array_expanded_dims = np.expand_dims(img_array, axis=0)
img_ready = preprocess_input(img_array_expanded_dims)

# on effecute la prédiction
prediction = model.predict(img_ready)

# on récupère la classe ainsi que son score
highest_score_index = np.argmax(prediction)
liste_classes = os.listdir(path_test)
meilleure_classe = liste_classes[highest_score_index]
highest_score = float(np.max(prediction))

# on affiche les résultats
print(f"Classe prédite: {meilleure_classe}, avec un taux de confiance de {round(highest_score, 2)}")
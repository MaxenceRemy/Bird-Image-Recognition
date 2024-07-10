from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from timeit import default_timer as timer
import numpy as np
import os

class TimingCallback(Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)

# Définition du chemin vers le dataset
dataset_path = "./data"

train_path = os.path.join(dataset_path, "train")
valid_path = os.path.join(dataset_path, "valid")
test_path = os.path.join(dataset_path, "test")

# Définition de la batch size
batch_size = 16

# Définition des callbacks
reduce_learning_rate = ReduceLROnPlateau(monitor="val_loss", patience=2, min_delta=0.01, factor=0.1, cooldown=4, verbose=1)
early_stopping = EarlyStopping(patience=3, min_delta=0.01, verbose=1, mode='min', monitor='val_loss')
time_callback = TimingCallback()
    


# Création des générateurs d'images avec augmentation des données
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(train_path, target_size=(224, 224), batch_size=batch_size)
valid_generator = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224, 224), batch_size=batch_size)
test_generator = ImageDataGenerator().flow_from_directory(test_path, target_size=(224, 224), batch_size=batch_size)

# Récupération du nombre de classes
num_classes = train_generator.num_classes

# on se base sur le modèle pré-entrainé EfficientNetB0
base_model = EfficientNetB0(weights='imagenet', include_top=False)

# on gèle les couches du modèle de base
for layer in base_model.layers: 
    layer.trainable = False

# on ajoute nos couches
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1280, activation='relu')(x) 
x = Dropout(rate=0.2)(x)
x = Dense(640, activation='relu')(x)  
x = Dropout(rate=0.2)(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compilation du modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', 'mean_absolute_error'])

# Entraînement du modèle
training_history = model.fit(train_generator,
                    epochs=8,  
                    steps_per_epoch=train_generator.samples//train_generator.batch_size,
                    validation_data=valid_generator,
                    validation_steps=valid_generator.samples//valid_generator.batch_size,
                    callbacks=[reduce_learning_rate, early_stopping, time_callback], verbose=1)

# on dégèle les 20 dernières couches pour affiner le modèle
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Compilation du modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', 'mean_absolute_error'])

# Entraînement du modèle
training_history = model.fit(train_generator,
                    epochs=8,  
                    steps_per_epoch=train_generator.samples//train_generator.batch_size,
                    validation_data=valid_generator,
                    validation_steps=valid_generator.samples//valid_generator.batch_size,
                    callbacks=[reduce_learning_rate, early_stopping, time_callback], verbose=1)

# Évaluation du modèle sur le set de test

test_accuracy = model.evaluate(test_generator)
print(test_accuracy)
print(training_history.history['val_acc'][-1])

# Sauvegarde des poids du modèle
model.save_weights('./weights/main_model.h5')
# Bird Image Classification

Le code présent dans ce repository permet de télécharger un dataset contenant près de 60 000 images d'oiseaux et 525 espèces, d'appliquer un préprocessing, d'entraîner un modèle Tensorflow et de faire l'inférence sur une image.

## Prérequis

Pour commencer, créez et activez un nouvel environnement virtuel sous Python 3.10 :

```bash
# Création de l'environnement virtuel
py -3.10 -m venv birdEnv

# Activation de l'environnement sous LINUX
source birdEnv/bin/activate

# Activation de l'environnement sous WINDOWS
birdEnv/Scripts/activate
```

Ensuite, installer les requirements :

```bash
pip install -r requirements.txt
```

Téléchargez maintenant le dataset en exécutant le script suivant : 

```bash
python downloadDataset.py
```

## Inférence

Vous pouvez faire l'inférence sur une image en indiquant la classe et le nom du fichier dans le script single_image_inference.py situé dans le dossier inference,
puis en exécutant le fichier : 

```bash
python inference/single_image_inference.py
```

## Entraînement

Commencez par le preprocessing (suppression de mauvaises classes, redimensionnement, répartition Train/Test/Valid) en éxecutant le script suivant :

```bash
python preprocessing/preprocess_dataset.py
```

Ensuite, lancez le script d'entraînement :

```bash
python training/train_model.py
```

ATTENTION : pour ne pas écraser les poids du modèle par défaut, il faut changer le nom du fichier .h5 que vous allez enregistrer.

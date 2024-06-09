import os
import urllib.request
import zipfile
import shutil

def get_dataset(url, destination):

    temporaire = "./temporaire"
    
    # On créer le dossier temporaire pour télécharger le fichier
    if not os.path.exists(temporaire):
        os.makedirs(temporaire)

    # On créer le dossier du dataset s'il n'existe pas
    if not os.path.exists(destination):
        os.makedirs(destination)
    
    # On télécharge le fichier
    zip_path = os.path.join(temporaire, "dataset.zip")
    urllib.request.urlretrieve(url, zip_path)
    print("Fichier téléchargé !")
    
    # On extrait l'archive
    with zipfile.ZipFile(zip_path, 'r') as zip:
        zip.extractall(destination)
    print(f"Téléchargement terminé !")

    # On supprime les fichiers temporaires
    os.remove(zip_path)
    os.remove("./data/birds.csv")
    os.remove("./data/EfficientNetB0-525-(224 X 224)- 98.97.h5")
    shutil.rmtree(temporaire)


url = "https://storage.googleapis.com/kaggle-data-sets/534640/5468571/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240608%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240608T104245Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=805bdc234fb7c6b168ab289d6ca4e02a1e2eb4ebe844814d49b889a3dd3496954e23316f1f00fab702ad42b3a3eeef3a1d1d315bbebe83dd8b6766d0e659c44de02c4f8ae824af7601c38936a51a94d14982b40ca6f63e42fbae737b273c465e9a337013ebaf5808cf8efa3a4b703aa72791fa10f099cb2b69c12569902e6c39a5e2d190d773e08514b13639d763e16fbc4832cd479dedadfcf37fbb4422c9c2998bd3f7c47ac757315ceb6f5314fda06f1e628eeecfb326f6744c50cabab790a664bc8afb36495fa646a3fb2b716c55d84ab7ca481f6f6ce4fa51eb7bf755c43263d9e1c1f87c5125c55c449684d4d99cfd320376d1a3fdfb85cd1bc560680d"
destination = "./data"
get_dataset(url, destination)
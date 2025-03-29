import os
import shutil
import random

def split_train_val(source_dir, dest_dir, train_ratio=0.8):
    """
    source_dir : chemin du dossier contenant toutes les classes à séparer (ex : "pytorch_cnn/data/animals_original")
                avec la structure :
                     animals_original/
                         elephant/
                         gorilla/
                         leopard/
                         ...
    dest_dir   : dossier de destination (ex : "pytorch_cnn/data/animals")
    train_ratio: pourcentage d'images à mettre dans le train (ex : 0.8 = 80%)
    """

    # Création des dossiers train et val si pas déjà existants
    train_dir = os.path.join(dest_dir, "train")
    val_dir   = os.path.join(dest_dir, "val")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Pour chaque classe (dossier) dans source_dir
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)

        # Vérifie que c'est bien un dossier (une classe)
        if os.path.isdir(class_path):
            # Liste toutes les images de la classe
            images = os.listdir(class_path)
            random.shuffle(images)

            # On calcule la limite pour le train/val
            train_count = int(len(images) * train_ratio)

            # Sépare en deux listes
            train_images = images[:train_count]
            val_images   = images[train_count:]

            # Création du sous-dossier de la classe pour train et val
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

            # Copie des images de train
            for img in train_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(train_dir, class_name, img)
                shutil.copy2(src, dst)

            # Copie des images de val
            for img in val_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(val_dir, class_name, img)
                shutil.copy2(src, dst)

            print(f"Classe '{class_name}': {len(train_images)} en train, {len(val_images)} en val.")

if __name__ == "__main__":
    source_dir = "pytorch_cnn/data/animals"  # dossier où se trouvent les classes
    dest_dir   = "test_moi/data_sep"          # dossier final avec train/val
    split_train_val(source_dir, dest_dir, train_ratio=0.8)

# Projet de Machine Learning

Ce projet met en œuvre une solution de Machine Learning en utilisant PyTorch pour entraîner des modèles de classification d'images sur un dataset d'animaux. Deux approches sont proposées :  
- **Fine-tuning complet** (modèle : `model_ft_full.pth`)  
- **Transfer learning sur la dernière couche** (modèle : `model_conv_full.pth`)

Le projet inclut des scripts pour la préparation des données, l'entraînement, la visualisation des résultats, ainsi qu'un notebook pour tester le modèle.

---

## Architecture du Projet
```plaintext
├── code/ 
│ ├── separation_data.py # Script pour séparer les données en ensembles train et validation 
│ ├── test_model.ipynb # Notebook pour tester le modèle 
│ └── transfer_learning_test.py # Script principal pour l'entraînement, la visualisation et la sauvegarde des modèles
│
├── data_model/
│ ├── model_conv_full.pth # Modèle entraîné avec transfer learning (seule la dernière couche) 
│ └── model_ft_full.pth # Modèle entraîné en fine-tuning complet
│
├── data_sep/ # Images séparées en ensembles train/validation grâce à separation_data.py 
└── image_test/ # Images pour tester le notebook
```

---

## Détails des Scripts

### separation_data.py

- **Fonctionnalité :**  
  Ce script parcourt un dossier source contenant des sous-dossiers par classe (par exemple, `pytorch_cnn/data/animals_original`) et répartit les images dans deux ensembles :  
  - `train` (80% des images par défaut)  
  - `val` (20% des images par défaut)

- **Utilisation :**  
  - Modifier les variables `source_dir` et `dest_dir` si nécessaire.
  - Exécuter le script pour créer la structure de données dans le dossier de destination.

### transfer_learning_test.py

- **Fonctionnalité :**  
  Ce script effectue les étapes suivantes :
  - **Préparation des données :**  
    - Applique des transformations (data augmentation pour l'entraînement et normalisation pour la validation).
    - Charge les images depuis le dossier `data_sep/` en créant des DataLoaders pour les ensembles `train` et `val`.
  - **Entraînement des modèles :**  
    - Entraîne deux versions d'un modèle ResNet18 :
      - **Fine-tuning complet** : Tous les paramètres du modèle sont optimisés.
      - **Transfer learning** : Seuls les paramètres de la dernière couche sont optimisés.
  - **Visualisation et sauvegarde :**  
    - Visualise quelques prédictions sur les données de validation.
    - Sauvegarde les modèles entraînés dans le dossier `data_model/`.

- **Utilisation :**  
  - Assurez-vous que le dossier `data_sep/` est correctement structuré (avec des sous-dossiers pour chaque classe).
  - Exécutez le script pour entraîner et sauvegarder les modèles.

### test_model.ipynb

- **Fonctionnalité :**  
  Ce notebook permet de tester le modèle entraîné sur de nouvelles images et de visualiser les prédictions.

- **Utilisation :**  
  - Ouvrir le notebook dans Jupyter.
  - Exécuter les cellules pour charger le modèle et tester sur des images situées dans le dossier `image_test/`.

---

## Prérequis

- **Python 3.x**
- **PyTorch**
- **Torchvision**
- **Matplotlib**
- **NumPy**
- **PIL (Pillow)**

Installez les dépendances nécessaires via `pip` :

```bash
pip install torch torchvision matplotlib numpy pillow

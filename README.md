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
├── data_model/
│ ├── model_conv_full.pth # Modèle entraîné avec transfer learning (seule la dernière couche) 
│ └── model_ft_full.pth # Modèle entraîné en fine-tuning complet 
├── data_sep/ # Images séparées en ensembles train/validation grâce à separation_data.py 
└── image_test/ # Images pour tester le notebook
```

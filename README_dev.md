/Applications/Blender.app/Contents/Resources/2.91/python/bin/python3.7m -m ensurepip
/Applications/Blender.app/Contents/Resources/2.91/python/bin/python3.7m -m pip install -U pip
/Applications/Blender.app/Contents/Resources/2.91/python/bin/python3.7m -m pip install lxml
/Applications/Blender.app/Contents/Resources/2.91/python/bin/python3.7m -m pip install opencv-python


# generation d'image
```
blender --background --python generate_images.py
```

# environnement
## create
```
conda env create -f environment.yml
conda activate presence
```
## update

conda env export > environment.yml


# Entrainement
## Avant l'entraînement
### Format des données

Les données doivent être sous le format suivant (voir VOCLilke)

Dans Annotations se trouvent les fichiers *.xml* d'annotations
Dans JPEGImages se trouvent les images d'extension *.jpg*
Dans ImageSets/Main se trouvent le découpage en train, validation, test set

```bash
the/name/of/the/root/file
└── VOC2021
  ├── Annotations
  ├── ImageSets
  │ └── Main
  │ ├── train.txt
  │ └── val.txt
  │ └── test.txt
  └── JPEGImages
```
Pour compléter le jeu de données réel, il faut suivre les étapes suivantes:

### Renommage des images

Pour plus d'informations sur les arguments des CLI, voir **CLI_data.py**

Vous pouvez renommer les images KO ou OK (en EAF_OK_chiffre.jpg ou EAF_KO_chiffre.jpg) d'un dossier avec la CLI suivante:

python CLI_data.py rename 

#exemple 

python CLI_data.py rename --path='path/du/dossier/images' start=0 is_ok=True

Après avoir renommer les images, il faut placer les images dans *VOC2021/JPEGImages*.

### L'Annotation

L'annotation peut se faire avec le module open-source labelImg (https://github.com/tzutalin/labelImg) en sauvegardant les annotations sous xml (PASCAL VOC format).
Les fichiers d'annotation sont à placer dans *VOC2021/Annotations*

### Modifier les fichiers de découpage

Ensuite il faut ajouter les noms (sans l'extension) aux fichiers Main/*.txt* soit dans le train.txt (pour compléter les données d'entraînement), soit dans le val.txt (pour compléter les données de validation), soit le test.txt (pour compléter les données de test)

## CLI pour l'entraînement

Pour plus d'informations sur les arguments des CLI, voir **CLI_detector.py**

```
python CLI_detector.py train_from_pretrained (entrainement à partir du modèle pré-entraîné)

#exemple
python CLI_detector.py train_from_pretrained --data_path='../Data/EAF' --save_prefix='save_name_model' --batch_size=10

python CLI_detector.py train_from_finetuned (entraînement à partir d'un modèle sauvegardé en local)

#exemple 
python CLI_detector.py train_from_finetuned --save_prefix='ssd_512' --data_path='../Data/EAF_real' --model_name='models/model_name_best.params' --batch_size=10 --epoch=15

python CLI_detector.py eval (évaluation d'un modèle en fixant un taux de faux positif en affichant la matrice de  confusion dans la console et en le sauvegardant dans un fichier log dans *logs* et en sauvegardant la courbe ROC curve dans *results_ROC*)

#exemple
!python CLI_detector.py eval --data_path_test='../Data/EAF_real' --save_prefix='fake400_19style+real_on_real' --model_name='models/path/to/model' --taux_fp=0.143

python CLI_detector.py predict (faire une prédiction sur une image)

#exemple
python CLI_detector.py predict model_name='models/ssd_512_best.params' input_path='inputs/EAF3.jpg' output_folder='outputs/' thresh=0.3 data_path_test='../Data/EAF_real'
```
# Pour matthieu (entraînement et evaluate)
```bash
├── Annotations
├── Data
│   ├── EAF_false
│   │   └── VOC2021
│   │       ├── Annotations
│   │       ├── ImageSets
│   │       │   └── Main
│   │       └── JPEGImages
│   ├── EAF_real
│   │   └── VOC2021
│   │       ├── Annotations
│   │       ├── ImageSets
│   │       │   └── Main
│   │       └── JPEGImages
│   ├── EAF_test
│   │   └── VOC2021
│   │       ├── Annotations
│   │       ├── ImageSets
│   │       │   └── Main
│   │       └── JPEGImages
│   ├── Photos
│   └── Photos2
├── Images
├── Visualisations
├── blender
├── blender_test
├── label_img
├── notebook
└── src
    ├── Detector
    ├── app
    │   ├── api
    │   │   └── eaf
    │   │       └── endpoints
    │   ├── models
    │   ├── outputs
    │   ├── tests
    │   └── uploads
    ├── detector_utils
    ├── inputs
    ├── logs
    ├── models
    ├── outputs
    ├── results_ROC
    ├── results_train
    └── tests
```
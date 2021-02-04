Voici le repository pour le server du projet DTY du Paris Digital Lab pour Sonergia.

Il contient :

- une api qui permet de vérifier la présence d'un écart au feu à partir d'une image
- une api qui permet de vérifier la conformité 'un écart au feu à partir d'une image
- une cli qui permet de générer des images sythétiques
- une cli qui permet de tranferer des style d'un groupe d'image à un autre
- une cli qui permet d'entrainer, évaluer un modèle de détection et de prédire en utlisant ce modèle

# setup generation d'image

LA génération d'image se fait avec l'application blender, une application open-source qui permet de faire de la modélisation 3D, pour lancer une génération il faut installer blender 2.9 et vérifier que l'application est en anglais ( changer la langue directement dans l'application si ce n'est pas le cas ).
Ensuite il faut installer quelques dépendances dans l'environement de blender avec les commandes :

```bash
/Applications/Blender.app/Contents/Resources/2.91/python/bin/python3.7m -m ensurepip

/Applications/Blender.app/Contents/Resources/2.91/python/bin/python3.7m -m pip install -U pip

/Applications/Blender.app/Contents/Resources/2.91/python/bin/python3.7m -m pip install lxml

/Applications/Blender.app/Contents/Resources/2.91/python/bin/python3.7m -m pip install opencv-python
```

Vérifiez que le chemin mène bien à votre installation blender (c'est lignes de commande devraient marcher uniquement sur macOS )

# generation d'image

Pour générer des images, utilisez la commande suivante dans le dossier lib ( uniquement dans le dossier lib ):

```bash
blender -b --python generate_images.py -- -a generate_train -y 300 -n 300
```

Les arguments sont les suivants :
-a :
soit "generate_train" pour générer un jeu de train, qui sera séparé à 70% pour l'entrainement et 30% pour la validation
soit "generate_test" pour générer un jeu de test

-y : le nombre d'images avec une protection à générer, à defaut en créé 200
-n : le nombre d'images sans protection à générer, à defaut en créé 200
-r : path vers le dossier ou vous voulez générer votre jeu, à defaut on créé un dossier EAF

( N'oubliez pas les -- avant les arguments, ils sont indispensables )

# style

Pour ajouter des style à vos images il faut créér 3 dossiers :

- un dossier ou vous mettez vous images de bases ( par exemple les images générées)
- un dossier ou vous mettez les images qui vont servir de modèle pour le style
- un dossier ou recevoir les images stylisées

Lancez avec la commande suivante

```bash
python style.py -c "/path/to/contentimages" -s "/path/to/styleimages" -o "/path/to/output"
```

Les arguments sont les suivants :
-c : path vers le dossier d'images de base, à default: './EAF/VOC2021/JPEGImages'
-s : path vers le dossier d'images de reférence pour le style
-c : path vers le dossier ou mettre les images stylisées

# environnement

## create

```
conda env create -f environment.yml
conda activate presence
```

## update

conda env export > environment.yml

# Entrainement

## Données

```bash
the/name/of/the/root/file
└── VOC2021
  ├── Annotations
  ├── ImageSets
  │ └── Main
  │ ├── test.txt
  │ └── trainval.txt
  └── JPEGImages
```

# API

python app.py

# CLI

```
python CLI.py train_from_pretrained (train_from_the_ssd_mobile_net)

#exemple
python CLI.py train_from_pretrained --data_path='../Data/EAF' --save_prefix='save_name_model' --batch_size=10

python CLI.py train_from_finetuned (train_from_a_saved_model)

#exemple
python CLI.py train_from_finetuned --save_prefix='ssd_512' --data_path='../Data/EAF_real' --model_name='models/model_name_best.params' --batch_size=10 --epoch=15

python CLI.py eval

#
python CLI.py predict
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

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

## Avant l'entraînement

### Format des données

Les données doivent être sous le format suivant (voir VOCLilke)

Dans Annotations se trouvent les fichiers _.xml_ d'annotations
Dans JPEGImages se trouvent les images d'extension _.jpg_
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

Après avoir renommer les images, il faut placer les images dans _VOC2021/JPEGImages_.

### L'Annotation

L'annotation peut se faire avec le module open-source labelImg (https://github.com/tzutalin/labelImg) en sauvegardant les annotations sous xml (PASCAL VOC format).
Les fichiers d'annotation sont à placer dans _VOC2021/Annotations_

### Modifier les fichiers de découpage

Ensuite il faut ajouter les noms (sans l'extension) aux fichiers Main/_.txt_ soit dans le train.txt (pour compléter les données d'entraînement), soit dans le val.txt (pour compléter les données de validation), soit le test.txt (pour compléter les données de test)

## CLI pour l'entraînement, l'évaluation et la prédiction

Pour plus d'informations sur les arguments des CLI, voir **CLI_detector.py**

```
python CLI_detector.py train_from_pretrained (entrainement à partir du modèle pré-entraîné)

#exemple
python CLI_detector.py train_from_pretrained --data_path='../Data/EAF' --save_prefix='save_name_model' --batch_size=10

--------------------------------------------------------------------------

python CLI_detector.py train_from_finetuned (entraînement à partir d'un modèle sauvegardé en local)

#exemple
python CLI_detector.py train_from_finetuned --save_prefix='ssd_512' --data_path='../Data/EAF_real' --model_name='models/model_name_best.params' --batch_size=10 --epoch=15

--------------------------------------------------------------------------

python CLI_detector.py eval (évaluation d'un modèle en fixant un taux de faux positif en affichant la matrice de  confusion dans la console et en le sauvegardant dans un fichier log dans *logs* et en sauvegardant la courbe ROC curve dans *results_ROC*)

#exemple
!python CLI_detector.py eval --data_path_test='../Data/EAF_real' --save_prefix='fake400_19style+real_on_real' --model_name='models/path/to/model' --taux_fp=0.143

--------------------------------------------------------------------------

python CLI_detector.py predict (faire une prédiction sur une image)

#exemple
python CLI_detector.py predict model_name='models/ssd_512_best.params' input_path='inputs/EAF3.jpg' output_folder='outputs/' thresh=0.3 data_path_test='../Data/EAF_real'
```

# Structure du repository

```bash
├── Data
│   ├── EAF_real
│   │   └── VOC2021
│   │       ├── Annotations
│   │       ├── ImageSets
│   │       │   └── Main
│   │       └── JPEGImages
└── src
    ├── Detector
    ├── app
    │   ├── models (dossier où se trouve le modèle utilisé par lapp)
    │   ├── outputs (sortie du modèle)
    │   ├── tests
    │   └── uploads (entrée du modèle)
    ├── detector_utils
    ├── inputs
    ├── logs (les logs notamment ceux concernant lévaluation)
    ├── models (modèles en sortie des entraînements)
    ├── outputs
    ├── results_ROC (endroit où sont sauvegardés les courbes ROC)
    ├── results_train (endroit où sont sauvegardés les courbes dentraînement)
    └── tests
```

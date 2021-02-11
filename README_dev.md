Voici le repository pour le server du projet DTY du Paris Digital Lab pour Sonergia contenant le cas d'usage de détection de l'écart au feu et de sa conformité.

# Installation

Pour installer l'environnement veuillez regarder le README_usage.mdr


# Test

Avant d'effectuer le test :

- mettre dans tests/test_data/models le modèle.

```
python -m unittest discover -s tests
```

# Entrainement

Le site de gluoncv peut être utile si il y a besoin d'approfondir le code (https://cv.gluon.ai/contents.html)

## Avant l'entraînement

### Format des données

Les données doivent être sous le format suivant (voir VOCLilke)

Dans Annotations se trouvent les fichiers _.xml_ d'annotations
Dans JPEGImages se trouvent les images d'extension _.jpg_
Dans ImageSets/Main se trouvent le découpage en train, validation, test set

```bash
le/nom/du/dossier/racine
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

```
python src/CLI_data.py rename 

#exemple

python src/CLI_data.py rename --path='path/du/dossier/images' --start=0 --is_ok=True

Après avoir renommer les images, il faut placer les images dans *VOC2021/JPEGImages*.
```

### L'Annotation

L'annotation peut se faire avec le module open-source labelImg (https://github.com/tzutalin/labelImg) en sauvegardant les annotations sous xml (PASCAL VOC format).
Les fichiers d'annotation sont à placer dans _VOC2021/Annotations_

### Modifier les fichiers de découpage

Ensuite il faut ajouter les noms (sans l'extension) aux fichiers Main/_.txt_ soit dans le train.txt (pour compléter les données d'entraînement), soit dans le val.txt (pour compléter les données de validation), soit le test.txt (pour compléter les données de test)

## Comment entraîné un modèle à partir du modèle pré entraîné fourni par gluoncv ?

Pour plus d'informations sur les arguments des CLI, voir **CLI_detector.py**

```
python src/CLI_detector.py train_from_pretrained

#exemple
python src/CLI_detector.py train_from_pretrained --save_prefix='save_name_prefix' --data_path='Data/EAF_real' --batch_size=10 --epoch=15  --train_result_folder='results_train/' --log_foler='logs/' --model_folder='src/models/'

```

En sortie:
Sauvegarde les courbes d'entraînement dans le dossier indiqué en paramètre

## Comment entraîné un modèle à partir d'un modèle en local ?

Pour plus d'informations sur les arguments des CLI, voir **CLI_detector.py**

```

python src/CLI_detector.py train_from_finetuned 

#exemple 
python src/CLI_detector.py train_from_finetuned --save_prefix='save_name_prefix' --data_path='Data/EAF_real' --model_path='src/models/fake_best.params' --batch_size=10 --epoch=15  --train_result_folder='src/results_train/' --log_foler='src/logs/' --model_folder='src/models/'

```

En sortie:
Sauvegarde les courbes et les logs d'entraînement dans les dossiers indiqués en paramètre

## Comment évaluer un modèle ?

```
python src/CLI_detector.py eval (évaluation d'un modèle en fixant un taux de faux positif)


#exemple

python src/CLI_detector.py eval --data_path_test='Data/EAF_real' --save_prefix='fake400_19style+real_on_real' --model_path='src/models/fake400_7style+real_best.params' --taux_fp=0.143 --results_folder='src/results_ROC/' --log_folder='src/logs/'
```

En sortie:
Sauvegarde les logs (matrice de confusion) et résultat d'évaluation (ROC_curves) dans les dossiers indiqués en paramètre.

## Comment faire une prédiction sur une image ?

```
python src/CLI_detector.py predict (faire une prédiction sur une image)

#exemple
python src/CLI_detector.py predict --model_name='src/models/fake400_7style+real_best.params' --input_path='inputs/EAF3.jpg' --output_folder='outputs/' --thresh=0.3

```

En sortie:
Sauvegarde l'image de la prédiction dans le dossier outpute renseigné en paramètre

# Conformité
Mise à disposition d'une CLI qui permet d'afficher les résultats au sujet de l'évaluation de la conformité pour une photographie de chantier donnée.
Des exemples de commandes sont:
```
#visulaiser les résulats d'évaluation de conformité pour l'image située à image_path
python CLI_conformity.py image_path

#suavegarder l'image dans un fichier json
python CLI_conformity.py --save image_path
```

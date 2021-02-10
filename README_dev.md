Voici le repository pour le server du projet DTY du Paris Digital Lab pour Sonergia contenant le cas d'usage de détection de l'écart au feu et de sa conformité.

# Test

Avant d'effectuer le test :

- mettre dans tests/test_data/models le modèle.

```
python -m unittest discover -s tests
```

# Entrainement

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
python CLI_data.py rename

#exemple

python CLI_data.py rename --path='path/du/dossier/images' start=0 is_ok=True

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
python CLI_detector.py train_from_pretrained

#exemple
python CLI_detector.py train_from_pretrained --save_prefix='save_name_prefix' --data_path='../Data/EAF_real'  --batch_size=10 --epoch=15 --results_folder='results_ROC/' --log_foler='logs/'

```

En sortie:
Sauvegarde les courbes d'entraînement dans le dossier indiqué en paramètre

## Comment entraîné un modèle à partir d'un modèle en local ?

Pour plus d'informations sur les arguments des CLI, voir **CLI_detector.py**

```

python CLI_detector.py train_from_finetuned

#exemple
python CLI_detector.py train_from_finetuned --save_prefix='save_name_prefix' --data_path='../Data/EAF_real' --model_path='models/model_name_best.params' --batch_size=10 --epoch=15  --results_folder='results_ROC/' --log_foler='logs/'

```

En sortie:
Sauvegarde les courbes et les logs d'entraînement dans les dossiers indiqués en paramètre

## Comment évaluer un modèle ?

```
python CLI_detector.py eval (évaluation d'un modèle en fixant un taux de faux positif)


#exemple

python CLI_detector.py eval --data_path_test='../Data/EAF_real' --save_prefix='fake400_19style+real_on_real' --model_name='models/path/to/model' --taux_fp=0.143
```

En sortie:
Sauvegarde les logs (matrice de confusion) et résultat d'évaluation (ROC_curves) dans les dossiers indiqués en paramètre.

## Comment faire une prédiction sur une image ?

```
python CLI_detector.py predict (faire une prédiction sur une image)

#exemple
python CLI_detector.py predict model_name='models/ssd_512_best.params' input_path='inputs/EAF3.jpg' output_folder='outputs/' thresh=0.3 data_path_test='../Data/EAF_real'

```

En sortie:
Sauvegarde l'image de la prédiction dans le dossier outpute renseigné en paramètre

```

# La conformité
La classe Conformity permet de créer un objet qui permet d'accéder aux éléments de conformité de l'eaf dans un chantier.
On peut créer cet objet comme suit:
```

my_conformity = Conformity(imagePath, tolerance)

```
Par défaut, **tolerance==0.02**

On peut accéder aux informations de conformité en appelant la méthode get_conformity() de cette classe:
```

my_conformity.get_conformity()

```

L'objet de conformité est composé de différents sous objets qui décrivent un chantier: un objet pour la réglette, un objet pour la protection et un objet pour l'image elle même.

## L'objet de réglette
On peut le créer comme suit:
```

my_ruler = Ruler(imagePath)

```

Cet objet offre des accès publiques à des éléments décrivants la réglette ou sa conformité. En voici quelques exemples:
```

#Accès aux graduations
my_ruler.get_digits()

#Accès à la conformité de l'inclinaison de la réglette par rapport au sol
my_ruler.check_inclinaison_conformity(tolerance)

#Accès à l'axe de la réglette
my_ruler.get_axis()

```

## L'objet de la protection
On peut le créer comme suit:
```

my_protection = Protection(imagePath)

```

Cet objet offre des accès publiques à des éléments décrivants la protection ou sa conformité. En voici quelques exemples:
```

#Accès à l'axe de la protection
my_protection.get_axis_from_edges()

#Accès la conformité de la protection (protection détectée ou pas)
my_protection.check_protection()

```

## L'objet d'Image
Cet objet permet d'avoir une version de l'image correspondant à celle que prend la pipeline de conformité en entrée.
On peut créer cet objet comme suit:
```

my_image = Image(imagePath)

```
Cet objet offre des accès à des formes transformées de l'image initiale comme:
```

#Accès à la versions B&W
my_image.gray_scale()

#Accès à l'image des bords
my_image.edges_detection()

#Accès à l'image des bords dilatés
my_image.dilate(iteration)

```

```

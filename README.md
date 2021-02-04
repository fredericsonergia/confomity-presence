/Applications/Blender.app/Contents/Resources/2.91/python/bin/python3.7m -m ensurepip
/Applications/Blender.app/Contents/Resources/2.91/python/bin/python3.7m -m pip install -U pip
/Applications/Blender.app/Contents/Resources/2.91/python/bin/python3.7m -m pip install lxml
/Applications/Blender.app/Contents/Resources/2.91/python/bin/python3.7m -m pip install opencv-python

# generation d'image

```
blender --background --python generate_images.py
```

blender -b --python generate_images.py -- -a generate_train -y 3 -n 3

# style

python lib/style.py -c "/path/to/contentimages" -s "/path/to/styleimages" -o "/path/to/output"

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

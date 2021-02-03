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

## Données
```bash
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

python CLI.py train
python CLI.py eval
python CLI.py predict

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
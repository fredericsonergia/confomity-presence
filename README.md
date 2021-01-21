# environnement
## create
conda env create -f environment.yml
conda activate presence
## update
conda env export > environment.yml

# Entrainement
## Données 

└── VOC2021
    ├── Annotations
    ├── ImageSets
    │   └── Main
    │       ├── test.txt
    │       └── trainval.txt
    └── JPEGImages

# API
python app.py

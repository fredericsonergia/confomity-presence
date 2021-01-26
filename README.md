blender --background --python generate_images.py

/Applications/Blender.app/Contents/Resources/2.91/python/bin/python3.7m -m ensurepip
/Applications/Blender.app/Contents/Resources/2.91/python/bin/python3.7m -m pip install -U pip
/Applications/Blender.app/Contents/Resources/2.91/python/bin/python3.7m -m pip install lxml
/Applications/Blender.app/Contents/Resources/2.91/python/bin/python3.7m -m pip install opencv-python

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
│ └── Main
│ ├── test.txt
│ └── trainval.txt
└── JPEGImages

# API

python app.py

# CLI

python CLI.py train
python CLI.py eval
python CLI.py predict

blender --background --python generate_images.py
source ~/.bash_profile 
conda activate presence
cd src
python CLI.py train_from_scratch --root='../EAF' --save_prefix='false_test'

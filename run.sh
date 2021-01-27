#blender --background --python generate_images.py
source ~/.bash_profile 
conda activate presence
cd src
python CLI.py train_from_pretrained --data_path='../Data/EAF_false' --save_prefix='fake' --batch_size=20
python CLI.py eval --data_path_test='../Data/EAF_real' --save_prefix='fake' --model_name='models/fake_best.params'
# python CLI.py eval --data_path_test='../Data/EAF_real' --save_prefix='real' --model_name='models/ssd_512_best.params'
# python CLI.py eval --data_path_test='../Data/EAF_test' --save_prefix='real' --model_name='models/ssd_512_best.params'
python CLI.py train_from_finetuned --save_prefix='fake+real' --data_path='../Data/EAF_real' --model_name='models/fake_best.params' --batch_size=10
python CLI.py eval --data_path_test='../Data/EAF_real' --save_prefix='real+fake' --model_name='models/fake+real_best.params'

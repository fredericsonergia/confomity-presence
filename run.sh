blender --background --python generate_images.py
source ~/.bash_profile 
conda activate presence
cd src
python CLI.py train_from_scratch --root='../Data/EAF_false' --save_prefix='fake'
python CLI.py eval --data_path_test='../Data/EAF_real' --save_prefix='fake' --model_name='models/fake_best.params'
python CLI.py eval --data_path_test='../Data/EAF_real' --save_prefix='real' --model_name='models/ssd_512_best.params'
python CLI.py train --save_prefix='fake_finetuned_with_real' --root='../Data/EAF_real' --model_name='models/false_test_best.params'
python CLI.py eval --data_path_test='../Data/EAF_real' --save_prefix='real' --model_name='models/ssd_512_best.params'
python CLI.py eval --data_path_test='../Data/EAF_real' --save_prefix='real+fake' --model_name='models/fake_finetuned_with_real_best.params'

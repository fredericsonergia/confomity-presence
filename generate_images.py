import time
import sys
import os
import random
import argparse
import glob
sys.path.append("./src")
from detector_utils.VOC_form import sorted_alphanumeric

from pathlib import Path
sys.path.append("./blender")
sys.path.append("./label_img")

from Scene import SceneFactory
from annotate_images import create_annotation_file
from visualisation import create_visualisation


# parser = argparse.ArgumentParser(
#     description="evaluate a model"
# )

# parser.add_argument(
#     "--number-ok", default=10, dest="number_ok", help="the number of ok images"
# )

# parser.add_argument(
#     "--number-ko", default=10, dest="number_ko", help="the number of ko images"
# )

# args = parser.parse_args()

def generate_image(image_folder_path, filename):
    image_path = image_folder_path + "/" + filename + ".jpg"
    s = SceneFactory().createRandomScene()
    s.clear()
    s.generate_scene()
    s.prepare_camera()
    shapes = s.annotate()
    s.render(image_path)
    print("annotation")
    create_annotation_file(shapes, filename, image_path)


def generate_ok_image(image_folder_path, filename, annot_path):
    image_path = image_folder_path + "/" + filename + ".jpg"
    s = SceneFactory().createRandomSceneWithProtection()
    s.clear()
    s.generate_scene()
    s.prepare_camera()
    shapes = s.annotate()
    s.render(image_path)
    print("annotation")
    create_annotation_file(shapes, filename, image_path, annot_path)


def generate_ko_image(image_folder_path, filename, annot_path):
    image_path = image_folder_path + "/" + filename + ".jpg"
    s = SceneFactory().createRandomSceneNoProtection()
    s.clear()
    s.generate_scene()
    s.prepare_camera()
    shapes = s.annotate()
    s.render(image_path)
    print("annotation")
    create_annotation_file(shapes, filename, image_path, annot_path)


def generate_set(number_of_ok, number_of_ko, image_folder_path, annot_path, file_name_template, start=0):
    for i in range(number_of_ok):
        generate_ok_image(image_folder_path, file_name_template + "_ok_" + str(start+i), annot_path)
    for i in range(number_of_ko):
        generate_ko_image(image_folder_path, file_name_template + "_ko_" + str(start+i), annot_path)

def write_txt(filename, txt_path, files):
    if os.path.exists(txt_path+filename):
        os.remove(txt_path+filename)
    with open(txt_path+filename, 'w') as f:
        for path in files:
            filename, _ = os.path.splitext(path)
            f.write(filename+'\n')
        f.close()

def create_Sets(image_folder_path, txt_path, proportion_val):
    l_path = os.listdir(image_folder_path)
    lr_path = random.sample(l_path,len(l_path))
    train_files = lr_path[: int(proportion_val*len(lr_path))]
    val_files = lr_path[int(proportion_val*len(lr_path))+1 :]
    write_txt('train.txt', txt_path, train_files)
    write_txt('val.txt', txt_path, val_files)


def delete_files(root_name, path):
    files = os.listdir(root_name + path)
    for f in files:
        os.remove(os.path.join(root_name + path, f))

def delete_all_files(root_name):
    delete_files(root_name, '/VOC2021/Annotations')
    delete_files(root_name, '/VOC2021/JPEGImages')
    delete_files(root_name, '/VOC2021/ImageSets/Main')

def generate_train_folder(root_name, number_of_ok, number_of_ko, image_folder_path, annot_path, file_name_template):
    generate_set(number_of_ok, number_of_ko, image_folder_path, annot_path, "EAF")

class Datagenerator:
    def __init__(self, root_name, start=0, filename='EAF'):
        self.root_name = root_name
        self.filename_template = filename
        self.start = start
        self.image_folder_path = self.root_name + '/VOC2021' + '/JPEGImages/'
        self.annot_path =  self.root_name+ '/VOC2021' +'/Annotations/'
        self.txt_path = self.root_name + '/VOC2021' +'/ImageSets/Main/'

    def create_folder(self):
        ''' create the VOC like folder from the root_name'''
        Path(self.root_name).mkdir(parents=True, exist_ok=True)
        Path(self.root_name + '/VOC2021/').mkdir(parents=True, exist_ok=True)
        Path(self.image_folder_path).mkdir(parents=True, exist_ok=True)
        Path(self.annot_path).mkdir(parents=True, exist_ok=True)
        Path(self.root_name+'/VOC2021/ImageSets/').mkdir(parents=True, exist_ok=True)
        Path(self.txt_path).mkdir(parents=True, exist_ok=True)

    def create_train_sets(self, proportion_val):
        '''
        create the text file spliting training and val
        '''
        l_path = os.listdir(self.image_folder_path)
        lr_path = random.sample(l_path,len(l_path))
        val_files = lr_path[: round(proportion_val*len(lr_path))]
        train_files = lr_path[round(proportion_val*len(lr_path)):]
        delete_files(self.root_name, '/VOC2021/ImageSets/Main')
        write_txt('train.txt', self.txt_path, train_files)
        write_txt('val.txt', self.txt_path, val_files)
    
    def create_test_set(self):
        '''
        create the text file
        '''
        test_files = os.listdir(self.image_folder_path)
        stest_files = sorted_alphanumeric(test_files)
        delete_files(self.root_name, '/VOC2021/ImageSets/Main')
        write_txt('test.txt', self.txt_path, test_files)

    def generate_folder(self, number_of_ok, number_of_ko):
        generate_set(number_of_ok, number_of_ko, self.image_folder_path, self.annot_path, self.filename_template, self.start)

def test_generation():
    '''
    generate a test set
    '''
    test_gen = Datagenerator('./Data/EAF_test', 20)
    test_gen.create_folder()
    test_gen.generate_folder(30,30)
    test_gen.create_test_set()

def train_generation(number_ok, number_ko, start):
    '''
    generate images for training
    Args:
    - number_ok (int): the number of ok images.
    - number_ko (int): the number of ko images.
    - start (int): the number from which we name the files.
    '''
    gen = Datagenerator('./Data/EAF_false400', start)
    gen.create_folder()
    gen.generate_folder(number_ok, number_ko)
    gen.create_train_sets(0.3)

if __name__ == "__main__":
    start = time.time()
    #test_generation()
    #delete_all_files('./Data/EAF_false')
    train_generation(200, 200, 0)
    #generate_set(3, 3, "./Images", './Annotations/',"test_set")
    #create_visualisation()
    #generate_ok_image("./Images", "test_set4", "./Annotations/")
    #create_visualisation()
    end = time.time()
    print("the generation took " + str(end - start) + " seconds")

import os 
import re


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def rename_img(path, name):

    img_path = os.getcwd()  + path + '/Images/'
    PATH_KO_LIST = sorted_alphanumeric(os.listdir(img_path))
    for index, path_ko in enumerate(PATH_KO_LIST):
        GENERIC_KO = name
        os.rename(img_path+path_ko, img_path+GENERIC_KO+str(index)+'.jpg')


def add_text(path):
    txt_path = os.getcwd() + path + '/ImagesSets/'
    try:
        os.listdir(txt_path)
    except:
        os.mkdir(txt_path)
    if os.path.exists(txt_path+'trainval.txt'):
        os.remove(txt_path+'trainval.txt')
    img_path = os.getcwd()  + path + '/Images/'
    PATH_KO_LIST = sorted_alphanumeric(os.listdir(img_path))
    with open(txt_path+'trainval.txt', 'w') as f:
        for path in PATH_KO_LIST:
            filename, _ = os.path.splitext(path)
            f.write(filename+'\n')
        f.close()
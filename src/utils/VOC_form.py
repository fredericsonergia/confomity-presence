import os 
import re


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


def rename_img(path, name, start=1):
    path_list = sorted_alphanumeric(os.listdir(path))
    for index, img_path in enumerate(path_list):
        os.rename(path+img_path, path+name+str(index+int(start))+'.jpg')


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
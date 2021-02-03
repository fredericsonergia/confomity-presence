import os 
import re


def sorted_alphanumeric(data):
    '''
    sort data with alphanumeric order
    Args:
    - data (list of str): data to sort.
    '''
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


def rename_img(path, start=1, is_ok=True):
    '''
    rename images from a given path with KO or OK name 
    Arg
    '''
    path_list = sorted_alphanumeric(os.listdir(path))
    if is_ok:
        for index, img_path in enumerate(path_list):
            os.rename(path+img_path, path+'EAF_OK'+str(index+int(start))+'.jpg')
    else:
        for index, img_path in enumerate(path_list):
            os.rename(path+img_path, path+'EAF_KO'+str(index+int(start))+'.jpg')


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
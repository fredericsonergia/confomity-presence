import time
import sys

sys.path.append("./blender")
sys.path.append("./label_img")

for path in sys.path:
    print(path)

from Scene import SceneFactory
from annotate_images import create_annotation_file


def generate_image(path, has_protection=True):
    s = SceneFactory().createRandomScene()
    s.clear()
    if has_protection:
        s.generate_scene_EAF()
    else:
        s.generate_scene_no_EAF()
    s.prepare_camera()
    s.render(path)


def generate_image_and_annotation(path, annot_filename, has_protection=True):
    s = SceneFactory().createRandomScene()
    s.clear()
    if has_protection:
        s.generate_scene_EAF()
    else:
        s.generate_scene_no_EAF()
    s.prepare_camera()
    shapes = s.annotate()
    s.render(path)
    create_annotation_file(shapes, annot_filename, path)


def generate_set(number_of_ok, number_of_ko, path, file_name_template):
    for i in range(number_of_ok):
        generate_image(path + file_name_template + "ok_" + str(i) + ".jpg")
    for i in range(number_of_ko):
        generate_image(path + file_name_template + "ko_" + str(i) + ".jpg", False)


if __name__ == "__main__":
    start = time.time()
    # generate_image("/Users/matthieu/Documents/Project3/presence/Images/rectangle.jpg")
    # generate_image_and_annotation("./Images/rectangle.jpg", "rectangle")
    end = time.time()
    print("the generation took " + str(end - start) + " seconds")

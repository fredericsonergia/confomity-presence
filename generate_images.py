import time
import sys

sys.path.append("./blender")
sys.path.append("./label_img")

from Scene import SceneFactory
from annotate_images import create_annotation_file
from visualisation import create_visualisation


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


def generate_ok_image(image_folder_path, filename):
    image_path = image_folder_path + "/" + filename + ".jpg"
    s = SceneFactory().createRandomSceneWithProtection()
    s.clear()
    s.generate_scene()
    s.prepare_camera()
    shapes = s.annotate()
    s.render(image_path)
    print("annotation")
    create_annotation_file(shapes, filename, image_path)


def generate_ko_image(image_folder_path, filename):
    image_path = image_folder_path + "/" + filename + ".jpg"
    s = SceneFactory().createRandomSceneNoProtection()
    s.clear()
    s.generate_scene()
    s.prepare_camera()
    shapes = s.annotate()
    s.render(image_path)
    print("annotation")
    create_annotation_file(shapes, filename, image_path)


def generate_set(number_of_ok, number_of_ko, image_folder_path, file_name_template):
    for i in range(number_of_ok):
        generate_ok_image(image_folder_path, file_name_template + "_ok_" + str(i))
    for i in range(number_of_ko):
        generate_ko_image(image_folder_path, file_name_template + "_ko_" + str(i))


if __name__ == "__main__":
    start = time.time()
    generate_ok_image("./Images", "test_set")
    create_visualisation()
    end = time.time()
    print("the generation took " + str(end - start) + " seconds")

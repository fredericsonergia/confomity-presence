import time
import sys

sys.path.append("./blender")

from Scene import Scene  # pylint: disable=import-error


def generate_image(path, has_protection=True):
    s = Scene()
    s.clear()
    if has_protection:
        s.generate_scene_EAF()
    else:
        s.generate_scene_no_EAF()
    s.prepare_camera()
    s.get_annotation_chimney()
    s.render(path)


def generate_set(number_of_ok, number_of_ko, path, file_name_template):
    for i in range(number_of_ok):
        generate_image(path + file_name_template + "ok_" + str(i) + ".jpg")
    for i in range(number_of_ko):
        generate_image(path + file_name_template + "ko_" + str(i) + ".jpg", False)


if __name__ == "__main__":
    start = time.time()
    generate_set(1, 1, "/Users/matthieu/Documents/Project3/image_generator/", "image_")
    end = time.time()
    print("the generation took " + str(end - start) + " seconds")

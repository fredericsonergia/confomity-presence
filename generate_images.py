import time
import sys

sys.path.append("./blender")

from Scene import Random_Scene  # pylint: disable=import-error


def generate_image(path, has_protection=True):
    s = Random_Scene()
    s.clear()
    if has_protection:
        s.generate_scene_EAF()
    else:
        s.generate_scene_no_EAF()
    s.prepare_camera()
    s.get_annotation_chimney()
    s.render(path)


if __name__ == "__main__":
    start = time.time()
    generate_image(
        "/Users/matthieu/Documents/Project3/image_generator/testing_camera.jpg"
    )
    end = time.time()
    print("the generation took " + str(end - start) + " seconds")

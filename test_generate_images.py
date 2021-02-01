import sys

sys.path.append("./blender")
sys.path.append("./blender_test")

from utils_test import test_get_area, test_hsv_to_rgb


if __name__ == "__main__":
    print("----------------")
    print("testing")
    test_get_area()
    test_hsv_to_rgb()
    print("succes")

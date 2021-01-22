import sys

sys.path.append("./blender")
sys.path.append("./blender_test")

from utils_test import test_get_area  # pylint: disable=import-error


if __name__ == "__main__":
    print("----------------")
    print("testing")
    test_get_area()
    print("succes")

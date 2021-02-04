import sys
import mathutils
import bpy

sys.path.append("./blender")

import utils as b  # pylint: disable=import-error


def test_get_area():
    assert b.get_area((0, 0), (1, 2)) == 2


def test_hsv_to_rgb():
    assert b.hsv_to_rgb(359.0 / 360.0, 1, 1) == (255, 0, 4)


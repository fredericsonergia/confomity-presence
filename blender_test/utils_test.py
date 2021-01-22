import sys
import mathutils
import bpy

sys.path.append("./blender")

import utils as b  # pylint: disable=import-error


def test_get_area():
    assert b.get_area((0, 0), (1, 2)) == 2


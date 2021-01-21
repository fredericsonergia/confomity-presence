import sys
import mathutils
import bpy

sys.path.append("./blender")

import utils as b  # pylint: disable=import-error
from Scene import Scene  # pylint: disable=import-error


def test_get_area():
    assert b.get_area((0, 0), (0, 2), (1, 0)) == 2


def test_get_max_rec():
    s = Scene((1, 2), (0.5, 0.5, 5))
    s.clear()
    s.generate_scene_EAF()
    s.prepare_camera()

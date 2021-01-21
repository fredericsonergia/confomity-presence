import sys
import mathutils

sys.path.append("./blender")

from Chimney import Chimney  # pylint: disable=import-error


def test_chimney():
    c = Chimney(size=(1, 2), loc=(0.5, 0.5, 5))
    assert len(c.get_box()) == 8
    assert c.get_box() == [
        mathutils.Vector((2, 3, 0)),
        mathutils.Vector((2, 0, 0)),
        mathutils.Vector((0, 0, 0)),
        mathutils.Vector((0, 3, 0)),
        mathutils.Vector((2, 3, 10)),
        mathutils.Vector((2, 0, 10)),
        mathutils.Vector((0, 0, 10)),
        mathutils.Vector((0, 3, 10)),
    ]

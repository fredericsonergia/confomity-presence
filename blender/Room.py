import bpy


class Ground:
    def __init__(self):
        self.size = 20

    def draw(self):
        bpy.ops.mesh.primitive_cube_add(size=self.size, location=(0, 0, self.size / 2))

    def get_object(self):
        return bpy.data.objects["Cube"]

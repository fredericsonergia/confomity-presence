import random
import mathutils
import bpy


class ChimneyFactory:
    def createRandomChimney(self):
        random_type = random.choice(["cylinder", "cuboid"])
        if random_type == "cuboid":
            return CubicChimney()
        elif random_type == "cylinder":
            return RoundChimney()

    def createRoundChimney(self, size, loc, vertices):
        return RoundChimney(size, loc, vertices)

    def createCubicChimney(self, size, loc):
        return CubicChimney(size, loc)


class CubicChimney:
    def __init__(
        self,
        size=(random.random() * 2.8 + 1.5, random.random() * 2.8 + 1.5),
        loc=(
            (-1) ** random.randint(1, 3) * random.random() / 4,
            (-1) ** random.randint(1, 3) * random.random() / 4,
            3,
        ),
    ):
        self.size = size
        self.loc = loc

    def draw(self):
        bpy.ops.mesh.primitive_cube_add(size=1, location=self.loc)
        obj = bpy.data.objects["Cube.001"]
        obj.scale[0] = self.size[0]
        obj.scale[1] = self.size[1]
        obj.scale[2] = 6

    def get_box(self):
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.transform_apply(scale=True)
        bpy.ops.object.select_all(action="DESELECT")
        obj = bpy.data.objects["Cube.001"]
        points = [vert.co for vert in obj.data.vertices]
        for point in points:
            if point.z < 0:
                point.z = 0
        return points.copy()

    def get_object(self):
        return bpy.data.objects["Cube.001"]


class RoundChimney:
    def __init__(
        self,
        size=(random.random() * 2.8 + 1.5, random.random() * 2.8 + 1.5),
        loc=(
            (-1) ** random.randint(1, 3) * random.random() / 4,
            (-1) ** random.randint(1, 3) * random.random() / 4,
            3,
        ),
        vertices=random.randint(3, 65),
    ):
        self.size = size
        self.loc = loc
        self.vertices = vertices

    def draw(self):
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=self.vertices, radius=0.5, location=self.loc
        )
        obj = bpy.data.objects["Cylinder"]
        obj.scale[0] = self.size[0]
        obj.scale[1] = self.size[1]
        obj.scale[2] = 3

    def get_box(self):
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.transform_apply(scale=True)
        bpy.ops.object.select_all(action="DESELECT")
        obj = bpy.data.objects["Cylinder"]
        points = [vert.co for vert in obj.data.vertices]
        for point in points:
            if point.z < 0:
                point.z = 0
        return points

    def get_object(self):
        return bpy.data.objects["Cylinder"]

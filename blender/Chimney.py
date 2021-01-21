import random
import mathutils
import bpy


class Chimney:
    def __init__(
        self,
        size=(random.random() * 2.8 + 1.5, random.random() * 2.8 + 1.5),
        loc=(
            (-1) ** random.randint(1, 3) * random.random() / 4,
            (-1) ** random.randint(1, 3) * random.random() / 4,
            5,
        ),
    ):
        self.size = size
        self.loc = loc

    def draw(self):
        random_type = random.choice(["cylinder", "cuboid"])
        if random_type == "cuboid":
            bpy.ops.mesh.primitive_cube_add(size=1, location=self.loc)
            bpy.context.object.scale[0] = self.size[0]
            bpy.context.object.scale[1] = self.size[1]
            bpy.context.object.scale[2] = 10
        elif random_type == "cylinder":
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=random.randint(3, 65), radius=0.5, location=self.loc
            )
            bpy.context.object.scale[0] = self.size[0]
            bpy.context.object.scale[1] = self.size[1]
            bpy.context.object.scale[2] = 6

    def get_box(self, show=False):
        points = [
            mathutils.Vector((0.5, 0.5, 0)),
            mathutils.Vector((0.5, -0.5, 0)),
            mathutils.Vector((-0.5, -0.5, 0)),
            mathutils.Vector((-0.5, 0.5, 0)),
        ]
        for point in points:
            point += mathutils.Vector((self.loc[0], self.loc[1], 0))
        if self.size[0] > 0:
            points[0] += mathutils.Vector((self.size[0], 0, 0))
            points[1] += mathutils.Vector((self.size[0], 0, 0))
        if self.size[0] < 0:
            points[2] += mathutils.Vector((self.size[0], 0, 0))
            points[3] += mathutils.Vector((self.size[0], 0, 0))
        if self.size[1] > 0:
            points[0] += mathutils.Vector((0, self.size[1], 0))
            points[3] += mathutils.Vector((0, self.size[1], 0))
        if self.size[1] < 0:
            points[1] += mathutils.Vector((0, self.size[1], 0))
            points[2] += mathutils.Vector((0, self.size[1], 0))
        for i in range(len(points)):
            points.append(points[i] + mathutils.Vector((0, 0, 10)))

        if show:
            for point in points:
                bpy.ops.mesh.primitive_monkey_add(size=0.1, location=point)
        return points


class CubicChimney(Chimney):
    def draw(self):
        bpy.ops.mesh.primitive_cube_add(size=1, location=self.loc)
        bpy.context.object.scale[0] = self.size[0]
        bpy.context.object.scale[1] = self.size[1]
        bpy.context.object.scale[2] = 10


class RoundChimney(Chimney):
    def __init__(
        self,
        size=(random.random() * 2.8 + 1.5, random.random() * 2.8 + 1.5),
        loc=(
            (-1) ** random.randint(1, 3) * random.random() / 4,
            (-1) ** random.randint(1, 3) * random.random() / 4,
        ),
        vertices=random.randint(3, 65),
    ):
        super().__init__(size, loc)
        self.vertices = vertices

    def draw(self):
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=self.vertices, radius=0.5, location=self.loc
        )
        bpy.context.object.scale[0] = self.size[0]
        bpy.context.object.scale[1] = self.size[1]
        bpy.context.object.scale[2] = 6


#     def color(self):
# bpy.ops.material.new()
# bpy.data.materials["Material"].node_tree.nodes["Principled BSDF"].inputs[
#     0
# ].default_value = (
#     random.randint(0, 256),
#     random.randint(0, 256),
#     random.randint(0, 256),
#     random.random(),
# )

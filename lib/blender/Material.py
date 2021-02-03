import random
import bpy

from utils import hsv_to_rgb


def brown():
    (r, g, b) = hsv_to_rgb(
        random.randint(32, 36) / 360,
        random.randint(83, 88) / 100,
        0.2
        # random.randint(20, 60) / 100,
    )
    print("rgb", (r, g, b))
    return [r / 255, g / 255, b / 255]


class MaterialFactory:
    def create_random_color(self, name):
        return Material(name, [random.random(), random.random(), random.random()])

    def create_random_brown(self, name):
        return Material(name, brown())

    def create_color(self, name, rgb):
        return Material(name, rgb)


class Material:
    def __init__(self, name, rgb):
        print("constructed")
        self.name = name
        self.rgb = rgb
        print(rgb)

    def create_mat(self):
        mat = bpy.data.materials.get(self.name)
        if mat is None:
            mat = bpy.data.materials.new(name=self.name)
        mat.use_nodes = True
        mat.node_tree.nodes.get("Principled BSDF").inputs[0].default_value = (
            self.rgb[0],
            self.rgb[1],
            self.rgb[2],
            1,
        )
        return mat

    def add_to_object(self, ob):
        mat = self.create_mat()
        if ob.data.materials:
            ob.data.materials[0] = mat
        else:
            ob.data.materials.append(mat)
        ob.active_material = mat


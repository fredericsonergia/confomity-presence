import random
import bpy


class MaterialFactory:
    def create_random_color(self):
        return Material(
            "color",
            "Anisotropic BSDF",
            [random.random(), random.random(), random.random()],
        )


class Material:
    def __init__(self, name, node_type, rgb):
        self.name = name
        self.node_type = node_type
        self.rgb = rgb

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


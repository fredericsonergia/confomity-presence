import bpy
import random


class RandomProtection:
    def __init__(self):
        self.number_of_edges = 4
        self.verts = []
        self.faces = []
        self.base_height = random.random() * 2 + 1
        self.thckness = random.random() * 0.2

        for i in range(self.number_of_edges):
            height = self.base_height + random.random()
            if i % 4 == 0:
                x = -(random.random() * 0.5 + 2.5)
                y = -(random.random() * 0.5 + 2.5)
                self.verts.append([x, y, 0.0])
                self.verts.append([x, y, height])
            elif i % 4 == 1:
                y = -(random.random() * 0.5 + 2.5)
                x = random.random() * 0.5 + 2.5
                self.verts.append([x, y, 0.0])
                self.verts.append([x, y, height])
            elif i % 4 == 2:
                x = random.random() * 0.5 + 2.5
                y = random.random() * 0.5 + 2.5
                self.verts.append([x, y, 0.0])
                self.verts.append([x, y, height])
            elif i % 4 == 3:
                y = random.random() * 0.5 + 2.5
                x = -(random.random() * 0.5 + 2.5)
                self.verts.append([x, y, 0.0])
                self.verts.append([x, y, height])
        n = len(self.verts)
        for i in range(0, n, 2):
            self.faces.append([i % n, (i + 1) % n, (i + 3) % n, (i + 2) % n])

    def draw(self):
        name = "Protection"
        mesh = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(name, mesh)
        col = bpy.data.collections[0]
        col.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        mesh.from_pydata(self.verts, [], self.faces)
        bpy.ops.object.modifier_add(type="SOLIDIFY")
        bpy.context.object.modifiers["Solidify"].thickness = 0.11

    def get_vert(self):
        return self.verts

    def get_object(self):
        return bpy.data.objects["Protection"]

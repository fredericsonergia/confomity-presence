import bpy
import bpy_extras
import mathutils
import random

from Room import Ground
from Protection import RandomProtection
from Light import Light
from Chimney import ChimneyFactory
from Camera import Camera

from utils import get_annot_chimney, get_annot_protection


class SceneFactory:
    def createRandomScene(self):
        random_type = random.choice(["pro", "no_pro"])
        if random_type == "pro":
            return SceneWithProtection()
        else:
            return Scene()

    def createRandomSceneNoProtection(self):
        return Scene()

    def createRandomSceneWithProtection(self):
        return SceneWithProtection()


class Scene:
    def __init__(self):
        self.ground = Ground()
        self.chimney = ChimneyFactory().createRandomChimney()
        self.light = Light()
        self.camera = Camera()

    def generate_scene(self):
        self.ground.draw()
        self.chimney.draw()
        self.light.draw()

    def prepare_camera(self):
        self.camera.place()
        self.camera.update()
        self.camera.setup_format()

    def render(self, filePath):
        self.camera.prepare_render(filePath)
        self.camera.render()

    def clear(self):
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        for block in bpy.data.meshes:
            if block.users == 0:
                bpy.data.meshes.remove(block)

        for block in bpy.data.materials:
            if block.users == 0:
                bpy.data.materials.remove(block)

        for block in bpy.data.textures:
            if block.users == 0:
                bpy.data.textures.remove(block)

        for block in bpy.data.images:
            if block.users == 0:
                bpy.data.images.remove(block)

        for block in bpy.data.cameras:
            if block.users == 0:
                bpy.data.cameras.remove(block)

    def get_annotation_chimney(self):
        scene = bpy.context.scene
        camera = bpy.data.objects["Camera"]
        box = self.chimney.get_box()
        result = get_annot_chimney(scene, camera, box)
        return result

    def annotate(self):
        points_cheminee = self.get_annotation_chimney()
        return ({"points": points_cheminee, "label": "cheminee", "difficult": 0},)


class SceneWithProtection(Scene):
    def __init__(self):
        super().__init__()
        self.protection = RandomProtection()

    def generate_scene(self):
        self.ground.draw()
        self.chimney.draw()
        self.protection.draw()
        self.light.draw()

    def get_annotation_protection(self):
        scene = bpy.context.scene
        camera = bpy.data.objects["Camera"]
        coors = self.protection.get_vert()
        box = []
        for coor in coors:
            box.append(mathutils.Vector(coor))
        result = get_annot_protection(scene, camera, box)
        return result

    def annotate(self):
        points_cheminee = self.get_annotation_chimney()
        points_eaf = self.get_annotation_protection()
        return (
            {"points": points_eaf, "label": "eaf", "difficult": 0},
            {"points": points_cheminee, "label": "cheminee", "difficult": 0},
        )

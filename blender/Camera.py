import random
import bpy
import mathutils
from utils import sph2cart


class Camera:
    def __init__(self):
        self.az = random.randint(0, 360)
        self.el = random.randint(30, 50)
        self.r = 10
        self.focus_point = mathutils.Vector(
            (random.random() / 2, random.random() / 2, random.random() / 2)
        )
        self.lens = random.randint(15, 30)

    def place(self):
        bpy.ops.object.camera_add(
            location=sph2cart(self.az, self.el, self.r), rotation=[0.6799, 0, 0.8254]
        )
        bpy.context.scene.camera = bpy.context.object
        bpy.context.object.data.lens = 18

    def update(self, distance=8.0):
        print(bpy.data.objects)
        try:
            camera = bpy.data.objects["Camera"]
        except:
            camera = bpy.data.objects[0]
        looking_direction = camera.location - self.focus_point
        rot_quat = looking_direction.to_track_quat("Z", "Y")
        camera.rotation_euler = rot_quat.to_euler()
        camera.location = rot_quat @ mathutils.Vector((0.0, 0.0, distance))

    def setup_format(self):
        print("setting up format")
        bpy.context.scene.render.engine = "CYCLES"
        bpy.context.scene.render.image_settings.color_mode = "RGB"
        bpy.context.scene.render.resolution_percentage = 25
        bpy.context.scene.render.image_settings.file_format = "JPEG"

    def prepare_render(self, path):
        print("preparing render")
        bpy.context.scene.render.filepath = path

    def render(self):
        bpy.ops.render.render(write_still=True)

import bpy
import random


class Light:
    def __init__(self):
        self.loc_x = random.random() * 6
        self.loc_y = random.random() * 6
        self.loc_z = random.random() + 9
        self.lamp_strength = random.randint(500, 2000)

    def draw(self):
        bpy.ops.object.light_add(location=(self.loc_x, self.loc_y, self.loc_z))
        bpy.context.object.data.energy = self.lamp_strength

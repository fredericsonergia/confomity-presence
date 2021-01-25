import numpy as np
import bpy
import bpy_extras
import mathutils
import math


def sph2cart(az, el, r):
    az = az * np.pi / 180
    el = el * np.pi / 180
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return [x, y, z]


def get_area(bottom, top):
    long = abs(top[1] - bottom[1])
    short = abs(bottom[0] - top[0])
    return long * short


def dichotomie_find_max_z(scene, camera, x, y):
    a = 0
    b = 5
    debut = a
    fin = b
    k = 1
    e = 0.0001
    # calcul de la longueur de [a,b]
    ecart = b - a
    while ecart > e:
        # calcul du milieu
        m = (debut + fin) / 2
        co = mathutils.Vector((x, y, m))
        top = bpy_extras.object_utils.world_to_camera_view(  # pylint: disable=assignment-from-no-return
            scene, camera, co
        ).y
        if top > k:
            # la solution est inférieure à m
            fin = m
        else:
            # la solution est supérieure à m
            debut = m
        ecart = fin - debut
    return m


def index_of_max_y(list):
    max = -float("inf")
    max_index = None
    for i, el in enumerate(list):
        if el[1] > max:
            max = el[1]
            max_index = i
    return max_index


def get_2d_coor(scene, camera, co):
    co_2d = bpy_extras.object_utils.world_to_camera_view(  # pylint: disable=assignment-from-no-return
        scene, camera, co
    )
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
        int(scene.render.resolution_x * render_scale),
        int(scene.render.resolution_y * render_scale),
    )
    return (
        round(co_2d.x * render_size[0]),
        round(co_2d.y * render_size[1]),
    )


def change_origin_from_bottom_left_to_top_left(coor, height):
    x, y = coor
    return (x, height - y)


def get_rec(scene, camera, box, keep_all=True):
    bottom_vects = []
    top_vects = []
    for i in range(0, len(box), 2):
        vector = box[i]
        bottom_vects.append(get_2d_coor(scene, camera, vector))
    for i in range(1, len(box), 2):
        vector = box[i]
        max_z = dichotomie_find_max_z(scene, camera, vector.x, vector.y)
        if vector.z > max_z:
            vector.z = max_z
        top_vects.append(get_2d_coor(scene, camera, vector))

    if not keep_all:
        for _ in range(len(top_vects) // 2):
            index = index_of_max_y(top_vects)
            top_vects.pop(index)
            bottom_vects.pop(index)

    min_x, min_y, max_x, max_y = float("inf"), float("inf"), 0, 0
    for point in bottom_vects:
        if point[0] > max_x:
            max_x = point[0]
        if point[0] < min_x:
            min_x = max(point[0], 0)
        if point[1] < min_y:
            min_y = max(point[1], 0)
        if point[1] > max_y:
            max_y = point[1]
    for point in top_vects:
        if point[0] > max_x:
            max_x = point[0]
        if point[0] < min_x:
            min_x = max(point[0], 0)
        if point[1] > max_y:
            max_y = point[1]
        if point[1] < min_y:
            min_y = max(point[1], 0)
    result = []
    for co in [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]:
        result.append(
            change_origin_from_bottom_left_to_top_left(
                co, scene.render.resolution_y * scene.render.resolution_percentage / 100
            )
        )
    return result

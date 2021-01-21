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


def generator():
    yield (0, 1)
    yield (0, 2)
    yield (0, 3)
    yield (1, 2)
    yield (1, 3)
    yield (2, 3)


def get_area(bottom, top):
    L = abs(top[1] - bottom[1])
    l = abs(bottom[0] - top[0])
    return l * L


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


def get_max_rec(scene, camera, box):
    bottom_vects = []
    top_vects = []
    for i in range(0, len(box), 2):
        vector = box[i]
        bottom_vects.append(get_2d_coor(scene, camera, vector))
    for i in range(1, len(box), 2):
        vector = box[i]
        max_z = dichotomie_find_max_z(scene, camera, vector.x, vector.y)
        vector.z = max_z
        top_vects.append(get_2d_coor(scene, camera, vector))
    min_x, min_y, max_x, max_y = float("inf"), float("inf"), 0, 0
    for point in bottom_vects:
        if point[0] > max_x:
            max_x = point[0]
        if point[0] < min_x:
            min_x = point[0]
        if point[1] < min_y:
            min_y = point[1]
    for point in top_vects:
        if point[0] > max_x:
            max_x = point[0]
        if point[0] < min_x:
            min_x = point[0]
        if point[1] > max_y:
            max_y = point[1]
    return [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]
    # for (i, j) in generator():
    #     area = get_area(bottom_vects[i], top_vects[j])
    #     if area > max:
    #         max = area
    #         max_rect = [
    #             bottom_vects[i],
    #             (bottom_vects[i][0], top_vects[j][1]),
    #             (top_vects[j][0], bottom_vects[i][1]),
    #             top_vects[j],
    # ]

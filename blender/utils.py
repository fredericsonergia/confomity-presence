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


def hsv_to_rgb(h, s, v):
    if s == 0.0:
        v *= 255
        return (v, v, v)
    i = int(h * 6.0)  # XXX assume int() truncates!
    f = (h * 6.0) - i
    p, q, t = (
        int(255 * (v * (1.0 - s))),
        int(255 * (v * (1.0 - s * f))),
        int(255 * (v * (1.0 - s * (1.0 - f)))),
    )
    v *= 255
    i %= 6
    if i == 0:
        return (v, t, p)
    if i == 1:
        return (q, v, p)
    if i == 2:
        return (p, v, t)
    if i == 3:
        return (p, q, v)
    if i == 4:
        return (t, p, v)
    if i == 5:
        return (v, p, q)


def get_area(bottom, top):
    long = abs(top[1] - bottom[1])
    short = abs(bottom[0] - top[0])
    return long * short


def index_of_max_y(list):
    max = -float("inf")
    max_index = None
    for i, el in enumerate(list):
        if el[1] > max:
            max = el[1]
            max_index = i
    return max_index


def get_2d_coor(scene, camera, co):
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
        int(scene.render.resolution_x * render_scale),
        int(scene.render.resolution_y * render_scale),
    )
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, camera, co)
    if co_2d.x > 1:
        co_2d.x = 1
    if co_2d.y > 1:
        co_2d.y = 1
    if co_2d.x < 0:
        co_2d.x = 0
    if co_2d.y < 0:
        co_2d.y = 0
    return (
        round(co_2d.x * render_size[0]),
        round(co_2d.y * render_size[1]),
    )


def change_origin_from_bottom_left_to_top_left(coor, height):
    x, y = coor
    return (x, height - y)


def split_into_top_bottom(box):
    bottom_vects = []
    top_vects = []
    for i in range(0, len(box), 2):
        bottom_vects.append(box[i])
    for i in range(1, len(box), 2):
        top_vects.append(box[i])
    return bottom_vects, top_vects


def get_max_rec_from_bottom_and_top_points(bottom_vects, top_vects):
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
    return [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]


def get_annot_chimney(scene, camera, box):
    box_2d = list(map(lambda vector: get_2d_coor(scene, camera, vector), box))
    bottom_vects, top_vects = split_into_top_bottom(box_2d)
    max_rectangle = get_max_rec_from_bottom_and_top_points(bottom_vects, top_vects)
    result = []
    for co in max_rectangle:
        result.append(
            change_origin_from_bottom_left_to_top_left(
                co, scene.render.resolution_y * scene.render.resolution_percentage / 100
            )
        )
    return result


def get_protect_rec(box):
    min_y, min_x = float("inf"), float("inf")
    mid_y_list = []
    max_x = 0
    for point in box:
        x, y = point[0], point[1]
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        mid_y_list.append(y)
    mid_y = sorted(mid_y_list)[(len(mid_y_list) // 2)]
    return [(min_x, min_y), (min_x, mid_y), (max_x, mid_y), (max_x, min_y)]


def get_annot_protection(scene, camera, box):
    box_2d = list(map(lambda vector: get_2d_coor(scene, camera, vector), box))
    max_rectangle = get_protect_rec(box_2d)
    result = []
    for co in max_rectangle:
        result.append(
            change_origin_from_bottom_left_to_top_left(
                co, scene.render.resolution_y * scene.render.resolution_percentage / 100
            )
        )
    return result

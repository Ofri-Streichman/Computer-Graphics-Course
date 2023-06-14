import sys
import numpy as np

global EPSILON
global MAX_DEPTH
EPSILON = 1e-8


def normalize(vector):
    return vector / np.linalg.norm(vector)


def sphere_intersect(center, radius, p0, V):
    b = 2 * np.dot(V, p0 - center)
    c = (np.linalg.norm(p0 - center)) ** 2 - float(radius) ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return sys.maxsize


def plane_intersect(normal, offset, po, v):
    under = (np.dot(v, normal))
    if under != 0:
        t = -(np.dot(po, normal) - offset) / under
        if t > 0:
            return t
    return sys.maxsize


def cubes_intersect(center, length, p0, v):
    t_min = np.zeros(3)
    t_max = np.zeros(3)
    radius = np.array([length / 2, length / 2, length / 2])
    low_bound = center - radius
    high_bound = center + radius
    if v[0] == 0:
        t_min[0] = sys.maxsize
        t_max[0] = sys.maxsize
    else:
        t_min[0] = (low_bound[0] - p0[0]) / v[0]
        t_max[0] = (high_bound[0] - p0[0]) / v[0]

    if t_min[0] > t_max[0]:
        temp = t_min[0]
        t_min[0] = t_max[0]
        t_max[0] = temp
    if v[1] == 0:
        t_min[1] = sys.maxsize
        t_max[1] = sys.maxsize
    else:
        t_min[1] = (low_bound[1] - p0[1]) / v[1]
        t_max[1] = (high_bound[1] - p0[1]) / v[1]

    if t_min[1] > t_max[1]:
        temp = t_min[1]
        t_min[1] = t_max[1]
        t_max[1] = temp

    if t_min[0] > t_max[1] or t_min[1] > t_max[0]:
        return sys.maxsize

    if t_min[1] > t_min[0]:
        t_min[0] = t_min[1]

    if t_max[1] < t_max[0]:
        t_max[0] = t_max[1]

    if v[2] == 0:
        t_min[2] = sys.maxsize
        t_max[2] = sys.maxsize
    else:
        t_min[2] = (low_bound[2] - p0[2]) / v[2]
        t_max[2] = (high_bound[2] - p0[2]) / v[2]

    if t_min[2] > t_max[2]:
        temp = t_min[2]
        t_min[2] = t_max[2]
        t_max[2] = temp

    if (t_min[0] > t_max[2]) or (t_min[2] > t_max[0]):
        return sys.maxsize

    if t_min[2] > t_min[0]:
        t_min[0] = t_min[2]

    if t_max[2] < t_max[0]:
        t_max[0] = t_max[2]

    return t_min[0]


def find_nearest_intersection_object(scene_dict, camera, v):
    min_t = sys.maxsize
    min_primitive = None
    counter = 0
    for obj in scene_dict["sph"]:
        t = sphere_intersect(obj["center"], obj["radius"], camera, v)
        if t < min_t:
            min_t = t
            min_primitive = scene_dict["sph"][counter]
        counter += 1
    counter = 0
    for obj in scene_dict["pln"]:
        t = plane_intersect(obj["normal"], obj["offset"], camera, v)
        if t < min_t:
            min_t = t
            min_primitive = scene_dict["pln"][counter]
        counter += 1
    counter = 0
    for obj in scene_dict["box"]:
        t = cubes_intersect(obj["center"], obj["scale"], camera, v)
        if t < min_t:
            min_t = t
            min_primitive = scene_dict["box"][counter]
        counter += 1
    return min_primitive, min_t


def soft_shadow(light, intersection_point, scene_dict, object):
    # create a squrae perpendicular to the point_to_light vector
    successful_rays = 0
    N = scene_dict["set"]["shadow_rays_num"]
    width = light["radius"]  # width = height
    pos = light["position"]
    vz = normalize(pos - intersection_point)

    # like we found a screen in the first part:
    rotation_matrix = calculate_rotation_matrix(vz)
    matrix = np.array(rotation_matrix)
    vx = np.multiply([1, 0, 0], matrix)[0]
    vy = np.multiply([0, 1, 0], matrix)[:, 1]

    # vx and vy create the plane on which the "light square" sits
    p0 = pos - (width / 2) * vx - (width / 2) * vy

    for i in range(N):
        p = np.copy(p0)
        for j in range(N):
            rand_x = np.random.random() * (width / N)
            rand_y = np.random.random() * (width / N)
            rand_p = p + (vx * rand_x) + (vy * rand_y)
            ray = normalize(
                intersection_point - rand_p)  # construct ray from "light" square to intersection point on object
            shifted_point = rand_p + EPSILON * ray
            nearest, distance = find_nearest_intersection_object(scene_dict, shifted_point,
                                                                 ray)  # find intersection with ray
            distance_to_our_point = np.linalg.norm(shifted_point - intersection_point)

            if nearest is not None and nearest["id"] == object["id"]:
                # make sure the object itself isn't blocking the light from intersection point
                if np.abs(distance - distance_to_our_point) < EPSILON:
                    successful_rays += 1  # ray hit the object

            p += vx * (width / N)
        p0 += vy * (width / N)

    return float(successful_rays) / (N ** 2)


def calculate_rotation_matrix(vz):
    a = vz[0]
    b = vz[1]
    c = vz[2]
    sinx = -b
    cosx = np.sqrt(1 - (sinx ** 2))
    siny = -a / cosx
    cosy = c / cosx
    return [[cosy, 0, siny], [-sinx * siny, cosx, sinx * cosy], [-cosx * siny, -sinx, cosx * cosy]]


def get_color(origin, direction, scene_dict, recursion_level):
    output_color = np.zeros(3)
    diffuse = np.zeros(3)
    specular = np.zeros(3)

    if recursion_level >= MAX_DEPTH:
        return output_color

    # transparency
    background_color = scene_dict["set"]["background_color"]
    nearest_object, t = find_nearest_intersection_object(scene_dict, origin, direction)
    if nearest_object is None:
        # ray didn't hit an object - return background color
        return background_color

    # ray hit an object
    intersection_point = origin + t * direction
    material = scene_dict["mtl"][nearest_object["material"]]
    n = material["phong"]

    trans = material["transparency"]
    if trans > 0:
        # need to calculate the color behind the object
        new_point = intersection_point + direction * EPSILON
        color_behind = get_color(new_point, direction, scene_dict, recursion_level + 1)
        output_color += trans * color_behind

    # calculate the normal going "upwards" from intersection point
    if nearest_object['type'] == "sph":
        normal = normalize(intersection_point - nearest_object['center'])

    elif nearest_object['type'] == "pln":
        normal = normalize(nearest_object["normal"])

    elif nearest_object["type"] == "box":
        normal = np.array([1, 1, 1])
        normals = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        for f in range(6):
            point_center = nearest_object["center"] + nearest_object["scale"] / 2 * normals[f]
            vec = point_center - intersection_point
            if np.abs(np.dot(vec, normals[f]) < EPSILON):
                normal = normals[f]
                break

    shifted_point = intersection_point + EPSILON * normal
    for light in scene_dict["lgt"]:
        pos = light["position"]

        V = normalize(origin - intersection_point)
        VL = normalize(pos - shifted_point)  # vector from shifted intersect to light

        _, min_distance = find_nearest_intersection_object(scene_dict, shifted_point, VL)
        intersection_to_light_distance = np.linalg.norm(VL)

        # VL = normalize(pos - intersection_point)  # vector from intersect to light

        # soft shadow
        shadow_intensity = light["shadow intensity"]
        s_shadow = soft_shadow(light, intersection_point, scene_dict, nearest_object)
        light_intensity = ((1 - shadow_intensity) + (shadow_intensity * s_shadow)) * (
                1.0 / intersection_to_light_distance)

        # diffuse
        diffuse += light["light_color"] * np.dot(VL, normal) * light_intensity

        # specular
        R_vector = 2 * max(0.0, np.dot(VL, normal)) * normal - VL
        specular += light["specular_intensity"] * light["light_color"] * (np.dot(V, R_vector) ** n)

    output_color += (diffuse * material["diffuse_color"] + specular * material["specular_color"]) * (1 - trans)

    # reflection
    reflection = material["reflection_color"]
    if np.linalg.norm(reflection) > 0:
        # need to calculate the reflection
        reflected_V = direction + 2 * max(0.0, np.dot(-direction, normal)) * normal
        illumination = reflection * get_color(shifted_point, reflected_V, scene_dict, recursion_level + 1)
        output_color += illumination

    return output_color


def ray_tracing(scene_dict, width, height):
    image = np.zeros((height, width, 3))
    camera = scene_dict["cam"]["position"]
    ratio = float(height) / width

    look_at_position = scene_dict["cam"]["look_at_position"]
    looking_vector = look_at_position - camera
    distance = scene_dict["cam"]["screen_distance"]

    screen_center = camera + normalize(looking_vector) * float(distance)
    vz = normalize(screen_center - camera)

    rotation_matrix = calculate_rotation_matrix(vz)
    matrix = np.array(rotation_matrix)
    vx = np.multiply([1, 0, 0], matrix)[0]
    vy = np.multiply([0, 1, 0], matrix)[:, 1]

    screen_width = int(scene_dict["cam"]["screen_width"])
    screen_height = screen_width * ratio

    p0 = np.copy(screen_center) - (screen_height / 2) * vy - (screen_width / 2) * vx

    global MAX_DEPTH
    MAX_DEPTH = int(scene_dict["set"]["max_recursions"])

    for i in range(height):

        p = np.copy(p0)
        for j in range(width):
            direction = normalize(p - camera)
            image[height - i - 1, j] = get_color(camera, direction, scene_dict, recursion_level=1)
            p += vx * (screen_width / width)
        p0 += vy * (screen_height / height)

    image = np.clip(a=image * 255, a_min=0, a_max=255)
    return image

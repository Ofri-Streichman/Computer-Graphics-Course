# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import argparse
import numpy as np
import ray_tracing as rt
from PIL import Image


def get_args():
    """A Helper function that defines the program arguments."""
    parser = argparse.ArgumentParser(description='ray tracing algorithm')
    parser.add_argument('--scene_path', type=str, help='The scene file path')
    parser.add_argument('--output_img', type=str, help='The output image path')
    parser.add_argument('--width', type=int, help='The output image width size', default=500)
    parser.add_argument('--height', type=int, help='The output image height size', default=500)
    args = parser.parse_args()
    return args


def main(args):
    """
    The program main function.
    :param args: the command-line input arguments.
    """
    scene_path = args.scene_path
    scene_dict = file_parser(scene_path)
    output_path = args.output_img
    img = rt.ray_tracing(scene_dict, args.width, args.height)
    img = Image.fromarray(np.uint8(img))
    img.save(output_path)


def file_parser(scene_path):
    scene_file = open(scene_path, "r")
    lines = scene_file.readlines()
    # index issues - it should start with index 1 instead of 0!!!!
    scene_dict = {"mtl": [], "sph": [], "pln": [], "box": [], "lgt": []}
    id = 0

    for line in lines:
        if len(line) == 0 or line[0] == '#':
            continue
        category = line[0:3]
        id += 1

        if category == 'cam':
            # camera case
            line_arr = line[7:].split("\t")
            scene_dict["cam"] = {"position": np.array(line_arr[1:4]).astype(float),
                                 "look_at_position": np.array(line_arr[4:7]).astype(float),
                                 "up_vector": np.array(line_arr[7:10]).astype(float),
                                 "screen_distance": line_arr[10],
                                 "screen_width": line_arr[11]}
        if category == 'set':
            # setting case
            line_arr = line[5:].split("\t")
            scene_dict["set"] = {"background_color": np.array(line_arr[1:4]).astype(float),
                                 "shadow_rays_num": int(line_arr[4]),
                                 "max_recursions": int(line_arr[5])}

        if category == 'mtl':
            # material case
            line_arr = line[4:].split("\t")
            scene_dict["mtl"].append({"diffuse_color": np.array(line_arr[1:4]).astype(float),
                                      "specular_color": np.array(line_arr[4:7]).astype(float),
                                      "reflection_color": np.array(line_arr[7:10]).astype(float),
                                      "phong": float(line_arr[10]),
                                      "transparency": float(line_arr[11])})

        if category == 'sph':
            # sphere case
            line_arr = line[4:].split("\t")
            scene_dict["sph"].append({"center": np.array(line_arr[1:4]).astype(float),
                                      "radius": float(line_arr[4]),
                                      "material": int(line_arr[5]) - 1,
                                      "type": "sph",
                                      "id": id})

        if category == 'pln':
            # plane case
            line_arr = line[4:].split("\t")
            scene_dict["pln"].append({"normal": np.array(line_arr[1:4]).astype(float),
                                      "offset": int(line_arr[4]),
                                      "material": int(line_arr[5]) - 1,
                                      "type": "pln",
                                      "id": id})

        if category == 'box':
            # box case
            line_arr = line[4:].split("\t")
            scene_dict["box"].append({"center": np.array(line_arr[1:4]).astype(float),
                                      "scale": float(line_arr[4]),
                                      "material": int(line_arr[5]) - 1,
                                      "type": "box",
                                      "id": id})
        if category == 'lgt':
            # light case
            line_arr = line[4:].split("\t")
            scene_dict["lgt"].append({"position": np.array(line_arr[1:4]).astype(float),
                                      "light_color": np.array(line_arr[4:7]).astype(float),
                                      "specular_intensity": float(line_arr[7]),
                                      "shadow intensity": float(line_arr[8]),
                                      "radius": float(line_arr[9])})
    return scene_dict


if __name__ == '__main__':
    prog_args = get_args()
    main(prog_args)

import numpy as np
import re
import sys
import os
import cv2

def read_semantic_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')

    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

def apply_semantic_colormap(input):
    input = input.astype(np.uint8)
    color_map = {
        0: [255, 120, 50],
        1: [255, 192, 203],
        2: [255, 255, 0],
        3: [0, 150, 245],
        4: [0, 255, 255],
        5: [255, 127, 0],
        6: [255, 0, 0],
        7: [255, 240, 150],
        8: [135, 60, 0],
        9: [160, 32, 240],
        10: [255, 0, 255],
        11: [139, 137, 137],
        12: [75, 0, 75],
        13: [150, 240, 80],
        14: [230, 230, 250],
        15: [0, 175, 0],
        16: [0, 255, 127],
        17: [222, 155, 161],
        18: [140, 62, 69],
        19: [227, 164, 30],
        20: [0, 128, 0]
    }
    # 将字典转换成张量
    lut = np.array([color_map[i] for i in range(21)], dtype=np.uint8)
    output = lut[input]
    return output


def convert_pfm2rgb(pfm_path, save_path, cameras=['camera_FRONT']):

    for cam in cameras:
        print("Processing semantic camera: ", cam)
        seg_file = os.path.join(pfm_path, cam, "seg")
        all_frames = len(os.listdir(seg_file))

        camera_path = os.path.join(save_path, cam)
        os.makedirs(camera_path, exist_ok=True)

        for index in range(0, all_frames):
            pfm_file = os.path.join(seg_file, "{:08}.pfm".format(index))
            semantic, _ = read_semantic_pfm(pfm_file)
            color_map = apply_semantic_colormap(semantic)
            semantic_path = os.path.join(camera_path, "{:08}".format(index) + ".jpg")
            sparse_map = cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR)
            cv2.imwrite(semantic_path, sparse_map)

    print("Convert pfm2rgb Done!")
    return